# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data` (152 files, 11 subjects) and opens each one directly with `h5py` rather than `pynwb`, reading HDF5 paths explicitly (`processing/behavior/BehavioralTimeSeries/...`, `processing/ophys/...`). Every file is converted; nothing is skipped at the file level. Behavior channels are read whole into memory; the neural matrix is left on disk and sliced per trial. Subjects/sessions are parsed from the directory and file names. A second, smaller "sample" pickle is also written from 8 hand-picked sessions for fast checks.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths

def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse session/day from {path.name}")
    return subject, int(match.group(1))
```
```python
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]
        times = beh["position"]["timestamps"][()].astype(np.float64)
        pos = beh["position"]["data"][()].astype(np.float32)
        ...
```
```python
    for idx, path in enumerate(files, start=1):
        record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
        session_records.append(record)
```

iii. The AI first opened a file with `pynwb` (trajectory steps 23-30) and found the top-level `trials` and `units` tables absent, so it concluded it had to reconstruct trials from framewise behavior channels. It then switched to raw `h5py` access for speed/robustness. Its notes report the recovered structure as a sanity check: "Unique subjects recovered: 11; Sessions recovered: 152; Session counts by mouse: m11=12, all other mice =14 ... The reduced session count for m11 matches the methods description that imaging for m11 started on day 3."

## 1-b. How are the data split into subjects?

i. One subject per `sub-<id>` directory; the id is taken from the parent directory name of each NWB file. `subjects` is the sorted set of ids and `subject_idx` maps each session to its subject.

ii.
```python
subject = path.parent.name.replace("sub-", "")
...
subjects = sorted({record["subject"] for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_to_idx[record["subject"]] for record in session_records], dtype=np.int64),
```

iii. The AI verified that 11 subjects are recovered, matching the paper's 11 imaged mice, and that session counts per mouse (14, except m11 with 12) match the methods.

## 1-c. How are the data split into sessions?

i. One session per NWB file; the day index is parsed from the `ses-<NN>` token in the file name and stored in `session_info['day']`. No cross-session cell registration is attempted (each session's neurons are treated independently).

ii.
```python
match = re.search(r"ses-(\d+)", path.name)
...
return subject, int(match.group(1))
...
session_info["day"] = day  # via session_summary_record(...)
```

iii. The AI used the day index to verify the experimental schedule it recovered ("11 day-8 environment switches as expected"; `print_subject_schedules` prints `d01:ENV1:A | d03:ENV1:A->B | ...` per mouse), which it treated as evidence that the file-name session parsing is correct.

## 1-d. How are the data split into trials?

i. Trials are reconstructed framewise: start = frames where the `trial_start` channel is non-zero, end = frames where the `teleport` channel is non-zero. Each trial is `[start, stop)` — trial_start frame inclusive, teleport frame exclusive. The code hard-fails if the number of starts and teleports differ.

ii.
```python
trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
if trial_starts.size != teleports.size:
    raise ValueError(
        f"{path.name}: trial starts ({trial_starts.size}) and teleports ({teleports.size}) do not match"
    )
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    T = int(stop - start)
```

iii. From the trajectory: "the top-level NWB `trials` and `units` tables are absent, so the converter has to reconstruct trials from the framewise `trial_start` and `teleport` channels exactly as the paper's code does." The notes add that this "matches the reference preprocessing logic that keeps on-track frames and excludes the teleport sample itself" (the paper's code indexes `sess.trial_start_inds` / `sess.teleport_inds`). The AI checked that the resulting mean of 79.9 trials/session is consistent with the paper's 80.5 ± 7.4.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, only the first of which removes anything in practice:
- **Lick-sensor error**: a trial is dropped if more than 35% of its frames have a cumulative lick count > 2 (69 of 12,216 trials dropped).
- Trials with `T <= 0`.
- Trials whose inferred reward-zone or environment label is `None`.
No minimum trial-length filter is applied, and no sessions or mice are dropped.

ii.
```python
def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    drop_mask = np.zeros(len(trial_starts), dtype=bool)
    error_fraction = np.zeros(len(trial_starts), dtype=np.float32)
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        lick_trial = lick[start:stop]
        frac = float(np.mean(lick_trial > 2))
        error_fraction[idx] = frac
        if frac > threshold:
            drop_mask[idx] = True
    return drop_mask, error_fraction
```
```python
    if drop_mask[trial_idx]:
        dropped_trials.append(trial_idx); continue
    T = int(stop - start)
    if T <= 0:
        dropped_trials.append(trial_idx); continue
    zone_label = zone_schedule["filled"][trial_idx]
    env_label = env_schedule["filled"][trial_idx]
    if zone_label is None or env_label is None:
        dropped_trials.append(trial_idx); continue
```

iii. The AI located `correct_lick_sensor_error` in the paper's `behavior.py` (`if sum(licks[t_start:t_end] > 2)/len(licks[t_start:t_end]) > correction_thr`) and the `lick_correction_thr = 0.35` used in `dayData.py` / the figure notebooks. Its notes state: "The decoder format cannot represent those trials with NaN licks while keeping all arrays consistent, so the converter drops such trials entirely. This is the main trial-quality filter beyond basic structural validity."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrix is the stored NWB `processing/ophys/Deconvolved/plane0/data` array (Suite2p's own deconvolution of raw fluorescence), restricted to `iscell` ROIs of plane 0. The `Fluorescence` (F) and `Neuropil` (Fneu) arrays are read only to obtain the plane-0 ROI index list; they are never used to compute a signal. The `Deconvolved/plane1` series (present in all m17/m18 files) is not used.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```
```python
"neural_signal": "Deconvolved calcium events from processing/ophys/Deconvolved/plane0",
```

iii. The AI's notes justify this as: "The paper states that analyses used deconvolved calcium event time series sampled at the imaging frame rate, approximately 15.5 Hz. The converter therefore uses `processing/ophys/Deconvolved/plane0/data`." It had read the Methods paragraph describing the paper's own dF/F + OASIS pipeline (maximin baseline, neuropil subtraction, Gaussian smoothing) and had read `preprocessing.dff` in the paper's code (trajectory step 39), but it never compared the stored `Deconvolved` series against a re-implementation and did not record a reason for preferring the stored array.

## 2-b. How is the `neural` data processed?

i. No processing at all. The stored deconvolved values are sliced per trial, transposed to (n_neurons, n_timepoints) and cast to `float32`. There is no neuropil subtraction, no per-trial maximin baseline, no dF/F normalization, no Gaussian smoothing, no OASIS deconvolution, no z-scoring, and no per-cell normalization.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
neural_trials.append(neural)
```

iii. Notes: "No extra temporal smoothing or rebinning was added." The implicit justification is that the NWB `Deconvolved` series is already the paper's analysis-level signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: Suite2p's manual-curation `iscell` flag, applied to the ROIs of imaging **plane 0 only**. For the two multi-plane mice (m17, m18) all plane-1 cells are discarded. The paper's additional exclusion of putative interneurons (dF/F correlated with running speed) is deliberately not implemented. No neurons are otherwise excluded; all neurons are labelled brain region `CA1`.

ii.
```python
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]            # rois = Fluorescence/plane0/rois
selected_indices = np.flatnonzero(selected_mask)
plane_idx = f["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][()][selected_rois]
...
"brain_regions": ["CA1"],
"brain_region_idx": [record["brain_region_idx"] for record in session_records],
```
```python
"roi_selection": (
    "ROIs referenced by processing/ophys/Fluorescence/plane0/rois and retained "
    "only if the corresponding PlaneSegmentation iscell flag is true."
),
```

iii. Two justifications are recorded.
- Interneurons: at trajectory step 41 the AI correlated `F - 0.7*Fneu` with speed for the first 200 `iscell` cells of one session and found `frac>0.5 = 0.0`, concluding (step 44) "If it's negligible relative to the `iscell` curation, I can keep the converter simpler."
- Planes: the notes claim "Their segmentation tables contain rows from two imaging planes, but the published response series available in NWB is only the linked `plane0` series. Using all segmentation rows would overcount neurons and break alignment with the stored activity matrix. Restricting to linked `plane0` ROIs gives neuron counts much closer to the paper's reported range." (Step 56: "the stored response matrix only appears to include one ROI response series named `plane0`.")

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the trial slice begins exactly at the `trial_start` frame, so sample 0 of every neural trial is the alignment event. Metadata records `temporal_alignment_event = "trial start ..."`, `off_start = 0.0`, `off_end = None` (trials have variable length, ending at teleport).

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
"temporal_alignment_event": "trial start (entry to the start of the 450 cm linear track)",
"off_start": 0.0,
"off_end": None,
"trial_window_definition": "Frames from trial_start inclusive to teleport exclusive.",
```

iii. Behavioral and imaging streams are already sampled on the same frame clock in the NWB files, so the same frame index range defines the trial for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. Data are kept at the native imaging/behavior frame rate. The bin size is computed per session as the median difference of behavior timestamps and the dataset-level `time_bin_size` is the median across sessions: 64.48362720402656 ms (~15.5 Hz per plane). (Verified: this value is identical in all 152 sessions.)

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
time_bin_sizes_ms = [1000.0 * info["time_bin_size_sec"] for info in session_info]
...
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. Notes: "No extra temporal smoothing or rebinning was added. The recovered bin size is constant across all sessions: 64.48362720402656 ms." The AI also noted (step 56) that although the NWB metadata reports 31 Hz acquisition for the two-plane mice, "the behavior timestamps are still 64.48 ms apart", i.e. the per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` of the `position` behavioral time series.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. Not explicitly justified; all behavior channels in these files share a single timestamp vector, so the choice of channel is immaterial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial's first timestamp, so every trial starts at 0 s; stored as `float32` and as the first row of the input array.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
input_trial = np.vstack([time_trial, ...])
```

iii. Directly implements the requested "time from start of trial in seconds".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same frame index range `[start, stop)` slices both the timestamp vector and the neural matrix, so the streams share sample indices. No offset, interpolation or resampling is applied, and no explicit check that the neural array and behavior arrays have the same length is performed.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. Implicit: imaging and VR streams are already synchronized frame-by-frame in the published NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
...
env_trial = environment[start:stop]
env_trial = env_trial[env_trial >= 0]
observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
```

iii. Notes: environment is inferred "from the framewise `environment` channel"; the recovered schedule "correctly recovers the day-8 environment switch and the fact that `m17` and `m18` start in `ENV2`", which the AI used as validation. Metadata records `{"0": "ENV1", "1": "ENV2"}`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, negative (invalid) samples are dropped, the median is taken and rounded to an integer, giving one scalar per trial. That per-trial value is then passed through a schedule-inference step: if the modal value over trials 0-29 differs from the modal value over trials 30+, every trial is *relabelled* as pre-mode (trials < 30) or post-mode (trials >= 30); otherwise every trial gets the session-wide mode. The resulting label is broadcast across all timepoints of the trial as input row 1. (Verified: this imputation never changes an observed label — the filled labels equal the per-trial observed medians for all 12,216 trials.)

ii.
```python
def infer_constant_or_switch_schedule(observed_values, switch_trial=30):
    ntrials = len(observed_values)
    pre = mode_or_none(observed_values[: min(switch_trial, ntrials)])
    post = mode_or_none(observed_values[switch_trial:]) if ntrials > switch_trial else None
    overall = mode_or_none(observed_values)
    is_switch = pre is not None and post is not None and pre != post
    filled = []
    if is_switch:
        for trial_idx in range(ntrials):
            filled.append(pre if trial_idx < switch_trial else post)
    else:
        filled = [overall for _ in range(ntrials)]
    ...
```
```python
env_schedule = infer_constant_or_switch_schedule(observed_env)
...
env_label = env_schedule["filled"][trial_idx]
input_trial = np.vstack([time_trial, np.full(T, env_label, dtype=np.float32), ...])
```

iii. Notes: "Environment identity is inferred the same way from the framewise `environment` channel, again using the 30-trial switch rule when needed." The 30-trial rule comes from the Methods, which the AI read: "Each switch occurred after 30 trials", and day 8's zone switch "coincided with a switch into the novel environment".

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The recorded `trial number` behavioral channel (not the loop counter), with the loop index only as a fallback if the channel contains no valid samples in a trial.

ii.
```python
trial_number = beh["trial number"]["data"][()].astype(np.float32)
...
trial_number_values = trial_number[start:stop]
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = (
    float(np.nanmedian(trial_number_values)) if trial_number_values.size else float(trial_idx)
)
```

iii. Step 76: "read the per-trial number from the recorded channel rather than assuming sequential indexing", i.e. prefer the recorded ground truth over an assumption. (Verified: the channel's per-trial median equals the sequential index for all 12,216 trials, so the two are identical in practice.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Negative sentinel samples are dropped, the median over the trial is taken as a scalar, and that scalar is broadcast across all timepoints as input row 2. Values are not renumbered after trials are dropped (original within-session numbering is preserved), and the value is not normalized; the range is 0-99.

ii.
```python
np.full(T, trial_number_scalar, dtype=np.float32)
```

iii. Same as 5-a; the median makes the extraction robust to stray samples at the trial boundaries.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` time series' `timestamps` (reward-delivery events), compared against the behavior timestamps at each trial's boundaries.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```

iii. Notes: "Per-trial reward outcome is computed from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` within each reconstructed trial window." Reward events carry their own timestamps rather than being a framewise channel, so a time-window test is used instead of frame indexing.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward outcome vector (computed over **all raw trials**, before any trial dropping) is shifted by one; the first trial of a session is assigned 0. The scalar is broadcast across all timepoints as input row 3. Because the shift is over raw trial indices, a trial's "previous outcome" still refers to the trial that actually preceded it even if that trial was later dropped.

ii.
```python
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
...
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. Follows the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"; the first trial has no predecessor, so 0 (omitted) is used.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` channel plus a per-trial reward-zone label. The label is inferred from the `reward_zone` channel: the median position over the frames where `reward_zone > 0` is matched to the nearest of the three fixed zone centres (A 80-130, B 200-250, C 320-370 cm). On trials with no `reward_zone` frames, the animal's position at the first reward timestamp is used instead; if neither is available the trial's label is `NaN`/`None` and is filled by the pre/post-trial-30 modal schedule (same `infer_constant_or_switch_schedule` used for environment).

ii.
```python
ZONE_BOUNDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_CENTERS_CM = {label: 0.5 * (bounds[0] + bounds[1]) for label, bounds in ZONE_BOUNDS_CM.items()}

def infer_zone_from_position(zone_position_cm):
    if not np.isfinite(zone_position_cm):
        return None
    return min(ZONE_CENTERS_CM, key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]))

def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    for start, stop in zip(trial_starts, teleports):
        pos_trial = pos[start:stop]
        zone_frames = np.flatnonzero(reward_zone_signal[start:stop] > 0)
        zone_position_cm = np.nan
        if zone_frames.size > 0:
            zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
        else:
            in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
            if in_trial_reward.size > 0:
                reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
                reward_idx = min(max(reward_idx, start), stop - 1)
                zone_position_cm = float(pos[reward_idx])
        observed_positions.append(zone_position_cm)
        observed_labels.append(infer_zone_from_position(zone_position_cm))
```
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
zone_label = zone_schedule["filled"][trial_idx]
```

iii. Notes: "The paper defines the hidden 50 cm reward zones as A: 80-130 cm, B: 200-250 cm, C: 320-370 cm. The converter infers the reward-zone label of each trial from the framewise `reward_zone` channel and animal position. On omission trials where the zone may not be directly observed, the converter fills missing labels using the dominant pre-switch and post-switch labels with the paper's 30-trial switch boundary." The AI validated the result by printing the recovered per-mouse day-by-day zone schedule (`d01:ENV1:A | d03:ENV1:A->B | ...`) and checking it against the paper's described A->B->C->A schedule with 77 switch sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to [0, 450] cm, then the signed distance to the nearest edge of the trial's reward zone is computed: `pos - zone_start` before the zone (negative), `pos - zone_end` after the zone (positive), exactly 0 while inside the zone.

ii.
```python
def signed_distance_to_zone(pos_cm, zone_bounds_cm):
    zone_start, zone_end = zone_bounds_cm
    dist = np.zeros_like(pos_cm, dtype=np.float32)
    before = pos_cm < zone_start
    after = pos_cm > zone_end
    dist[before] = pos_cm[before] - zone_start
    dist[after] = pos_cm[after] - zone_end
    return dist
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. Directly implements "distance to any location in the reward zone" (0 anywhere inside the zone) from the instructions, using the paper's zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned with explicit boolean masks: 0: < -50; 1: [-50, -10); 2: [-10, 0); 3: exactly 0; 4: (0, 10]; 5: (10, 50]; 6: > 50 cm.

ii.
```python
def bin_distance_to_zone(dist_cm):
    out = np.zeros(dist_cm.shape, dtype=np.int64)
    out[dist_cm < -50.0] = 0
    out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
    out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
    out[dist_cm == 0.0] = 3
    out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
    out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
    out[dist_cm > 50.0] = 6
    return out
```
Names are recorded in `output_values[0]` as "< -50 cm", "-50 to -10 cm", "-10 cm to < 0 cm", "0 cm", "> 0 cm to +10 cm", "+10 to +50 cm", "> +50 cm".

iii. The bin edges are copied verbatim from the Decoder Task specification in the instructions; the zero class is kept exact so that "in the reward zone" is its own category.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame index range `[start, stop)` as the neural slice; position, licks, speed and the neural matrix are all sliced identically, so the output is sample-for-sample aligned and has the same `T`.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
output_trial = np.vstack([bin_distance_to_zone(dist_trial), ...])
```

iii. Implicit: all streams share the NWB frame clock, so index-based slicing preserves alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series (VR corridor position in cm).

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Not separately justified; `position` is the recorded track position and metadata records `position_range_cm: [0.0, 450.0]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the [0, 450] cm track (the raw channel runs slightly outside the track, roughly -1 to +452 cm) and then floor-divided into 90 cm bins. Note the same clipped position is reused for the distance-to-zone output.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The 450 cm track length is from the Methods; clipping ensures the 5 requested equal-sized bins cover every sample rather than creating out-of-range classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins: 0: [0, 90); 1: [90, 180); 2: [180, 270); 3: [270, 360); 4: [360, 450], with anything outside the track folded into the end bins by the clip.

ii.
```python
return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
...
["0 to <90 cm", "90 to <180 cm", "180 to <270 cm", "270 to <360 cm", "360 to 450 cm"]
```

iii. Directly implements "Discretized into 5 equal-sized bins spanning the 450 cm track" from the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start, stop)` frame slice as the neural data; no offset or resampling.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
output_trial = np.vstack([..., bin_absolute_position(pos_trial), ...])
```

iii. As for 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series (per-frame lick counts, integer values 0-6 in these files).

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. Not separately justified beyond the instruction that lick is a binary time-varying output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization: any frame with a lick count > 0 becomes 1, else 0. No smoothing or lick-rate conversion. Trials whose lick channel looks like a capacitive-sensor failure (>35% of frames with count > 2) are removed entirely rather than being binarized (see 1-e).

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
...
["no", "yes"]
```

iii. Instructions specify "Lick, time-varying. 0 = no, 1 = yes"; the notes explain that the paper's `correct_lick_sensor_error` NaN-masking was replaced by dropping the trial because the dense array format cannot carry NaNs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start, stop)` frame slice as the neural data.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. As for 7-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Identical to 7-a: the `reward_zone` channel (plus `position`, and `Reward` timestamps as fallback), condensed to one A/B/C label per trial and regularized by the pre/post-trial-30 modal schedule.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
zone_label = zone_schedule["filled"][trial_idx]
```

iii. See 7-a.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The A/B/C label is mapped to 0/1/2 and broadcast across every timepoint of the trial (time-varying representation of a per-trial variable), as output row 4.

ii.
```python
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
...
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
...
["A", "B", "C"]
```

iii. Notes: "`reward_zone_location` and `reward_outcome` are trial-level values repeated across all time bins in that trial, which is the simplest format compatible with the decoder", consistent with the instruction "If at all possible, make it time-varying".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' timestamps, tested against each trial's start/end times (the same computation used for previous-trial outcome).

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
```

iii. See 6-a.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial (1 if any reward timestamp falls in `[t_start, t_teleport)`), broadcast across all timepoints as output row 5. The dataset-level rewarded fraction is 0.842 / omission 0.158.

ii.
```python
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
...
np.full(T, reward_outcomes[trial_idx], dtype=np.int64)
```

iii. Notes: the recovered omission rate (15.8%) is checked against the paper's "approximately 15%" omission rate, which the AI used as validation of the reward extraction.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing reward-zone / environment observations on a trial** (e.g. omission trials where the mouse never triggers the zone): filled from the pre/post-trial-30 modal schedule; if still unresolvable the trial is dropped.
- **Noisy per-trial labels**: medians (not single samples) are used for environment, trial number and zone position; negative sentinel values (`< 0`) in the `environment` and `trial number` channels are discarded before taking the median.
- **Lick-sensor failures**: whole trial dropped (see 1-e).
- **Degenerate trials** (`T <= 0`): dropped.
- **Structural inconsistency**: a `ValueError` is raised if the trial_start and teleport counts differ in a file, or if a session ends up with no valid trials.
- **Out-of-range position**: clipped into [0, 450] cm.
Not handled: there is no check that the neural matrix and the behavior channels have the same number of frames. (In fact 10 sessions — m17 ses-04/06 and eight m18 sessions — have one more imaging frame than behavior samples. Because the imaging array is the longer one and trials are indexed from the behavior channels, this is harmless here, but the code would silently emit short neural trials if the mismatch ever went the other way.)

ii.
```python
env_trial = env_trial[env_trial >= 0]
observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
...
trial_number_values = trial_number_values[trial_number_values >= 0]
...
if trial_starts.size != teleports.size:
    raise ValueError(f"{path.name}: trial starts ... do not match")
...
if not neural_trials:
    raise ValueError(f"{path.name}: no valid trials remained after filtering")
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. These are defensive checks the AI added while exploring the files; the notes justify only the label filling ("On omission trials where the zone may not be directly observed, the converter fills missing labels using the dominant pre-switch and post-switch labels with the paper's 30-trial switch boundary") and the lick-error drops. The session-level diagnostics (`trial_indices_dropped`, `observed_zone_positions_cm`, `lick_error_fraction_per_trial`) are written into `metadata['session_info']` so the decisions are auditable after the fact.

## 13-a. What are the most time-consuming steps of the code?

i. The code is dominated by I/O: (1) per-trial HDF5 reads of the neural matrix — `deconv[start:stop, selected_indices]` is a fancy (point-selection) read issued once per trial, i.e. ~12,200 separate HDF5 selections over ~150 large compressed datasets, instead of one bulk read per session; (2) reading the full behavior channels of all 152 files; (3) pickling the 8.2 GB `converted_data.pkl` plus a further 473 MB `sample_data.pkl`. The per-trial CPU work (binning, medians) is trivial by comparison. Each NWB file is opened only once.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)   # inside the per-trial loop
...
def save_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)
```

iii. The AI gave no analysis of runtime cost anywhere in its notes or trajectory; it did keep the neural array as a lazy `h5py` dataset rather than reading it whole, which limits peak memory at the cost of many small reads.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. `infer_trial_zone_positions`, the `observed_env` loop, `reward_outcomes_per_trial` and `drop_lick_error_trials` are four separate Python loops over the same trial boundaries, each re-slicing the same arrays; they could be fused into one pass or vectorized with `np.add.reduceat`/`np.searchsorted` over the trial edges. `reward_outcomes_per_trial` in particular is an O(n_trials x n_rewards) comparison that `np.searchsorted` would do in one call. The main conversion loop is inherently per-trial because trials have different lengths, but the binning functions inside it (`bin_distance_to_zone`, `bin_absolute_position`, `bin_speed`) could be applied once to the whole session. All the binning helpers are already vectorized over time.

ii.
```python
for start, stop in zip(trial_starts, teleports):   # infer_trial_zone_positions
for start, stop in zip(trial_starts, teleports):   # observed_env
for start, stop in zip(trial_starts, teleports):   # reward_outcomes_per_trial
for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):  # drop_lick_error_trials
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):  # main conversion
```

iii. No justification was recorded; the loops appear to have been written incrementally as each derived variable was added.

## 13-c. What processing does the code repeat multiple times?

i. Within a session, the trial-boundary loop is repeated five times over the same `(trial_starts, teleports)` pairs, and `pos[start:stop]` / `lick[start:stop]` are sliced in more than one of them. Across the run, `build_dataset` and `summarize_dataset` are executed twice (once for the full dataset and once for the sample), and the eight sample sessions' trials are serialized twice, once into each pickle. The NWB files themselves are each read only once — the conversion is a single pass over the data.

ii.
```python
full_data = build_dataset(session_records)
full_summary = summarize_dataset(full_data)
...
sample_records = select_sample_records(session_records)
sample_data = build_dataset(sample_records)
sample_summary = summarize_dataset(sample_data)
...
save_pickle(args.full_out, full_data)
save_pickle(args.sample_out, sample_data)
```

iii. The duplicated sample output is deliberate: the notes present `sample_data.pkl` as a "smaller representative subset for fast checks" used to validate format and decoder training before committing to the full run.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed and stored but never used by the decoder: the 473 MB `sample_data.pkl` (a duplicate of 8 sessions); the large `metadata['session_info']` block, which carries per-session per-trial diagnostics (`observed_zone_positions_cm`, `observed_zone_labels`, `observed_environment`, `reward_outcomes_all_trials`, `previous_reward_outcomes_all_trials`, `lick_error_fraction_per_trial`, `trial_indices_kept/dropped`); `plane_idx`, which is read and summarized only for metadata; and the `Fluorescence`/`Neuropil` ROI lookup, of which only the plane-0 `rois` index vector is actually used. Conversely, the paper's full dF/F pipeline is *not* computed at all, so nothing expensive is wasted there.

ii.
```python
plane_idx = f["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][()][selected_rois]
...
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
...
save_pickle(args.sample_out, sample_data)
```

iii. No explicit justification; the notes treat the extra metadata and the sample pickle as auditability and validation aids rather than decoder inputs.
