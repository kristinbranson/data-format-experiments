# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file under a `data/sub-*/` directory (relative to the working directory, `/app`) and processes them sequentially, one file per session. Files are opened directly with `h5py` rather than with `pynwb`, and the needed HDF5 datasets are read explicitly (behavior time series, ROI segmentation table, deconvolved ophys traces). All 152 files (11 subjects) are loaded; subject, session id, scene identifier and brain region are read from the file itself. Trials are then reconstructed inside each file from the behavior streams.

ii.
```python
def get_session_files(sample: bool) -> list[Path]:
    files = sorted(Path("data").glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```
```python
def load_session(path: Path, show_processing: bool) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        identifier = decode_h5_scalar(handle["identifier"])
        subject = decode_h5_scalar(handle["general/subject/subject_id"])
        session_id = decode_h5_scalar(handle["general/session_id"])
        region = decode_h5_scalar(handle["general/optophysiology/ImagingPlane/location"])
        scene_info = parse_scene(identifier)

        behavior = handle["processing/behavior/BehavioralTimeSeries"]
        position = behavior["position/data"][:].astype(np.float32)
        position_t = behavior["position/timestamps"][:].astype(np.float64)
        ...
        segmentation = handle["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        iscell = segmentation["iscell"][:]
        plane_idx = segmentation["planeIdx"][:].astype(np.int16)
        curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
```
```python
    for session_number, path in enumerate(session_files):
        session_data, stats = load_session(path=path, show_processing=show_processing and session_number < 2)
```

iii. From CONVERSION_NOTES Step 2: the AI enumerated the directory and found "152 NWB session files", one folder per subject with one NWB file per session named `sub-mN_ses-XX_behavior+ophys.nwb`, and it documented the full HDF5 layout it reads from. Step 6 justifies `h5py` over `pynwb` as a speedup: "Uses direct HDF5 access with `h5py` instead of slower NWB object loading." Step 4 documents the count reconciliation: 11 mice × 14 days = 154 planned sessions minus the two missing early m11 days = 152, matching the paper's statement that imaging for m11 started on day 3.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB metadata field `general/subject/subject_id` of each session file (not from the directory name). Unique subjects are accumulated in order of first appearance into `data['subjects']`, and `data['subject_idx']` stores the index of the owning subject for every session.

ii.
```python
subject = decode_h5_scalar(handle["general/subject/subject_id"])
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data["subjects"])
    data["subjects"].append(subject)
...
subject_idx.append(subject_to_idx[subject])
...
data["subject_idx"] = np.asarray(subject_idx, dtype=np.int16)
```

iii. CONVERSION_NOTES Step 2 documents "one folder per subject: `sub-m3`, `sub-m4`, … `sub-m19`" and Step 9 confirms 11 subjects with 12 sessions for m11 and 14 for every other mouse, which the AI checked against the paper's "n = 11 mice" and the note that m11's imaging started on day 3. Reading the id from the file rather than the path is treated as the authoritative source.

## 1-c. How are the data split into sessions?

i. One session = one NWB file. Session order is the sorted file order (by subject directory, then session number), and the session label/id comes from `general/session_id`. A session is dropped only if fewer than two trials survive QC (this never happened: all 152 sessions are kept).

ii.
```python
session_id = decode_h5_scalar(handle["general/session_id"])
...
session_label = f"sub-{subject}_ses-{session_id}"
...
if len(session_data["neural_trials"]) < 2:
    print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
    continue

data["neural"].append(session_data["neural_trials"])
data["input"].append(session_data["input_trials"])
data["output"].append(session_data["output_trials"])
```

iii. Step 2 of CONVERSION_NOTES: "Each subject folder contains one NWB file per session, named like `sub-m3_ses-01_behavior+ophys.nwb`". No cross-session neuron registration is attempted; each session's ROIs are treated as an independent population, and `brain_region_idx` is emitted per session. The two-trial minimum is the format requirement stated in the task instructions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the behavior streams: each trial runs from a `trial_start` sample to the first following `teleport` sample, with the teleport sample itself excluded (half-open interval `[trial_start, teleport)`). The stored `trial number` stream is deliberately not used. This yields 12,216 raw trials over the 152 sessions.

ii.
```python
def reconstruct_trials(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0.5)
    teleports = np.flatnonzero(teleport > 0.5)
    trials = []
    tp_ptr = 0

    for start in starts:
        while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
            tp_ptr += 1
        if tp_ptr >= len(teleports):
            break
        stop = teleports[tp_ptr]
        if stop > start:
            trials.append((int(start), int(stop)))
        tp_ptr += 1

    return trials
```
```python
pos_trial = position[start:stop]
speed_trial = speed[start:stop]
lick_trial = lick[start:stop]
time_trial = position_t[start:stop] - position_t[start]
neural_trial = deconvolved[start:stop].T
```

iii. From CONVERSION_NOTES Step 1/4/5: the reference code's `glmUtils.get_timeseries_data` keeps "only samples from `trial_start_inds` to `teleport_inds`", so the AI reconstructs the same window: "Reference code keeps samples from trial start until teleport onset. I will reconstruct trials from `trial_start` impulses to the next `teleport` impulse, excluding teleport frames." Step 10 records the edge case that motivated ignoring the stored trial-number stream: `sub-m11_ses-03` has 81 non-negative trial numbers but only 80 `trial_start`/`teleport` events, so the AI follows `trial_start`/`teleport`, "which is consistent with trial-start alignment."

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) degenerate trials with fewer than 2 samples are skipped; (2) lick-sensor-error trials are dropped, using the paper code's rule — a trial is bad if more than 35% of its samples have a cumulative lick count > 2; (3) a session is dropped if fewer than 2 trials survive. In the full run, 69 of 12,216 trials (0.565%) were dropped for lick-sensor error and no session was dropped, leaving 12,147 trials. Reward omission trials, low-speed samples and short-but-valid trials are all retained.

ii.
```python
LICK_ERROR_FRAC = 0.35  # Reference code threshold in glmUtils.get_timeseries_data.

def is_bad_lick_trial(lick_trial: np.ndarray) -> bool:
    if lick_trial.size == 0:
        return True
    return float(np.mean(lick_trial > 2)) > LICK_ERROR_FRAC
```
```python
if pos_trial.size < 2:
    continue
if is_bad_lick_trial(lick_trial):
    dropped_bad_lick += 1
    continue
```

iii. CONVERSION_NOTES Step 3 quotes the reference code rule ("if >50% of samples have a cumulative lick count of >2" in the comment, `> 0.35` in the actual code) and the paper's exclusion statistic ("~0.65% of all imaged trials, n = 81 out of 12,376"). Step 5 decision 6: "Because lick is a required decoder output and NaNs are undesirable, trials meeting the paper's lick-sensor failure rule will be excluded rather than carried with missing labels." Step 5 decision 8 explains why low-speed samples are *not* dropped even though the paper drops them for spatial analyses: "speed itself is an output class. I will preserve these frames and encode low speed as class `0`." Step 10 compares the realised exclusion rate (69/12,216 = 0.565%) with the paper's 0.654%.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is the NWB-stored `processing/ophys/Deconvolved/plane<i>/data` array (Suite2p's own deconvolved traces, in raw fluorescence units), restricted to curated ROIs. The `Fluorescence` (F) and `Neuropil` (Fneu) arrays present in the same files are **not** used, and no dF/F is computed. For two-plane sessions (m17, m18) the per-plane matrices are re-assembled into a single session matrix in segmentation-table order using `planeIdx`.

ii.
```python
deconv_group = handle["processing/ophys/Deconvolved"]
plane_keys = sorted(deconv_group.keys(), key=lambda key: int(key.replace("plane", "")))
if len(plane_keys) == 1:
    deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
else:
    n_frames = deconv_group[plane_keys[0]]["data"].shape[0]
    n_rois = plane_idx.shape[0]
    all_deconvolved = np.empty((n_frames, n_rois), dtype=np.float16)
    for plane_key in plane_keys:
        plane_number = int(plane_key.replace("plane", ""))
        cols = np.flatnonzero(plane_idx == plane_number)
        plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)
        if plane_data.shape[1] != cols.size:
            raise ValueError(...)
        all_deconvolved[:, cols] = plane_data
    deconvolved = all_deconvolved[:, curated_idx]
```
```python
"neural_signal": "NWB exported deconvolved calcium activity",
```

iii. CONVERSION_NOTES Step 4 states the reasoning: the reference `sess` object has dF/F and `events` added later by `preprocessing.dff`, while "NWB files already store aligned behavior plus `Fluorescence`, `Neuropil`, and `Deconvolved`", so the AI decided to "map `Deconvolved/plane0/data` to reference `events` unless a later check shows mismatch." Step 5 decision 1: "The reference decoder and place-cell pipeline use deconvolved calcium activity (`events`). NWB already exports `Deconvolved`, so this is the closest native match." The planned check of that equivalence ("(1) use NWB `Deconvolved` if it matches reference `events`; (2) recompute dF/F / deconvolution from `Fluorescence` and `Neuropil` if necessary") was never carried out — Step 10's sanity check only verifies that the converted array equals the raw `Deconvolved` array it was copied from.

## 2-b. How is the `neural` data processed?

i. No signal processing at all. The stored deconvolved traces are column-subset to curated ROIs, cast to `float16`, sliced per trial, and transposed to `(n_neurons, n_timepoints)`. There is no neuropil subtraction (`F - 0.7*Fneu`), no per-trial maximin baseline, no dF/F normalisation, no Gaussian smoothing, and no OASIS deconvolution with the paper's `tau = 0.7` / per-plane frame rate. Values therefore stay in raw fluorescence units (0 to ~1.5e4 in the session I inspected) rather than dF/F units.

ii.
```python
deconvolved = np.asarray(deconv_group[plane_keys[0]]["data"][:, curated_idx], dtype=np.float16)
...
neural_trial = deconvolved[start:stop].T
...
neural_trials.append(neural_trial)
```

iii. Step 6 of CONVERSION_NOTES: "Uses NWB `Deconvolved/plane0/data` as neural activity … Stores neural trials as `float16` to reduce output size; training code later casts to `float32`." Step 3 correctly records what the paper does ("dF/F is computed independently within each trial using a maximin baseline with a 20 s window … smoothed with a 2-sample Gaussian … Deconvolved `events` are extracted with OASIS"), but the AI assumed the exported `Deconvolved` array already is that signal. Step 12 attributes the modest decoder accuracies for `lick` and `reward_outcome` to the label representation rather than to the neural signal, and no alternative neural pipeline was tried.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: Suite2p manual curation, `iscell[:, 0] > 0.5`. This keeps 138,678 ROIs (mean 912/session, range 155–2341). The paper's additional exclusion of putative interneurons (cells whose dF/F correlates with running speed at r > 0.5) is **not** applied; the AI tested a proxy version of it on one session using the deconvolved traces, found 0 cells above threshold, and concluded the step was unnecessary.

ii.
```python
iscell = segmentation["iscell"][:]
plane_idx = segmentation["planeIdx"][:].astype(np.int16)
curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)
...
"neuron_filter": "Suite2p iscell[:,0] == 1",
```

iii. Step 5 decision 7: "Initial neuron filter = `iscell[:,0] == 1`: This matches Suite2p manual curation. Additional interneuron-style exclusions remain a consistency check because of the modest count mismatch with the paper." Step 10 check (b): "the converter applies `iscell` and lick-trial QC; reference also mentions a speed-correlation interneuron exclusion. A deconvolved-speed proxy check on the largest session found 0 neurons above `r > 0.5`, so this is not the source of the main count mismatch." Step 4/10 also flag that the data's maximum of 2341 curated ROIs/session exceeds the paper's stated 155–2172 range, which the AI attributes to a dataset-version difference rather than a missing filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond slicing: the ophys frames and the behavior samples are on the same frame index grid (both ~15.5 Hz, one behavior sample per imaging frame), so the same `[start, stop)` frame indices derived from `trial_start`/`teleport` index both the behavior streams and the deconvolved matrix. Each trial therefore begins exactly at the trial-start frame; metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None`.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    time_trial = position_t[start:stop] - position_t[start]
    ...
    neural_trial = deconvolved[start:stop].T
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 4/5 of CONVERSION_NOTES: "Behavioral streams use explicit timestamps; ophys series use `starting_time` with frame rate attribute" and "the reference code aligns Unity VR data to imaging frames before any later analyses", so behavior in the NWB export is already on the imaging-frame grid and no resampling or offsetting is needed. Step 10 records spot checks (session 0 trial 10; multi-plane session 84 trial 5) in which the converted neural trial matched the raw `Deconvolved` rows for the same frame range via `np.allclose(..., atol=1e-3)`, and the processing plots were inspected for temporal offsets between position/lick and neural activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging resolution is kept: 15.5078125 Hz per plane, i.e. a 64.4836 ms bin, stored in `metadata['time_bin_size']`. No rebinning, resampling, smoothing or trial-length padding is applied; trial lengths vary (T from 96 to 3359 samples). The frame rate is hardcoded as a module constant rather than read from each file's `rate` attribute (for the two-plane sessions the stored scanner rate is 31.0156 Hz and the per-plane rate is half that; the hardcoded value is in fact correct for all 152 sessions).

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
"frame_rate_hz": FRAME_RATE_HZ,
```

iii. Step 1: "`multi_anim_sess_README.md` states each sample is one imaging frame at about 15.5 Hz (~64.5 ms/sample)". Step 3 confirms from the methods that "All behavioral and neural time series were sampled at ~15.5 Hz". Step 5 decision 3: "Use native imaging-frame resolution: The reference data are sampled at ~15.5 Hz and behavior is already synchronized to this grid. No rebinning in time unless a later validation forces it." The paper's 10 cm spatial binning is noted as applying to trial matrices, not to the time axis.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the `position` behavior time series' `timestamps` array (all behavior streams in these files share one timestamp vector, so the choice of stream is immaterial).

ii.
```python
position_t = behavior["position/timestamps"][:].astype(np.float64)
...
time_trial = position_t[start:stop] - position_t[start]
```

iii. Step 5's mapping table lists "`…/position` timestamps → `input[0]` = `time_from_trial_start_s`", with the note "Continuous, time-varying, aligned to trial start as required by decoder task". Step 2 records that all behavioral streams carry explicit timestamps on the imaging-frame grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial's first timestamp, so each trial starts at 0 s; stored as float32 and left at native resolution. Range over the full dataset is [0.0, 216.5] s.

ii.
```python
time_trial = position_t[start:stop] - position_t[start]
...
input_trial = np.vstack(
    [
        time_trial.astype(np.float32),
        np.full(n_time, env_code, dtype=np.float32),
        np.full(n_time, float(trial_idx), dtype=np.float32),
        np.full(n_time, float(prev_reward), dtype=np.float32),
    ]
)
```

iii. Step 5 mapping: "`timestamps - timestamps[trial_start]` within each trial"; this is the direct reading of the decoder-task requirement "Time from start of trial in seconds … temporally align based on start of the trial". Step 10's sanity check re-derived time from the raw NWB timestamps for two trials and found an exact match.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No separate alignment step. Behavior and ophys share the frame index grid, so the time vector is built from the same `[start, stop)` slice used for the neural matrix. The input matrix is built with `n_time = neural_trial.shape[1]` for the per-trial constants, so any length disagreement between the neural slice and the behavior slice would raise in `np.vstack` rather than pass silently (the 10 sessions where the ophys array has one extra frame overall are unaffected, because the extra frame is beyond every trial window).

ii.
```python
n_time = neural_trial.shape[1]
input_trial = np.vstack([time_trial.astype(np.float32),
                         np.full(n_time, env_code, dtype=np.float32), ...])
```

iii. Step 4/5: behavior in the NWB export is already aligned to imaging frames by the reference pipeline, so the same indices address both streams. Step 10 verified this on specific trials against the raw file.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the session `identifier` string (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`), parsed into a pre-switch and post-switch environment, combined with the paper's rule that any switch occurs after 30 trials. The `environment` behavior time series is read into a local variable but is never used for the output.

ii.
```python
match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
if match:
    pre_env, pre_zone, post_env, post_zone = match.groups()
    return {"scene": scene, "pre_env": pre_env, "post_env": post_env,
            "pre_zone": pre_zone, "post_zone": post_zone, "switch": True}
```
```python
def zone_for_trial(scene_info: dict, trial_idx: int) -> tuple[str, str]:
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```
```python
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
env_code = float(ENV_TO_INT[env_name])   # ENV_TO_INT = {"Env1": 0, "Env2": 1}
```

iii. Step 4's discrepancy table notes that the NWB `environment` stream "contains valid values `0/1` plus pre-sync invalid `-1`" and resolves to "use valid `environment` values directly from NWB … and cross-check against scene identifier"; the implementation went the other way and took the scene identifier as the authoritative source while using `environment` only conceptually as the cross-check. Step 5 decision 5 justifies scene parsing generally: "NWB `reward_zone` is not the A/B/C label. Active reward-zone identity will be parsed from the session identifier … and the 30-trial switch rule from the paper/code", and the same parse supplies the environment, which matters for the cross-environment scenes (`Env1_B_to_Env2_C`). Step 10 spot-checked `sub-m11_ses-08`: "trial 29 resolves to `(Env1, B)` and trial 30 resolves to `(Env2, C)`, confirming cross-environment switch parsing."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Map `Env1 → 0`, `Env2 → 1`; the value is constant within a trial and repeated across all timepoints of the trial as a float32 row of the input matrix. On cross-environment switch sessions the value changes from the pre- to the post-switch environment at trial index 30.

ii.
```python
ENV_TO_INT = {"Env1": 0, "Env2": 1}
...
env_code = float(ENV_TO_INT[env_name])
...
np.full(n_time, env_code, dtype=np.float32),
```

iii. Step 5 mapping table: "Per-trial constant repeated across frames; valid values only (`0/1`) … encode `ENV1=0`, `ENV2=1`", and decision 4: "Although some decoder variables are per-trial, repeating them across each trial's time axis gives a uniform `(n_variables, n_timepoints)` representation."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The within-session index of the reconstructed trial (the `enumerate` counter over the `trial_start`→`teleport` list), not the stored `trial number` behavior stream. The index is assigned before QC, so dropped lick-error trials leave gaps in the numbering rather than shifting subsequent trials.

ii.
```python
for trial_idx, (start, stop) in enumerate(trials):
    ...
    np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. Step 5 mapping: "Use reconstructed trial order from `trial_start` events, not raw `trial number` during teleport", motivated by the Step 10 edge case in `sub-m11_ses-03` where the stored trial-number stream implies 81 trials but only 80 trial-start/teleport pairs exist. Keeping the pre-QC index also keeps the number consistent with the 30-trial switch rule used for zone/environment.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond casting the 0-based index to float32 and broadcasting it across the trial's timepoints. Observed range in the full dataset is [0, 99].

ii.
```python
np.full(n_time, float(trial_idx), dtype=np.float32),
```

iii. Step 5 mapping: "0-based per-trial index repeated across frames", mirroring the reference code's sample-wise `trials` variable in `glmUtils.get_timeseries_data`.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behavior time series' `timestamps` (reward delivery events, which have their own timestamps rather than being sampled on the frame grid), compared against each trial's start and end *times*. The resulting per-trial reward vector is computed once per session and then shifted by one trial.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```
```python
def reward_outcomes_from_timestamps(reward_times, trial_times, trials) -> np.ndarray:
    outcomes = np.zeros(len(trials), dtype=np.int8)
    reward_ptr = 0
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
    return outcomes
```

iii. Step 5 mapping: "Trial reward delivery from `BehavioralTimeSeries/Reward` … `1` if any reward event timestamp falls within trial, else `0`" and "Uses actual delivered reward, matching decoder task". Step 3 records the paper's design fact used as the consistency target: "Reward was randomly omitted on approximately 15% of trials"; Step 10 reports the realised rate of 0.8466 rewarded / 0.1534 omitted.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *t* the value is the reward outcome of trial *t−1* in the full (pre-QC) reconstructed trial list; the first trial of each session gets 0. Binary (0 = omitted, 1 = rewarded), constant within the trial and repeated across timepoints as float32.

ii.
```python
reward_code = int(reward_outcomes[trial_idx])
prev_reward = int(reward_outcomes[trial_idx - 1]) if trial_idx > 0 else 0
...
np.full(n_time, float(prev_reward), dtype=np.float32),
```

iii. Step 5 mapping: "For trial `t`, use reward outcome of trial `t-1`; first trial defaults to `0`; repeat across frames. Binary, `0=omitted/unrewarded`, `1=rewarded`", which is the decoder-task specification. Because the outcome vector is indexed over all reconstructed trials, the "previous trial" remains the true preceding lap even when that lap was removed by lick QC.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior time series together with the active reward zone for that trial, where the zone identity (A/B/C) comes from the session `identifier` scene string plus the 30-trial switch rule, and the zone's track coordinates are the paper's fixed spans A = 80–130, B = 200–250, C = 320–370 cm. The `reward_zone` behavior stream is not used (the AI judged it to be an occupancy/proximity signal, not a zone label).

ii.
```python
ZONE_COORDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
env_name, zone_name = zone_for_trial(scene_info, trial_idx)
zone_start, zone_end = ZONE_COORDS_CM[zone_name]
...
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. Step 4: "NWB `reward_zone` samples are not A/B/C labels; values increase within-zone and are only meaningful as `>0` occupancy … Parse scene/location from NWB `identifier`; use `reward_zone > 0` only as occupancy / entry signal, and derive A/B/C labels from scene metadata for per-trial zone identity." The coordinates are taken from the paper (Step 3: "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm") and the switch rule from "Each switch occurred after 30 trials". The reference code's `behavior.get_reward_zones` is cited as doing the same thing ("Code infers reward-zone coordinates from scene names like `Env2_LocationA_to_B`").

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the active zone: negative before the zone, exactly 0 while inside it, positive after it. Computed vectorised over the trial's position samples, then discretised (7-c).

ii.
```python
def discretize_distance_to_zone(position_cm, zone_start, zone_end) -> np.ndarray:
    distance = np.where(
        position_cm < zone_start,
        position_cm - zone_start,
        np.where(position_cm > zone_end, position_cm - zone_end, 0.0),
    )
    ...
```

iii. Step 5 mapping: "Signed distance to nearest point in active reward zone: negative before zone, zero inside, positive after zone", which follows the decoder-task wording "Distance to any location in the reward zone" (hence 0 anywhere inside the zone) and the paper's concept of reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks: 0 for < −50 cm, 1 for [−50, −10), 2 for [−10, 0), 3 for exactly 0 (inside the zone), 4 for (0, 10], 5 for (10, 50], 6 otherwise (> 50). Resulting full-dataset fractions: [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
    bins = np.full(distance.shape, 6, dtype=np.int16)
    bins[distance < -50.0] = 0
    bins[(distance >= -50.0) & (distance < -10.0)] = 1
    bins[(distance >= -10.0) & (distance < 0.0)] = 2
    bins[distance == 0.0] = 3
    bins[(distance > 0.0) & (distance <= 10.0)] = 4
    bins[(distance > 10.0) & (distance <= 50.0)] = 5
    return bins
```

iii. The edges are copied from the decoder-task specification in the instructions; class 3 is reserved for "0 cm", i.e. any sample inside the zone. Step 9's consistency table reports the resulting class distribution as an internal check, and the Step 7 processing plots overlay the raw position trace with the zone band and the derived bins to show the discretisation switches at the right moments.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No extra alignment: it is computed from the same `position[start:stop]` slice that indexes the neural matrix, so it is sample-for-sample aligned with the trial's neural frames.

ii.
```python
pos_trial = position[start:stop]
...
neural_trial = deconvolved[start:stop].T
dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
output_trial = np.vstack([dist_bin, pos_bin, speed_bin, lick_bin, ...])
```

iii. Behavior and ophys share the imaging-frame grid (Steps 4–5), and Step 10's raw-file sanity checks confirmed that the converted outputs for specific trials reproduce the raw position-derived values at the same frame indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450 cm virtual corridor), sliced to the trial window.

ii.
```python
position = behavior["position/data"][:].astype(np.float32)
...
pos_trial = position[start:stop]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. Step 5 mapping: "`position` → `output[1]` = `absolute_position_bin` … Use only in-trial frames; exclude teleport", noting from Step 2 that the raw stream also contains pre-sync/teleport sentinel values (−500) that never fall inside a trial window.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is first clipped into [0, 450) (using `np.nextafter` for the open upper end) and then digitised into 5 equal 90 cm bins with edges at 90/180/270/360. Clipping means the handful of samples marginally outside the track are absorbed into the end bins rather than creating extra classes. Full-dataset fractions: [0.212, 0.177, 0.231, 0.226, 0.154].

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, TRACK_START_CM, np.nextafter(TRACK_END_CM, TRACK_START_CM))
    edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. Step 5 decision 9: "Position bins will cover the 450 cm corridor only; teleport frames are excluded, so no teleport-specific bins are needed", with the 450 cm track length taken from the paper (Step 3).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes: 0 for < 90 cm, 1 for [90, 180), 2 for [180, 270), 3 for [270, 360), 4 for ≥ 360 cm — exactly the bins listed in the decoder task.

ii.
```python
edges = np.linspace(TRACK_START_CM, TRACK_END_CM, 6)
return np.digitize(clipped, edges[1:-1], right=False).astype(np.int16)
```

iii. Directly from the instruction's "Discretized into 5 equal-sized bins spanning the 450 cm track"; the AI derives the edges programmatically from the track constants rather than hardcoding them, and verified the resulting near-uniform occupancy distribution in Step 9.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start, stop)` slice as the neural matrix; no resampling or shifting.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconvolved[start:stop].T
```

iii. Same justification as 7-d: shared frame grid, confirmed by the Step 10 raw-file spot checks and the Step 7 processing plots (position ramping from ~0 to the track end within each trial, with no visible offset relative to the binned outputs).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (per-frame cumulative lick counts, integer values 0–6 in these files).

ii.
```python
lick = behavior["lick/data"][:].astype(np.float32)
...
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
```

iii. Step 1 notes the reference code's handling: "licks: `sess.timeseries['licks']`, with values >1 clipped to 1 and invalid sensors masked". Step 5 maps `lick` → `output[3]` with "Clip cumulative lick counts to binary `0/1`; bad-lick trials dropped by QC rule".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation by rounding and clipping to {0, 1}, i.e. any frame with at least one lick becomes 1 (the raw values are integers, so this is identical to a `> 0` test). Trials whose lick channel fails the sensor-error rule are removed entirely (1-e) rather than being NaN-masked as in the reference code. Resulting distribution: 77.7% no-lick / 22.3% lick.

ii.
```python
def binarize_licks(lick_trial: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lick_trial), 0, 1).astype(np.int16)
```

iii. Step 5 decision 6 explains the drop-vs-mask choice ("lick is a required decoder output and NaNs are undesirable"); the binarisation follows both the instruction ("Lick, time-varying. 0 = no, 1 = yes") and the reference code line `licks[licks > 1] = 1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice as the neural data; no shifting or smoothing (the reference code's `nansmooth(licks, 2)` used for its GLM analyses is deliberately not applied, since the target must stay binary).

ii.
```python
lick_trial = lick[start:stop]
lick_bin = binarize_licks(lick_trial)
output_trial = np.vstack([dist_bin, pos_bin, speed_bin, lick_bin, ...])
```

iii. Shared imaging-frame grid (Steps 4–5); Step 7's processing plots show the binarised lick trace against the raw lick channel, and Step 10's raw-file checks reproduced the lick output exactly for the inspected trials.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session `identifier` scene string (e.g. `Env1_LocationB_to_A`, `Env1_LocationA`, `Env1_B_to_Env2_C`), parsed into pre- and post-switch zone letters, with the switch applied at trial index 30. No behavioral stream is used.

ii.
```python
def parse_scene(identifier: str) -> dict:
    scene = identifier.rstrip("/").split("/")[-1]
    match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)      # non-switch day
    ...
    match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)   # within-env switch
    ...
    match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene) # cross-env switch
    ...
    raise ValueError(f"Unrecognized scene format: {scene}")
```
```python
SWITCH_TRIAL = 30
def zone_for_trial(scene_info, trial_idx):
    if scene_info["switch"] and trial_idx >= SWITCH_TRIAL:
        return scene_info["post_env"], scene_info["post_zone"]
    return scene_info["pre_env"], scene_info["pre_zone"]
```

iii. Step 4/5 decision 5: the NWB `reward_zone` stream is an occupancy signal, not a label, so "Active reward-zone identity will be parsed from the session identifier … and the 30-trial switch rule from the paper/code", mirroring the reference code's `behavior.get_reward_zones`. An unrecognised scene string raises rather than silently defaulting. Step 10 records the switch-boundary spot check on `sub-m11_ses-08` (trial 29 → Env1/B, trial 30 → Env2/C), and Step 9 reports the near-uniform zone distribution [0.332, 0.336, 0.332] across the cohort.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Letter → integer mapping A→0, B→1, C→2, constant within a trial and repeated across the trial's timepoints as a row of the output matrix; the same per-trial zone drives the distance-to-zone computation (7-a), keeping the two outputs consistent by construction.

ii.
```python
ZONE_TO_INT = {"A": 0, "B": 1, "C": 2}
...
zone_code = int(ZONE_TO_INT[zone_name])
...
np.full(n_time, zone_code, dtype=np.int16),
```

iii. Step 5 mapping: "Map `A/B/C -> 0/1/2`; repeat across frames in the trial … Derived from NWB `identifier` scene string and switch-after-30 rule", with `output_values` in the saved dictionary naming the classes "A", "B", "C" in that order.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` behavior time series' `timestamps` (discrete delivery events), the same vector used for the previous-trial-outcome input.

ii.
```python
reward_times = behavior["Reward/timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcomes_from_timestamps(reward_times, position_t, trials)
```

iii. Step 5 mapping: "Trial reward delivery from `BehavioralTimeSeries/Reward` … Uses actual delivered reward, matching decoder task", cross-referenced to `behavior.get_trial_types` / `rewardAnalysis.get_reward_inds` in the reference code.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A reward event is attributed to a trial if its timestamp falls in `[t(trial_start), t(teleport))`; the per-trial binary value is then repeated across the trial's timepoints. The event list is walked once per session with a monotone pointer instead of searching per trial. Full-dataset distribution: 15.8% unrewarded / 84.2% rewarded, matching the paper's ~15% omission design.

ii.
```python
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
        outcomes[trial_idx] = int(reward_ptr < len(reward_times) and reward_times[reward_ptr] < stop_t)
```
```python
reward_code = int(reward_outcomes[trial_idx])
...
np.full(n_time, reward_code, dtype=np.int16),
```

iii. Step 5 mapping and Step 10 check 4: "Raw NWB reconstructed reward rate is `0.8466`, omission rate `0.1534`, matching the paper's approximately 15% omission design." Step 12 additionally notes that encoding reward outcome as a trial-constant label (per the instructions) rather than as the reference code's post-delivery state variable limits how well it can be decoded from pre-reward activity.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Degenerate trials**: a trial with `stop <= start` or fewer than 2 samples is skipped; an empty lick slice counts as a bad-lick trial.
- **Trailing trial without a teleport**: `reconstruct_trials` breaks out when the trial-start impulse has no following teleport, so an unterminated final lap is dropped.
- **Extra stored trial numbers**: `sub-m11_ses-03` has 81 non-negative values in the `trial number` stream but only 80 trial-start/teleport pairs; using the impulse streams sidesteps it.
- **Pre-sync sentinel values** (`environment = -1`, `position = -500`, negative trial numbers) never enter the data because only in-trial windows are extracted.
- **Out-of-range positions**: clipped into the track interval before binning.
- **Two-plane sessions**: reassembled from `plane0`/`plane1` via `planeIdx`, with an explicit `ValueError` if the plane column count disagrees with the segmentation table.
- **Sessions too small to evaluate**: dropped if fewer than 2 usable trials remain.
- **Lick-sensor failures**: whole trials removed (see 1-e).

Not handled: there is no check or crop for a mismatch between the number of ophys frames and the number of behavior samples. Ten sessions (m17/m18) have exactly one more ophys frame than behavior sample; this is harmless here because the surplus frame lies past every trial window, but the code neither asserts nor crops, and it also never asserts that the different behavior streams share a timestamp vector.

ii.
```python
        if plane_data.shape[1] != cols.size:
            raise ValueError(
                f"{path.name}: {plane_key} has {plane_data.shape[1]} ROIs but planeIdx maps {cols.size}"
            )
```
```python
    for trial_idx, (start, stop) in enumerate(trials):
        if stop <= start:
            continue
        ...
        if pos_trial.size < 2:
            continue
        if is_bad_lick_trial(lick_trial):
            dropped_bad_lick += 1
            continue
```
```python
        if tp_ptr >= len(teleports):
            break
```
```python
        if len(session_data["neural_trials"]) < 2:
            print(f"  skipping {path.name}: fewer than 2 usable trials after QC")
            continue
```

iii. Step 10's edge-case section documents the `sub-m11_ses-03` trial-count anomaly and the multi-plane loading bug that was found and fixed ("Initial converter assumed a single `plane0` response matrix and failed on `m17/m18`. Fixed by reconstructing a full session matrix from `plane0` and `plane1` using `planeIdx`, then applying `iscell` curation in segmentation-table order"). Step 2 lists the invalid pre-sync encodings that motivated restricting everything to in-trial windows. Step 10 also documents two data-vs-paper mismatches that the AI decided not to "fix" (12,216 reconstructed trials vs the paper's 12,376; 2341 curated ROIs in one session vs the paper's stated max of 2172), attributing both to a dataset-version difference.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identified per-session HDF5 reads of the deconvolved matrices as the dominant cost (I/O bound), followed by writing the 4.5 GB pickle; the per-trial Python work is negligible. Per-session timing is printed so the bottleneck is measurable: 0.2–0.4 s for small single-plane sessions up to ~2.9 s for the largest two-plane sessions, and 182 s (about 3 min) for the whole 152-session conversion — well inside the 15-minute budget in the instructions.

ii.
```python
    t0 = time.perf_counter()
    for session_number, path in enumerate(session_files):
        session_t0 = time.perf_counter()
        session_data, stats = load_session(...)
        elapsed = time.perf_counter() - session_t0
        print(f"[{session_number + 1:03d}/{len(session_files):03d}] ... time={elapsed:.2f}s")
    total_elapsed = time.perf_counter() - t0
    print(f"Processed {len(data['neural'])} sessions in {total_elapsed:.2f}s")
```

iii. Step 6/7 of CONVERSION_NOTES: "Full-session deconvolved activity is currently loaded into memory per session before trial slicing"; speedups claimed are direct `h5py` access instead of `pynwb` object loading, reading only curated ROI columns, and `float16` storage ("roughly halves pickle size and write bandwidth"). Step 7 extrapolated from a large multi-plane benchmark (2.49 s/session) to a conservative ~6.3 min upper bound; the realised 182 s came in under that estimate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. All per-sample arithmetic is already vectorised (distance, position, speed, lick discretisation are pure NumPy over whole-trial arrays). The remaining Python loops are: the per-trial loop in `load_session`, the pointer loops in `reconstruct_trials` and `reward_outcomes_from_timestamps` (both could be replaced by `np.searchsorted` on the teleport/reward index arrays), and the per-plane loop for two-plane sessions. All operate on ~100 trials or ~80 reward events per session, so their cost is negligible next to the HDF5 reads; the AI did not flag them and did not vectorise them.

ii.
```python
    for start in starts:
        while tp_ptr < len(teleports) and teleports[tp_ptr] <= start:
            tp_ptr += 1
```
```python
    for trial_idx, (start, stop) in enumerate(trials):
        start_t = trial_times[start]
        stop_t = trial_times[stop]
        while reward_ptr < len(reward_times) and reward_times[reward_ptr] < start_t:
            reward_ptr += 1
```
```python
    for trial_idx, (start, stop) in enumerate(trials):
        pos_trial = position[start:stop]
        ...
        dist_bin = discretize_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The AI's efficiency notes (Step 6) list only two inefficiencies — "Full-session deconvolved activity is currently loaded into memory per session before trial slicing" and "No parallel file processing yet" — and its speedup work targeted I/O rather than loop vectorisation, which is consistent with its measurement that the run is I/O bound. Trials have variable length, so the per-trial loop cannot be collapsed without padding.

## 13-c. What processing does the code repeat multiple times?

i. Very little: each NWB file is opened exactly once and each array is read once, in a single pass (no separate survey/statistics pass). The per-trial constants (environment, trial number, previous outcome, zone, reward) are re-materialised as full-length rows for every trial, which duplicates values across time, but that is the storage format the instructions ask for. In `--show-processing` mode `binarize_licks` is recomputed for the plotted fallback trial, and the plotting path re-reads already-sliced arrays; neither runs in the default full conversion.

ii.
```python
    for session_number, path in enumerate(session_files):
        session_data, stats = load_session(path=path, show_processing=show_processing and session_number < 2)
```
```python
            input_trials, output_trials = ...,
            raw_examples=raw_examples or [
                {
                    "trial_index": 0,
                    "position": position[trials[0][0]:trials[0][1]],
                    "speed": speed[trials[0][0]:trials[0][1]],
                    "lick": binarize_licks(lick[trials[0][0]:trials[0][1]]),
                    ...
                }
            ],
```

iii. Step 6: the design goal was "Avoid unnecessary file I/O" and "Processes sessions sequentially to keep peak memory bounded by one session"; because the reward-zone identity is read from the file's own identifier string rather than inferred from pooled statistics, no pre-pass over the dataset is needed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three small items:
- The `environment` behavior stream is read from every file into a local variable and never used — the environment input is derived from the scene string instead.
- For two-plane sessions the code reads **all** ROI columns of both planes and converts the whole matrix to `float16` before subsetting to curated cells, so roughly half the data read and converted is thrown away. This contradicts the Step 6 claim that the converter "Reads only curated ROI columns from the deconvolved matrix", which holds only for single-plane sessions.
- `iscell[:, 1]` (the Suite2p classifier probability) and the `Fluorescence`/`Neuropil` arrays are present but unused; `brain_region_idx` is computed inside `load_session` and then recomputed in `build_dataset`.

ii.
```python
        environment = behavior["environment/data"][:].astype(np.float32)   # never used again
```
```python
            plane_data = np.asarray(deconv_group[plane_key]["data"], dtype=np.float16)  # all ROIs
            all_deconvolved[:, cols] = plane_data
        deconvolved = all_deconvolved[:, curated_idx]                       # then discarded
```
```python
        curated_idx = np.flatnonzero(iscell[:, 0] > 0.5)
        brain_region_idx = np.zeros(curated_idx.size, dtype=np.int16)       # unused local
...
        data["brain_region_idx"].append(
            np.full(session_data["stats"]["n_neurons_kept"], region_to_idx[region], dtype=np.int16)
        )
```

iii. The AI did not document any of these; its stated efficiency decisions (Step 6) were direct `h5py` access, reading only curated ROI columns, `float16` storage, and sequential session processing. In practice the waste is small relative to the 182 s total runtime, and the unused `environment` read reflects the mid-course decision to take environment identity from the scene identifier instead of the time series.
