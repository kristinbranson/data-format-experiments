# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent walks `/app/data`, treats every `.nwb` file under each `sub-*` directory as one session, sorts them by subject/session number, and loads each session directly with `h5py`. Within each session it reads raw fluorescence, neuropil, segmentation metadata, behavior arrays, timestamps, trial markers, and reward timestamps.

ii. 
```python
def list_sessions(sample=False, session_list=None):
    paths = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        subdir = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(subdir):
            continue
        for fn in sorted(os.listdir(subdir)):
            if fn.endswith(".nwb"):
                paths.append(os.path.join(subdir, fn))
...
    paths.sort(key=key)
    return paths
```

```python
def load_session(path):
    with h5py.File(path, "r") as f:
        subject = f["general/subject/subject_id"][()].decode()
        ...
        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        ...
        b = f["processing/behavior/BehavioralTimeSeries"]
        beh = {k: b[k]["data"][:] for k in
               ["position", "speed", "lick", "reward_zone", "environment",
                "trial number", "autoreward", "scanning"]}
        timestamps = b["position"]["timestamps"][:]
        trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]
        teleport = np.where(b["teleport"]["data"][:] > 0)[0]
        reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
```

iii. `CONVERSION_NOTES.md` Step 2 says the dataset is laid out as 11 subject directories with 152 NWB files and that each file already contains the aligned behavior and ophys products the agent needs, so it chose direct HDF5 access rather than reconstructing a `pynwb` object.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `sub-*` directories when enumerating files, and each session also carries `general/subject/subject_id`, which is used to populate `subjects` and `subject_idx`.

ii. 
```python
for sub in sorted(os.listdir(DATA_ROOT)):
    subdir = os.path.join(DATA_ROOT, sub)
    if not os.path.isdir(subdir):
        continue
```

```python
subject = f["general/subject/subject_id"][()].decode()
...
subjects = sorted({r["info"]["subject"] for r in results},
                  key=lambda s: int(s.replace("m", "")))
...
"subject_idx": np.array([subjects.index(r["info"]["subject"]) for r in results],
                        dtype=np.int64),
```

iii. Step 2 notes the DANDI layout is `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, so directory and NWB metadata agree on subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The script sorts file paths by `(subject number, session number)` and converts each file independently.

ii. 
```python
for fn in sorted(os.listdir(subdir)):
    if fn.endswith(".nwb"):
        paths.append(os.path.join(subdir, fn))
...
def key(p):
    base = os.path.basename(p)
    sub = int(base.split("_")[0].replace("sub-m", ""))
    ses = int(base.split("_")[1].replace("ses-", ""))
    return (sub, ses)
paths.sort(key=key)
```

iii. Step 2 identifies 152 NWB files and Step 9 reports 152 exported sessions, so the agent treated file granularity as session granularity.

## 1-d. How are the data split into trials?

i. Trials are laps delimited by NWB `trial_start` and `teleport`. The exported trial window is the half-open sample range `[trial_start, teleport)`.

ii. 
```python
trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]
teleport = np.where(b["teleport"]["data"][:] > 0)[0]
...
si, ti = S["trial_start"], S["teleport"]
...
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
```

iii. Step 5 states “A trial is one lap: NWB sample indices `[trial_start_idx, teleport_idx)`” and explicitly says this is the chosen temporal alignment event and exported window.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials flagged as lick-sensor failures: more than 30% of frames in the lap have cumulative lick count greater than 2. It does not drop short trials.

ii. 
```python
LICK_ERROR_FRAC = 0.30
...
lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                       for lo, hi in zip(si, ti)], dtype=bool)
...
for k, (lo, hi) in enumerate(zip(si, ti)):
    if lick_error[k]:
        continue
```

iii. Step 4 and Step 5 cite `behavior.correct_lick_sensor_error` from the reference code and note that this rule reproduces the paper’s reported 81 bad trials exactly; because `lick` is a decoder output, the agent chose to drop those trials instead of leaving NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported neural signal is derived from raw suite2p `Fluorescence` and `Neuropil`, restricted to ROIs with `iscell == 1`. It is not taken from the NWB `Deconvolved` dataset.

ii. 
```python
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = seg["iscell"][:, 0] > 0
...
for plane in sorted(f["processing/ophys/Fluorescence"].keys()):
    dset = f["processing/ophys/Fluorescence"][plane]["data"]
    ...
    F_parts.append(dset[:, :][:, keep].T.astype(np.float32))
    Fneu_parts.append(
        f["processing/ophys/Neuropil"][plane]["data"][:, :][:, keep].T.astype(np.float32))
```

iii. Step 1 says the paper computes its own neural signal from `F` and `Fneu`, while NWB `Deconvolved` is suite2p’s deconvolution of raw fluorescence and “not used in the paper analyses.”

## 2-b. How is the `neural` data processed?

i. The agent ports the paper’s dF/F pipeline: blank everything outside chosen baseline segments, subtract `0.7 * Fneu`, add back each segment’s mean neuropil, apply the maximin baseline (`gaussian_filter1d` sigma 15, then 300-frame min and max filters), compute `(F - baseline)/|baseline|`, smooth with sigma 2, and optionally deconvolve with OASIS. The default exported neural signal is dF/F, not events.

ii. 
```python
def compute_dff(F, Fneu, segments, deconvolve=True):
    ...
    f_ -= NEU_COEF * fneu_
    ...
    x = f_[:, lo:hi] + NEU_COEF * np.nanmean(fneu_[:, lo:hi], axis=1, keepdims=True)
    flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)
    flow = minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
    flow = maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
    d = (x - flow) / np.abs(flow)
    d = gaussian_filter1d(d, DFF_SMOOTH_SIGMA, axis=1)
    dff[:, lo:hi] = d
    if deconvolve:
        spks[:, lo:hi] = dcnv.oasis(np.ascontiguousarray(d), 2000, TAU, FRAME_RATE)
```

```python
activity = spks if signal == "events" else dff
activity = activity[keep_cells]
```

iii. Step 1 and Step 7 say this was copied from `preprocessing.dff`, but the agent changed the final exported signal to dF/F after directly comparing decoder accuracy for `events` vs `dff` and finding dF/F better for the provided classifier.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural data are filtered twice: first to `iscell == 1`, then to remove putative interneurons whose dF/F correlates with running speed above 0.5.

ii. 
```python
iscell = seg["iscell"][:, 0] > 0
...
keep = np.where(iscell[offset:offset + nroi])[0]
```

```python
d = dff[:, in_trial]
v = speed[in_trial].astype(np.float64)
...
r_speed = (dc.astype(np.float64) @ vc) / denom
...
keep_cells = r_speed <= INTERNEURON_SPEED_R
...
activity = activity[keep_cells]
```

iii. Step 1 cites the Methods for both curation rules and Step 9 reports the resulting interneuron fraction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to trial start simply by slicing the session-wide neural matrix on the same `[trial_start, teleport)` frame indices used to define laps.

ii. 
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
```

iii. Step 5 says the temporal alignment event is “start of trial” and that no extra offsetting or interpolation is needed because the behavior is already on the imaging frame grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native per-plane imaging rate: `FRAME_RATE = 15.5078125 Hz`, so each sample is `1 / FRAME_RATE = 0.0644836 s`. No temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 15.5078125
FRAME_PERIOD = 1.0 / FRAME_RATE
...
"time_bin_size": 1000.0 / FRAME_RATE,
```

iii. Step 3 and Step 5 say the paper’s GLM/decoder operate at the native imaging frame rate and that all sessions share the same per-plane rate once multipane acquisition is accounted for.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from trial length in imaging frames plus the global frame period, not directly from stored behavior timestamps. For a trial of length `T`, the agent constructs `0, 1, ..., T-1` and multiplies by `FRAME_PERIOD`.

ii. 
```python
FRAME_PERIOD = 1.0 / FRAME_RATE
...
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Step 5 says the agent chose the native frame period because all sessions had the same behavior dt and were already aligned to the imaging frame grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The processing is just frame counting from the beginning of each trial: `0, FRAME_PERIOD, 2*FRAME_PERIOD, ...`.

ii. 
```python
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Step 7’s review of sample plots says “time resets to 0 at each trial start and ramps at 1/15.5 s per frame.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector length equals the number of neural frames in the same `[lo:hi]` slice, and both come from the same per-trial frame indices after cropping session arrays to a common length.

ii. 
```python
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
timestamps = timestamps[:T]
...
T = hi - lo
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
...
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Step 2 notes the extra-frame mismatch in a few sessions and says the arrays are truncated to the common length before any trial slicing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the per-frame NWB behavior variable `environment`.

ii. 
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment",
        "trial number", "autoreward", "scanning"]}
...
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Step 2 identifies `environment` as the VR morph variable with values 0/1 on the aligned behavior grid.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent reduces the framewise `environment` series to one value per trial by taking the median within the trial and then broadcasting that scalar across all trial time bins.

ii. 
```python
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
...
inp[1] = environment[k]
```

iii. Step 5 says the task defines environment as a per-trial input, so the agent intentionally used a per-lap summary instead of a framewise trace.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the trial index implied by the ordered `trial_start`/`teleport` pairs, not from the raw NWB `trial number` series.

ii. 
```python
si, ti = S["trial_start"], S["teleport"]
...
trial_number = np.arange(ntrials_raw, dtype=np.int64)
```

iii. Step 5 says “trial index in session” is used and that it was verified against the NWB trial numbering at lap starts.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond sequential numbering within the session and broadcasting that number across all timepoints of the trial.

ii. 
```python
trial_number = np.arange(ntrials_raw, dtype=np.int64)
...
inp[2] = trial_number[k]
```

iii. Step 5 explicitly lists it as a 0-based lap index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward timestamps plus the current trial’s `reward_zone` trace: first the agent computes per-trial `is_reward` as “reward delivered during the trial and reward-zone entered on that trial,” then it shifts that vector by one trial.

ii. 
```python
reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
...
reward_idx = np.searchsorted(timestamps, reward_times)
...
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Step 5 maps previous outcome to the paper’s `get_trial_types` logic rather than to raw reward pulses alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trials after the first, the value is the previous entry of `is_reward`. For the first recorded trial of a session, the agent sets the value to `1` based on unrecorded warm-up laps from the preceding task context, then broadcasts the scalar across the trial.

ii. 
```python
prev_outcome = np.empty(ntrials_raw, dtype=np.int64)
prev_outcome[0] = 1
prev_outcome[1:] = is_reward[:-1]
...
inp[3] = prev_outcome[k]
```

iii. Step 5 says the first lap is set to rewarded because sessions were preceded by rewarded warm-up laps, and Step 166 in the trajectory flags this as a deliberate, documented deviation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the per-frame `position` trace plus the reward-zone identity inferred from the NWB scene string in `identifier`, with switch sessions changing zones after lap 30.

ii. 
```python
scene = parse_scene(f["identifier"][()].decode())
...
zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
zone_start = np.array([REWARD_ZONES[l][0] for l in zone_labels])
zone_end = np.array([REWARD_ZONES[l][1] for l in zone_labels])
...
p = pos[lo:hi]
out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Step 4 says the scene-name rule from `behavior.get_reward_zones` matched the empirical position of the first reward-zone flag on every trial within 8.5 cm, so the agent used the reference logic instead of inferring zones from noisy `reward_zone` values.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest point of the active reward zone: negative before the zone, zero inside the zone, positive after the zone, then bins it into 7 categories.

ii. 
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
...
out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Step 5 defines exactly this signed-distance convention and ties it to the decoder task bins.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The 7 categories are hard-coded by comparisons: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `> 50`.

ii. 
```python
def bin_distance_to_reward(dist):
    out = np.empty(dist.shape, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. Step 5 says these edges were chosen to match the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing `position` on the same trial frame interval used for neural activity and then computing the distance sample-by-sample on that slice.

ii. 
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
    p = pos[lo:hi]
    out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Step 5 says everything already lives on the imaging frame grid, so no extra interpolation is required.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the per-frame `position` behavior series.

ii. 
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment",
        "trial number", "autoreward", "scanning"]}
...
p = pos[lo:hi]
out[1] = bin_position(p)
```

iii. Step 2 describes `position` as the VR corridor position already resampled to imaging frames.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent slices each trial’s position trace and digitizes it into five 90 cm bins covering the 450 cm track.

ii. 
```python
def bin_position(pos):
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
...
out[1] = bin_position(p)
```

iii. Step 5 says the 5 equal bins are exactly those requested by the decoder task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It uses `np.digitize` edges at `90, 180, 270, 360` and clips the result to class IDs `0..4`.

ii. 
```python
def bin_position(pos):
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
```

iii. Step 5 states this is the 450 cm track divided into five equal 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `[lo:hi]` trial frame slice as the neural data, so alignment is one-to-one by frame index.

ii. 
```python
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
p = pos[lo:hi]
out[1] = bin_position(p)
```

iii. Step 3 and Step 5 say the behavior had already been aligned to the imaging frame grid upstream.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the per-frame `lick` behavior series.

ii. 
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment",
        "trial number", "autoreward", "scanning"]}
...
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. Step 2 notes that `lick` is the cumulative lick count per imaging frame in the aligned behavior data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The cumulative lick-count trace is binarized: any value greater than zero becomes class `1`, otherwise `0`.

ii. 
```python
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. Step 5 says the output must be binary and Step 7’s plot review says the exported binary matches the cumulative count being positive.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by using the same per-trial frame indices as the neural slice.

ii. 
```python
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
...
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. Step 5 says all streams are already on the same frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session’s scene string in `identifier` plus the within-session trial index, using the reference rule that switch scenes change zone after trial 30.

ii. 
```python
scene = parse_scene(f["identifier"][()].decode())
...
zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
zone_idx = np.array([ZONE_TO_IDX[l] for l in zone_labels], dtype=np.int64)
```

iii. Step 4 says this reproduces the paper’s `get_reward_zones` logic and was empirically checked against the reward-zone flag positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses scene names like `Env1_LocationA_to_B` or `Env1_B_to_Env2_C`, assigns pre-switch and post-switch labels around lap 30, and maps `A/B/C` to `0/1/2`. The resulting per-trial scalar is broadcast across all frames of the trial.

ii. 
```python
def reward_zone_labels(scene, ntrials, change_trial=SWITCH_TRIAL):
    ...
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]
        labels[change_trial:] = parts[3]
    elif len(parts) == 5 and parts[2] == "to":
        labels[:change_trial] = parts[1]
        labels[change_trial:] = parts[4]
...
out[4] = zone_idx[k]
```

iii. Step 5 lists this mapping explicitly and notes that all 77 switch sessions changed at lap 30, matching the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` timestamps together with the per-frame `reward_zone` trace; a trial is rewarded only if a reward occurred during the trial and the reward-zone flag was entered.

ii. 
```python
reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
...
reward_idx = np.searchsorted(timestamps, reward_times)
...
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Step 5 ties this directly to `behavior.get_trial_types` from the reference code and notes that omission trials never set the reward-zone flag.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped onto behavior frame indices with `np.searchsorted`, then reduced to one binary value per trial via `is_reward`; that scalar is broadcast across every frame in the trial.

ii. 
```python
reward_idx = np.searchsorted(timestamps, reward_times)
...
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
...
out[5] = is_reward[k]
```

iii. Step 4 says this criterion reproduced the expected reward rate and matched the reference code’s notion of rewarded trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The main explicit handling is cropping session arrays to the common fluorescence/behavior length when one extra imaging frame is present. The code also asserts structural consistency of trial markers, removes lick-sensor-failure trials, and errors out if a kept trial contains non-finite neural activity. It does not contain special handling for missing reward-zone labels because it derives reward zones from scene metadata instead.

ii. 
```python
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
timestamps = timestamps[:T]
```

```python
assert len(trial_start) == len(teleport), "trial_start/teleport count mismatch"
assert np.all(teleport > trial_start), "teleport must follow its trial start"
assert np.all(trial_start[1:] > teleport[:-1]), "trials must not overlap"
assert teleport[-1] <= T, "trial extends past the imaging data"
...
if lick_error[k]:
    continue
...
if not np.all(np.isfinite(act)):
    raise RuntimeError("non-finite activity in %s trial %d" % (path, k))
```

iii. Step 2 says ten sessions had a one-frame mismatch and were truncated; Steps 4 and 5 say the scene-name reward-zone rule avoided having to impute missing reward-zone labels from noisy framewise data.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are per-session NWB I/O and dF/F computation. The notes say dF/F (and OASIS when requested) dominates compute, while NWB reads dominate I/O; trial assembly is relatively cheap.

ii. 
```python
t_load = time.time()
S = load_session(path)
t_load = time.time() - t_load
...
t_dff = time.time()
dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                        deconvolve=(signal == "events") or show_processing)
t_dff = time.time() - t_dff
```

iii. Step 6 and Step 7 include timing tables showing load time and dF/F time per session and identify these as the dominant costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The notes say most neuronwise computation is already vectorized. Remaining Python loops are mainly over trials/laps: baseline-segment iteration in `compute_dff`, construction of the `in_trial` mask, per-trial reward/environment/lick-error summaries, and final trial assembly.

ii. 
```python
for lo, hi in segments:
    ...
    flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)
```

```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    if lick_error[k]:
        continue
    ...
    neural.append(act)
    inputs.append(inp)
    outputs.append(out)
```

iii. Step 6 says this per-lap structure is mostly unavoidable because baselines are defined per lap and output arrays are stored as variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats a few computations: it optionally computes OASIS events even when exporting dF/F if diagnostic plots are requested, repeatedly loops over trials to derive multiple per-trial summaries, and diagnostic plotting recomputes some derived traces for visualization.

ii. 
```python
dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                        deconvolve=(signal == "events") or show_processing)
```

```python
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                       for lo, hi in zip(si, ti)], dtype=bool)
```

iii. Step 6 says the agent deliberately removed a larger survey/conversion duplication from earlier prototypes, so the remaining repetition is local rather than whole-dataset rereading.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Under the default `--signal dff`, deconvolved events are still computed if `--show-processing` is enabled, but they are only used for plots. The loader also reads `autoreward`, `scanning`, and some metadata that are kept only for checks or notes, not for exported decoder arrays.

ii. 
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment",
        "trial number", "autoreward", "scanning"]}
```

```python
dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                        deconvolve=(signal == "events") or show_processing)
...
if show_processing:
    plot_processing(...)
```

iii. Step 6 explicitly notes that deconvolution is skipped unless needed, implying that when it is triggered for plotting during dF/F export it is extra work not used in the final pickle.
