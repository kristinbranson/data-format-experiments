# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by listing `sub-*` directories under `/app/data/` and collecting every `.nwb` file within each. Each NWB file is read using `h5py` (not `pynwb`). The `load_session()` function reads fluorescence, neuropil, behavior time series, trial indices, reward timestamps, and session metadata from the HDF5 structure. All 152 NWB files across 11 subjects are processed.

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

def load_session(path):
    with h5py.File(path, "r") as f:
        subject = f["general/subject/subject_id"][()].decode()
        exp_day = int(f["general/session_id"][()].decode())
        ...
        F_parts, Fneu_parts, plane_parts, roi_parts = [], [], [], []
        for plane in sorted(f["processing/ophys/Fluorescence"].keys()):
            dset = f["processing/ophys/Fluorescence"][plane]["data"]
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

iii. The AI documented in CONVERSION_NOTES.md that using h5py gives direct access to the HDF5 structure. All 152 files across 11 mice are found. The AI verified 152 sessions total (14 per mouse except m11 with 12).

## 1-b. How are the data split into subjects?

i. Subjects are identified from `sub-*` directories in the data directory, and also from the `general/subject/subject_id` field in each NWB file. Subjects are sorted by numeric ID.

ii.
```python
subjects = sorted({r["info"]["subject"] for r in results},
                  key=lambda s: int(s.replace("m", "")))
```

iii. The subject IDs (m3, m4, m7, m11-m15, m17-m19) are consistent across directory names and NWB metadata, totaling 11 switch mice matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by (subject number, session number) for consistent ordering.

ii.
```python
def list_sessions(sample=False, session_list=None):
    ...
    def key(p):
        base = os.path.basename(p)
        sub = int(base.split("_")[0].replace("sub-m", ""))
        ses = int(base.split("_")[1].replace("ses-", ""))
        return (sub, ses)
    paths.sort(key=key)
    return paths
```

iii. The AI verified that each NWB file name encodes subject and session (experiment day), and that session counts per subject match the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by the `trial_start` and `teleport` behavior variables. Trial starts are indices where `trial_start > 0`; trial ends are indices where `teleport > 0`. Each trial spans `[trial_start, teleport)`.

ii.
```python
trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]
teleport = np.where(b["teleport"]["data"][:] > 0)[0]
...
assert len(trial_start) == len(teleport), "trial_start/teleport count mismatch"
assert np.all(teleport > trial_start), "teleport must follow its trial start"
assert np.all(trial_start[1:] > teleport[:-1]), "trials must not overlap"
```

iii. The AI verified that every trial start is followed by exactly one teleport, that trials don't overlap, and that the last trial ends before the data ends. This matches the paper's definition of a trial as one lap on the track.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor failure are removed. A trial is flagged as a lick error if more than 30% of its frames have a cumulative lick count > 2, matching the paper's `behavior.correct_lick_sensor_error` criterion. This removed exactly 81 trials across the dataset, matching the paper's reported count.

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

iii. From CONVERSION_NOTES Step 4: "Lick-sensor error trials: `mean(lick>2) > thr` per trial, 81 / 12,216 trials (0.66%) with thr = 0.30 — Exact match" with the paper's count of 81 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p fluorescence (`Fluorescence`) and neuropil (`Neuropil`) traces stored in the NWB file. The `Deconvolved` field is explicitly NOT used (it is suite2p's own deconvolution of raw F, not the paper's signal).

ii.
```python
F_parts, Fneu_parts = [], []
for plane in sorted(f["processing/ophys/Fluorescence"].keys()):
    dset = f["processing/ophys/Fluorescence"][plane]["data"]
    ...
    F_parts.append(dset[:, :][:, keep].T.astype(np.float32))
    Fneu_parts.append(
        f["processing/ophys/Neuropil"][plane]["data"][:, :][:, keep].T.astype(np.float32))
```

iii. CONVERSION_NOTES Step 1: "Imaging data need dF/F computed from scratch: the NWB stores raw suite2p Fluorescence, Neuropil and suite2p's own Deconvolved (which suite2p ran on raw F, not on the authors' custom dF/F)."

## 2-b. How is the `neural` data processed?

i. dF/F is computed from F and Fneu following the paper's pipeline: neuropil subtraction (coefficient 0.7), add back per-trial mean neuropil, per-trial maximin baseline (Gaussian smooth sigma=15 frames, then 300-frame min filter, then 300-frame max filter), `dF/F = (F - baseline)/|baseline|`, Gaussian smooth sigma=2 frames. OASIS deconvolution is also computed but **dF/F is the signal exported**, not deconvolved events. Cells from multiple planes are pooled. The `keep_teleports` flag (from `teleport_metadata`) controls whether the baseline segment spans the inter-trial interval.

ii.
```python
def compute_dff(F, Fneu, segments, deconvolve=True):
    f_ -= NEU_COEF * fneu_
    for lo, hi in segments:
        x = f_[:, lo:hi] + NEU_COEF * np.nanmean(fneu_[:, lo:hi], axis=1, keepdims=True)
        flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)
        flow = minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        d = (x - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH_SIGMA, axis=1)
        dff[:, lo:hi] = d
        if deconvolve:
            spks[:, lo:hi] = dcnv.oasis(np.ascontiguousarray(d), 2000, TAU, FRAME_RATE)
    return dff, spks
```

iii. CONVERSION_NOTES Step 5, Key Decision #1: "Neural signal = dF/F... The paper's own decoding (Fig. 3) is fit on deconvolved events, but the decoder used for grading is a memoryless per-timepoint linear classifier. dF/F integrates activity over the indicator decay and performs better." The AI measured accuracy on both signals and dF/F won on every time-varying output (CONVERSION_NOTES Step 7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only ROIs with `iscell == 1` (suite2p manual curation) are kept. (2) Putative interneurons are removed: cells with Pearson r(dF/F, running speed) > 0.5, computed over all in-trial samples.

ii.
```python
iscell = seg["iscell"][:, 0] > 0
...
keep = np.where(iscell[offset:offset + nroi])[0]
F_parts.append(dset[:, :][:, keep].T.astype(np.float32))
...
# interneuron removal
d = dff[:, in_trial]
v = speed[in_trial].astype(np.float64)
dc = d - d.mean(axis=1, keepdims=True)
vc = v - v.mean()
denom = np.sqrt((dc.astype(np.float64) ** 2).sum(axis=1)) * np.sqrt((vc ** 2).sum())
r_speed = (dc.astype(np.float64) @ vc) / denom
keep_cells = r_speed <= INTERNEURON_SPEED_R
```

iii. Both filters come from the paper's Methods: "suite2p manual curation" and "a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed". The interneuron fraction (~0.35% +/- 0.60%) is consistent with the paper's 0.42 +/- 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data are aligned to trial start, which requires no additional processing beyond slicing the trial boundaries `[trial_start, teleport)`. Each trial's neural data starts at the trial_start index.

ii.
```python
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
```
where `lo = si[k]` (trial_start) and `hi = ti[k]` (teleport).

iii. Neural and behavioral data share the same frame grid (VR resampled to imaging frames), so slicing the same indices ensures alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate of 15.5078125 Hz (64.4836 ms per frame) is preserved. No temporal rebinning is applied. This rate is the same for all sessions (including 2-plane animals where the scanner rate is 31 Hz but the per-plane rate is 15.5 Hz).

ii.
```python
FRAME_RATE = 15.5078125          # Hz, per plane; identical for every session
FRAME_PERIOD = 1.0 / FRAME_RATE  # s
...
"time_bin_size": 1000.0 / FRAME_RATE,  # in metadata
```

iii. CONVERSION_NOTES Step 5: "Time bin = the native imaging frame, 64.4836 ms, identical for all sessions and animals. This is the rate the paper's GLM/decoder use."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial, multiplied by the frame period (1/15.5078125 s).

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Since all sessions have the same per-plane frame rate, the frame index gives a precise time from trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Each frame index within the trial (0, 1, 2, ...) is multiplied by the frame period (1/FRAME_RATE). No timestamps are read; the constant frame rate is used directly.

ii.
```python
FRAME_PERIOD = 1.0 / FRAME_RATE  # = 1/15.5078125 s
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. The AI verified that the behavior timestamps have a constant dt equal to the frame period across all sessions, so using the frame index is equivalent.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time starts at 0 for each trial and increments by one frame period per sample. Since neural data uses the same frame indices, alignment is inherent.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Both neural and behavioral data live on the same imaging frame grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` (morph) behavior time series.

ii.
```python
beh = {k: b[k]["data"][:] for k in ["position", "speed", "lick", "reward_zone", "environment", ...]}
...
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. The environment variable is 0 (ENV1) or 1 (ENV2), constant within a trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial median of the environment values is taken and rounded to integer. Since environment is constant within a trial, this is a robust way to extract the per-trial value.

ii.
```python
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
...
inp[1] = environment[k]
```

iii. The value is broadcast across all timepoints in the trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the 0-based lap index within the session (the loop variable over trials).

ii.
```python
trial_number = np.arange(ntrials_raw, dtype=np.int64)
...
inp[2] = trial_number[k]
```

iii. This is computed before trial filtering so the numbering reflects what the animal experienced.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; simply the 0-based sequential index. The value is broadcast across all timepoints in the trial.

ii.
```python
trial_number = np.arange(ntrials_raw, dtype=np.int64)
inp[2] = trial_number[k]
```

iii. CONVERSION_NOTES Step 5, Key Decision #10: "trial_number and previous_trial_outcome are computed before any trial is dropped, so lap numbering and outcome history stay faithful to what the animal experienced."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior time series. A trial is considered rewarded if a reward event occurred within the trial AND the reward zone was entered.

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

iii. This matches `behavior.get_trial_types`: `isreward = any(reward>0) AND any(rzone>0)` per trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial k, the previous trial outcome is `is_reward[k-1]`. For the **first trial** of a session, the value is set to **1 (rewarded)**, based on the reasoning that sessions were preceded by rewarded warm-up laps. The value is broadcast across all timepoints.

ii.
```python
prev_outcome = np.empty(ntrials_raw, dtype=np.int64)
prev_outcome[0] = 1
prev_outcome[1:] = is_reward[:-1]
...
inp[3] = prev_outcome[k]
```

iii. CONVERSION_NOTES Step 5, Key Decision #8: "First lap's previous_trial_outcome = 1 (rewarded). Each imaging session was preceded immediately by 30 warm-up laps... on which reward was delivered unless randomly omitted (~15%), so 'rewarded' is the expected value."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. The reward zone for each trial is determined from the **scene name** in the NWB `identifier` field, using the same logic as the reference code's `behavior.get_reward_zones`: parsing the VR scene name to determine which zone (A/B/C) is active, with a switch after trial 30 on switch days.

ii.
```python
def reward_zone_labels(scene, ntrials, change_trial=SWITCH_TRIAL):
    parts = scene.split("_")
    labels = np.empty(ntrials, dtype="<U1")
    if len(parts) == 2 and parts[1].startswith("Location"):
        labels[:] = parts[1][-1]
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]
        labels[change_trial:] = parts[3]
    elif len(parts) == 5 and parts[2] == "to":
        labels[:change_trial] = parts[1]
        labels[change_trial:] = parts[4]
    ...
...
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
```

iii. CONVERSION_NOTES Step 4: "Reward zone per trial: scene name + switch at trial 30... verified empirically: position at the first reward_zone flag is within 8.5 cm of the predicted zone start for every trial."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone: negative before the zone, 0 inside, positive after.

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
```

iii. This matches the paper's concept of reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins using explicit boolean conditions:
- 0: d < -50
- 1: -50 <= d < -10
- 2: -10 <= d < 0
- 3: d == 0 (inside zone)
- 4: 0 < d <= 10
- 5: 10 < d <= 50
- 6: d > 50

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

iii. The bin edges match the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices within each trial. Slicing `pos[lo:hi]` gives the same timepoints as `activity[:, lo:hi]`.

ii.
```python
p = pos[lo:hi]
out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Both neural and behavioral data are on the imaging frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
beh = {k: b[k]["data"][:] for k in ["position", ...]}
...
p = pos[lo:hi]
```

iii. Position records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond slicing per-trial and discretizing into 5 equal 90-cm bins spanning the 450 cm track.

ii.
```python
def bin_position(pos):
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
```

iii. The 5 bins of 90 cm each span the 450 cm track as specified.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using `np.digitize` with edges [90, 180, 270, 360], clipped to [0, 4]:
- 0: < 90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: >= 360 cm

ii.
```python
def bin_position(pos):
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
```

iii. Matches the instruction specification of 5 equal bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices. No additional alignment needed.

ii. `p = pos[lo:hi]` uses the same trial boundaries as neural data.

iii. Verified by shared frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
beh = {k: b[k]["data"][:] for k in [..., "lick", ...]}
...
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. The `lick` variable is a cumulative lick count per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any lick count > 0 is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. The instructions specify binary lick output (no/yes). The raw lick values are cumulative counts per frame.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices within each trial.

ii. `lick[lo:hi]` uses same `lo, hi` as neural data.

iii. Shared imaging frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` string, which encodes the VR scene name. The scene name determines which zone (A/B/C) is active for each trial, including the switch at trial 30 on switch days. This directly ports the reference code's `behavior.get_reward_zones`.

ii.
```python
def parse_scene(identifier):
    return identifier.strip("/").split("/")[-1]

def reward_zone_labels(scene, ntrials, change_trial=SWITCH_TRIAL):
    parts = scene.split("_")
    if len(parts) == 2 and parts[1].startswith("Location"):
        labels[:] = parts[1][-1]
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]
        labels[change_trial:] = parts[3]
    ...
...
zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
zone_idx = np.array([ZONE_TO_IDX[l] for l in zone_labels], dtype=np.int64)
out[4] = zone_idx[k]
```

iii. Verified in CONVERSION_NOTES Step 4: the empirical reward-zone entry position matches the scene-name assignment on every trial, with max error 8.5 cm.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed to extract zone labels (A/B/C), which are mapped to indices 0/1/2. On switch sessions, the zone changes after trial 30. The value is broadcast across all timepoints in the trial.

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
zone_idx = np.array([ZONE_TO_IDX[l] for l in zone_labels], dtype=np.int64)
out[4] = zone_idx[k]
```

iii. Matches the reference code's `get_reward_zones(change_trial=30)`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps and the `reward_zone` behavior time series. A trial is rewarded if a reward event occurred AND the reward zone was entered.

ii.
```python
reward_times = f["...Reward"]["timestamps"][:]
reward_idx = np.searchsorted(timestamps, reward_times)
...
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
...
out[5] = is_reward[k]
```

iii. This matches the reference code's `behavior.get_trial_types` definition: `isreward = any(reward>0) AND any(rzone>0)`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial: 1 if rewarded, 0 otherwise. The value is broadcast across all timepoints. The overall reward rate is 84.64%, matching the paper's ~85%.

ii.
```python
out[5] = is_reward[k]
```

iii. The paper states ~15% of trials are omission trials. The converted data has 15.8% omitted, consistent.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: In 10 of 152 sessions, fluorescence arrays are 1 frame longer than behavior arrays. Data is cropped to the common length.
- **Lick sensor failures**: 81 trials with stuck lick sensors (>30% of frames with cumulative lick count > 2) are dropped entirely, since lick is a decoder output and NaN values aren't allowed.
- **Teleport spanning**: The `keep_teleports` flag handles sessions where the laser was not blanked during inter-trial intervals, correctly widening the dF/F baseline window.
- **Trial assertions**: Multiple assertions verify trial_start/teleport count match, teleport follows its trial start, and trials don't overlap.

ii.
```python
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
...
lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                       for lo, hi in zip(si, ti)], dtype=bool)
for k, (lo, hi) in enumerate(zip(si, ti)):
    if lick_error[k]:
        continue
```

iii. CONVERSION_NOTES documents all these cases. The 81 lick-error trials match the paper exactly.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **NWB reading** (loading F, Fneu, and behavior arrays from HDF5): I/O bound, 0.1-1.8 s per session
2. **dF/F computation** (neuropil subtraction, baseline, smoothing): 0.2-7 s per session depending on neuron count
3. **Pickle writing** (9.6 GB output file)

ii. Timing is printed per session in the conversion output, e.g.:
```
[  1/152] sub-m3_ses-01  1013 neurons  80 trials  (load 1.6s dff 5.2s total 7.0s)
```

iii. CONVERSION_NOTES Step 7: "The largest session is 10.4 s; sessions scale with neuron count." Total wall clock: 43.8 s with 12 workers.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over trials to assemble per-trial arrays. The dF/F baseline computation loops over segments (trials). Neither is easily vectorizable due to variable trial lengths. The interneuron correlation is already vectorized (matrix multiplication instead of per-cell loop).

ii.
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
    inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
    ...
```

iii. Variable trial lengths make full vectorization impractical. The per-trial loop is ~80 iterations with fast array operations inside.

## 13-c. What processing does the code repeat multiple times?

i. The code does NOT have redundant processing. Each session is loaded once and processed in a single pass. The `load_session` and `convert_session` functions are combined into one pipeline per session.

ii. N/A - single-pass design.

iii. Unlike a survey-then-convert approach, the AI's code processes everything in one pass per session, avoiding redundant I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. When `--signal dff` (the default), the OASIS deconvolution is skipped for sessions where `show_processing` is False. The dF/F for interneuron cells (~0.35% of cells) is computed but then discarded after the correlation check; however, the dF/F is needed for the check itself, so this is not truly unnecessary. No other significant unnecessary processing was identified.

ii.
```python
dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                        deconvolve=(signal == "events") or show_processing)
```

iii. The deconvolution flag ensures OASIS is only run when needed.
