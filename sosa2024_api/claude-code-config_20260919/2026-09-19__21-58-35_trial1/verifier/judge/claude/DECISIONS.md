# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered with `glob.glob` over the `sub-*` directories. Each file is loaded using `pynwb.NWBHDF5IO`. The function `load_session` reads neural data (Fluorescence, Neuropil, iscell from ImageSegmentation), all behavioral time series, and metadata (subject ID, session ID, scene name from `nwb.identifier`) in a single pass per file. Sessions are processed in parallel via `ProcessPoolExecutor`.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))
...
def load_session(path):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        session_id = nwb.session_id
        ident_parts = nwb.identifier.rstrip("/").split("/")
        scene = ident_parts[-1]
        ...
```

iii. The glob pattern `sub-*/*.nwb` finds all 152 NWB files across 11 subject directories, matching the DANDI manifest. The AI verified 11 subjects and 152 sessions match the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` within each NWB file. Unique subjects are accumulated in order of first appearance during processing.

ii.
```python
subject = nwb.subject.subject_id
...
if info["subject"] not in subjects:
    subjects.append(info["subject"])
subject_idx.append(subjects.index(info["subject"]))
```

iii. The subject ID is read directly from the NWB metadata. The AI verified 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from `nwb.session_id`.

ii.
```python
session_id = nwb.session_id
```

iii. File naming convention (`sub-<id>_ses-<DD>_behavior+ophys.nwb`) and NWB metadata both identify sessions.

## 1-d. How are the data split into trials?

i. Trial starts are identified by frames where `trial_start > 0`. Trial ends are identified by frames where `teleport > 0`. Each trial spans `[start, end)`.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
ends = np.where(beh["teleport"] > 0)[0]
assert len(starts) == len(ends), "trial_start / teleport count mismatch"
assert np.all(ends > starts), "teleport must follow its trial_start"
```

iii. The AI uses `trial_start` and `teleport` behavioral time series to define trial boundaries, consistent with the reference code's `get_trial_types` which uses these same variables.

## 1-e. How are trials filtered based on quality controls?

i. Three filtering criteria: (1) lick-sensor failure trials (>30% of imaging frames with cumulative lick count >2, exactly matching the paper's description, finding 81 trials), (2) trials overlapping frames without 2P scanning (`scanning < 0`), (3) trials shorter than 2 frames. This removes 81 trials total (all lick-sensor errors; no unscanned or ultra-short trials were present in the data).

ii.
```python
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                       > LICK_ERR_FRAC_THRESH)
trial_unscanned[i] = np.any(scanning[s:e] < 0)
...
too_short = trial_len < 2
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
```

iii. From CONVERSION_NOTES: "Lick-error trial detection reproduces the paper's 81 trials" and "n = 81 out of 12,376 trials removed across 11 switch mice". The AI verified the exact count matches the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces `Fluorescence` (F) and `Neuropil` (Fneu), accessed via the NWB `ophys` processing module. The NWB `Deconvolved` field is explicitly not used (it contains suite2p's own deconvolution of raw fluorescence, not the paper's signal).

ii.
```python
fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
neuro = ophys.data_interfaces["Neuropil"].roi_response_series
...
F[rows] = np.asarray(fluo[k].data[:]).T
Fneu[rows] = np.asarray(neuro[k].data[:]).T
```

iii. From CONVERSION_NOTES: "the NWB stores raw suite2p F, Fneu and suite2p's own Deconvolved/spks, none of which are dF/F" and "The NWB 'Deconvolved' array is suite2p's spks from raw F ... not the paper's deconvolution of dF/F."

## 2-b. How is the `neural` data processed?

i. dF/F is computed via `compute_dff()`, a reimplementation of the paper's `preprocessing.py::dff()` for the `keep_teleports=False` configuration:
1. Restrict to within-trial samples (NaN everything else)
2. Neuropil subtraction: `F -= 0.7 * Fneu`
3. Add back per-trial mean neuropil
4. Gaussian smooth (sigma=15 samples)
5. Min filter (300 samples)
6. Max filter (300 samples)
7. `dF/F = (F - baseline) / |baseline|`
8. Smooth dF/F with Gaussian (sigma=2 samples) per trial

The AI uses **dF/F directly** as the neural signal (no deconvolution). The `keep_teleports` parameter is always False (the function does not implement the `keep_teleports=True` path).

ii.
```python
def compute_dff(F, Fneu, trial_starts, trial_ends):
    f_ = np.full(F.shape, np.nan)
    fneu_ = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    in_trial = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_
    ...
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)
        tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
        flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
    dff[:, in_trial] = ((f_[:, in_trial] - flow[:, in_trial]) / np.abs(flow[:, in_trial]))
    for s, e in zip(trial_starts, trial_ends):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
    return dff, in_trial
```

iii. From CONVERSION_NOTES Step 5, Decision 2: "Neural signal = dF/F ... Rationale: (a) the NWB's Deconvolved array is suite2p's spks from raw F and does not correspond to the paper's deconvolution ... (b) the paper itself uses binned dF/F for its spatial peak identification ... (c) I tested both signals with a PCA+logistic position decoder on three sessions -- dF/F was equal or better in every case." The AI also notes: "The deconvolution step itself is implemented and validated but not used."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) suite2p `iscell` filtering (manual curation, column 0 of `iscell` array == 1), (2) putative interneuron exclusion by Pearson correlation between dF/F and running speed > 0.5. Both match the paper's Methods. The iscell filter is applied before dF/F computation to save memory.

ii.
```python
iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
...
F = F[iscell]
Fneu = Fneu[iscell]
...
d0 = d - d.mean(axis=1, keepdims=True)
s0 = sp_ - sp_.mean()
denom = np.sqrt((d0 ** 2).sum(axis=1) * (s0 ** 2).sum())
speed_corr = (d0 @ s0) / denom
keep_cells = speed_corr <= SPEED_CORR_THRESH
```

iii. From CONVERSION_NOTES: "suite2p manual curation -> iscell[:,0] == 1" and "Pearson corr(dF/F, speed) > 0.5 -> putative interneuron, excluded." The AI verified ~0.35% of cells excluded as interneurons, consistent with the paper's 0.42 +/- 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to the trial start. Since neural and behavioral data share the same time indices (the NWB stores VR-aligned behavior on the 2P frame clock), alignment requires no additional processing beyond slicing into trials at `[start, end)`.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
```

iii. From CONVERSION_NOTES: "VR behaviour is interpolated onto the 2P frame times (vr_align_to_2P); the NWB already stores it that way (one shared timestamp vector)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used: 64.484 ms (15.5078 Hz per plane), identical across all 152 sessions. No rebinning is applied.

ii.
```python
dt = float(np.median(np.diff(ts)))
...
"time_bin_size": dt * 1000.0,
```

iii. From CONVERSION_NOTES Step 5, Decision 1: "Time bin = native imaging frame (64.484 ms). Every one of the 152 sessions has exactly the same dt ... Using the native grid therefore satisfies 'same bin size for all trials and sessions' with zero resampling."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the shared `timestamps` array of the behavioral time series (specifically `position.timestamps`).

ii.
```python
timestamps = np.asarray(bts["position"].timestamps[:])
...
inp[0] = ts[s:e] - ts[s]
```

iii. All behavioral time series share the same timestamps (the 2P frame clock).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first frame of the trial is subtracted from all timestamps in the trial.

ii.
```python
inp[0] = ts[s:e] - ts[s]
```

iii. Simple subtraction to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (both are on the 2P frame clock), so no additional alignment is needed. The same `[s, e)` slice is used for both.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
inp[0] = ts[s:e] - ts[s]
```

iii. From CONVERSION_NOTES: "the NWB already stores it that way (one shared timestamp vector)."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
env_ts = beh["environment"]
...
ev = np.unique(env_ts[s:e])
ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
```

iii. The `environment` channel encodes -1 (ITI), 0 (ENV1), 1 (ENV2). The per-trial value is extracted by taking the unique non-negative value within the trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique non-negative environment value within the trial is extracted. Values of -1 (ITI periods) are excluded. The value is constant within each trial.

ii.
```python
ev = np.unique(env_ts[s:e])
ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
...
inp[1] = trial_env[i]
```

iii. From CONVERSION_NOTES: "Each trial has a single environment value (verified for all 12,216 trials)."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-indexed position of the trial within the session, derived from the loop counter over raw trials.

ii.
```python
for i in np.where(keep_trial)[0]:
    ...
    inp[2] = i
```

iii. The trial number is the raw trial index within the session, preserving the original trial ordering even when some trials are filtered out.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. Simple assignment of the raw trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series (timestamps of reward deliveries) and the `reward_zone` time series. Reward delivery is defined as: any reward timestamp falls within the trial AND the reward zone channel is active during the trial.

ii.
```python
reward_times = np.asarray(bts["Reward"].timestamps[:])
...
rew_frames = np.searchsorted(ts, S["reward_times"])
...
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
```

iii. The definition of `isreward = any(reward>0) and any(rzone>0)` matches the reference code's `get_trial_types`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. For trial 0, the value is set to 1 (rewarded), justified by the fact that each imaging session is preceded by ~30 rewarded warm-up trials. The value is constant across all timepoints within a trial.

ii.
```python
prev_rewarded = np.empty(n_trials_raw, dtype=np.int64)
prev_rewarded[0] = 1
prev_rewarded[1:] = trial_rewarded[:-1]
...
inp[3] = prev_rewarded[i]
```

iii. From CONVERSION_NOTES Step 5, Decision 7: "`previous_trial_reward` for trial 0 = 1 (rewarded). Each imaging session is immediately preceded by ~30 warm-up trials of the same task with the same reward zone ... and rewards are omitted on only ~15% of trials, so 'rewarded' is the correct prior."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone location for the current trial. The reward zone is determined from the **scene name** (parsed from `nwb.identifier`) using the paper's `get_reward_zones` logic: for switch sessions, the zone changes at trial 30.

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

iii. From CONVERSION_NOTES Step 5, Decision 6: "Reward zone from the scene name + change_trial = 30 (the reference method), cross-checked against the position at which the reward_zone channel fires on every trial where it fires." Verified 0 mismatches in 10,394 observed zone entries.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the animal's position to the nearest edge of the reward zone. Distance is 0 inside the zone, negative before, positive after. The continuous distance is then discretized into 7 bins.

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    out = np.full(pos.shape, 3, dtype=np.int64)     # 3 == inside the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out, d
```

iii. The signed distance and discretization follow the instructions exactly. The reward zone coordinates come from the paper: A=[80,130], B=[200,250], C=[320,370].

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit comparisons: 0 (<-50), 1 (-50 to -10), 2 (-10 to <0), 3 (0, inside zone), 4 (>0 to +10), 5 (+10 to +50), 6 (>+50).

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

iii. Bins match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial. Both use `[s, e)` slice.

ii.
```python
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. Neural and behavioral data share the same time indexing (2P frame clock).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = beh["position"]
...
p = pos[s:e]
```

iii. The `position` variable records the animal's position on the 450 cm track in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing into 5 bins.

ii.
```python
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins of 90 cm each using `np.digitize` with bin edges `[90, 180, 270, 360]`, clipped to range [0, 4].

ii.
```python
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. The instructions specify "5 equal-sized bins spanning the 450 cm track", which is 90 cm per bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data. Both use `[s, e)` slice.

ii.
```python
p = pos[s:e]
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. Same 2P frame clock alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = beh["lick"]
...
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick count is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes). The raw values can be >1 (cumulative count), so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data. Both use `[s, e)` slice.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. Same 2P frame clock alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the **scene name** (parsed from `nwb.identifier`), using the paper's `get_reward_zones` logic. For switch sessions, the zone changes at trial 30 (`change_trial=30`). Zone names are mapped to indices: A=0, B=1, C=2.

ii.
```python
def reward_zone_labels_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    for z in ZONE_NAMES:
        if scene.endswith("_Location" + z):
            return [z] * n_trials
    zone1 = scene[-1]
    zone0 = None
    for z in ZONE_NAMES:
        if (z + "_to") in scene:
            zone0 = z
            break
    return [zone0] * min(change_trial, n_trials) + \
           [zone1] * max(n_trials - change_trial, 0)
...
out[4] = scene_zone[i]
```

iii. From CONVERSION_NOTES: "Reward zone from the scene name + change_trial = 30 (the reference method), cross-checked against the position at which the reward_zone channel fires on every trial where it fires." Verified 0 mismatches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name string is parsed to extract the reward zone letter(s). For switch sessions (containing "_to"), the pre-switch zone is used for the first 30 trials and the post-switch zone for the rest. The zone letter is mapped to an integer index: A=0, B=1, C=2.

ii.
```python
scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
...
out[4] = scene_zone[i]
```

iii. The logic directly follows the reference code's `get_reward_zones` function and the paper's "each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series (timestamps of reward deliveries) and the `reward_zone` time series.

ii.
```python
reward_times = np.asarray(bts["Reward"].timestamps[:])
rew_frames = np.searchsorted(ts, S["reward_times"])
...
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
out[5] = trial_rewarded[i]
```

iii. Reward is defined as: any reward timestamp falls within the trial AND the reward zone channel is active, matching the reference code's `get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to the nearest behavior frame using `searchsorted`. A trial is considered rewarded if any reward event occurred AND the reward zone was active during the trial. The value is constant across all timepoints in the trial.

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

iii. The definition `isreward = any(reward>0) and any(rzone>0)` matches the reference code's `get_trial_types`. The AI verified ~15.36% omission rate, matching the paper's ~15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Multi-plane sessions** (m17, m18): Planes are pooled by scattering traces into rows of the `PlaneSegmentation` table using `planeIdx`.
- **1-frame length mismatch** (10 two-plane sessions): If fluorescence has 1 more frame than behavior timestamps, the trailing frame is dropped with an assertion that excess is <= 1.
- **Lick-sensor failures**: 81 trials removed (matching the paper).
- **Missing reward zone data**: On omission trials where the `reward_zone` channel doesn't fire, the zone is determined from the scene name (not dependent on the channel).
- **Sessions with < 2 kept trials**: Dropped with a warning.

ii.
```python
n_extra = F.shape[1] - len(timestamps)
assert 0 <= n_extra <= 1, ...
if n_extra:
    F = F[:, :len(timestamps)]
    Fneu = Fneu[:, :len(timestamps)]
...
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                       > LICK_ERR_FRAC_THRESH)
...
if len(n) < 2:
    print(f"  WARNING: dropping {info['file']} with {len(n)} trials")
    continue
```

iii. From CONVERSION_NOTES: "Multi-plane sessions crashed ... Fixed by pooling planes via planeIdx" and "10 two-plane sessions had a 1-frame length mismatch ... caught by an assertion, fixed by trimming the trailing frame."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **NWB file I/O** (loading F, Fneu, behavior arrays) -- 0.4-2 s per session
2. **dF/F computation** (neuropil subtraction, baseline estimation, smoothing) -- 0.3-5.9 s per session
3. **Pickling the output** (9.6 GB of float32 data) -- ~14 s

ii. N/A

iii. With 12 parallel workers, the full conversion runs in ~46 s + 14 s pickling = 60 s total for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `compute_dff` and in the trial assembly section of `process_session` iterate over each trial sequentially. However, the variable trial lengths make full vectorization awkward (would require padding or masking). The speed-correlation computation IS vectorized (single matrix-vector product instead of a per-cell loop).

ii. N/A

iii. The AI notes the vectorized speed correlation is ~50x faster than a per-cell loop.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session exactly once (no separate survey step). However, within `process_session`, the trial boundaries are iterated twice: once for per-trial behavioral variable extraction (building `trial_rewarded`, `trial_env`, etc.) and once for building the final per-trial arrays. The dF/F computation also loops over trials multiple times (for masking, baseline, and smoothing).

ii. N/A

iii. The single-pass design avoids the overhead of loading NWB files twice (unlike a separate survey step).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `dff_all` (dF/F for all cells including interneurons) which is only used for diagnostic plots when `show_processing=True`. When plotting is disabled, this array persists in memory unused until the function returns. The `speed_corr` array for all cells is also computed but only the boolean mask is retained. The `zone_labels` sanity check (comparing scene-derived zones to observed positions) is useful for validation but has no effect on the output.

ii.
```python
dff_all = dff  # kept for potential plotting
dff = dff[keep_cells]  # only the filtered version is used
```

iii. These are minor inefficiencies; the dominant cost is I/O and the dF/F computation itself.
