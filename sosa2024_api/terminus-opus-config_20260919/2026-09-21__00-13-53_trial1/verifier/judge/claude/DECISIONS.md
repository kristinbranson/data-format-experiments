# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `pynwb.NWBHDF5IO` to load each NWB file. All NWB files are discovered by globbing `sub-*/sub-*.nwb` under the data directory. All 152 NWB files across 11 subjects are processed. Data is loaded via the `read_nwb_session()` function which reads behavioral time series, fluorescence, neuropil, and ROI metadata from each file.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
# ...
def read_nwb_session(path):
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        oph = nwb.processing['ophys'].data_interfaces
        # ... reads all behavioral and neural data
```

iii. The AI documented in CONVERSION_NOTES.md that all 152 NWB files (11 mice x 14 days, m11 has 12) are included. It verified the number of sessions matches expectations from the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `nwb.subject.subject_id` field within each NWB file. After processing all sessions, unique subjects are collected and sorted by their numeric ID.

ii.
```python
d['subject'] = nwb.subject.subject_id
# ...
subjects = sorted({inf['subject'] for inf in infos}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(inf['subject']) for inf in infos], dtype=np.int64)
```

iii. The AI verified 11 subjects match the paper. Subject IDs come directly from NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are discovered by globbing and sorted alphabetically. The AI processes all sessions (152 total) and skips any session with fewer than 2 usable trials.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
# ...
if len(n) < 2:
    print(f"  SKIPPING {info['file']}: only {len(n)} usable trials")
    continue
```

iii. The AI documented that each NWB file is one session, consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial_start` and `teleport` behavioral time series. Trial starts are frames where `trial_start > 0`, and trial ends are frames where `teleport > 0`. The trial window follows the reference convention `[start-1, stop-1)`. The `starts` array is also clamped to be >= 1 to avoid negative indexing.

ii.
```python
d['trial_starts'] = np.where(np.asarray(beh['trial_start'].data[:]) > 0)[0]
d['teleports'] = np.where(np.asarray(beh['teleport'].data[:]) > 0)[0]
# ...
starts = np.maximum(starts, 1)
# ...
sl = slice(s - 1, e - 1)
```

iii. The AI documented that trial boundaries match `glmUtils.get_timeseries_data` from the reference code, using `[trial_start-1, teleport-1)`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three ways: (1) lick-sensor error trials are dropped (>30% of frames with cumulative lick count > 2), (2) trials with fewer than 2 time points are dropped, (3) trials with non-finite neural data are dropped. The lick-error filter reproduces the paper's count of 81 excluded trials exactly.

ii.
```python
LICK_ERROR_FRAC = 0.30
LICK_ERROR_COUNT = 2
# ...
lick_error[i] = lk.size > 0 and (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC
# ...
if lick_error[i]:
    continue
# ...
if T < 2:
    continue
if not np.all(np.isfinite(ev_trial)):
    continue
```

iii. The AI documented the lick-error rule from the paper's methods and verified it reproduces exactly 81 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` and `Neuropil` time series from `processing/ophys`, NOT the pre-computed `Deconvolved` series. The AI correctly identified that the NWB's `Deconvolved` is suite2p's own deconvolution of raw fluorescence, not the signal the paper analyzes.

ii.
```python
planes = sorted(oph['Fluorescence'].roi_response_series.keys())
Fs, Fneus, roi_idx = [], [], []
for pl in planes:
    rrs = oph['Fluorescence'].roi_response_series[pl]
    Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
    Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:], dtype=np.float32))
```

iii. The AI verified that the NWB `Deconvolved` has r=0.38 with the paper's pipeline output and is not the correct signal.

## 2-b. How is the `neural` data processed?

i. The AI recomputes the paper's full dF/F + OASIS pipeline via `compute_events()`: (1) per-trial masking with NaN outside trials, (2) neuropil subtraction (F - 0.7*Fneu), (3) add back per-trial neuropil mean, (4) maximin baseline (Gaussian smooth sigma=15, min filter 300, max filter 300), (5) dF/F = (F-F0)/|F0|, (6) 2-sample Gaussian smoothing, (7) OASIS deconvolution with tau=0.7 and per-plane frame rate. The `keep_teleports` flag is set per animal/day from the teleport_metadata table.

ii.
```python
def compute_events(F, Fneu, starts, stops, fs, keep_teleports=False):
    starts, stops = baseline_segments(starts, stops, keep_teleports)
    # ... neuropil subtraction
    f_ -= NEU_COEF * fneu_
    # ... per-trial neuropil mean added back
    f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
    # ... maximin baseline
    seg = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH])
    seg = minimum_filter1d(seg, BASELINE_WINDOW, axis=-1)
    seg = maximum_filter1d(seg, BASELINE_WINDOW, axis=-1)
    # ... dF/F
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    # ... smoothing and deconvolution
    dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
    events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl]), 2000, TAU, fs)
```

iii. The AI documented that this is a faithful port of `preprocessing.dff()` from the reference code, matching the paper's methods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) only ROIs with `iscell == 1` (suite2p manual curation) are kept, (2) putative interneurons are excluded based on Pearson correlation between dF/F and running speed > 0.5. The interneuron detection is vectorized using matrix operations.

ii.
```python
d['iscell'] = iscell[roi_idx] == 1
F = raw['F'][raw['iscell']]
Fneu = raw['Fneu'][raw['iscell']]
# ...
dm_c = dm - dm.mean(axis=1, keepdims=True)
sp_c = sp - sp.mean()
denom = np.sqrt((dm_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
speed_corr = (dm_c @ sp_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
```

iii. The AI verified 0.35% of cells were excluded as interneurons, consistent with the paper's 0.42 +/- 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is "trial start". Since the trial data is extracted starting from the trial_start frame, no additional temporal shift is needed. The neural and behavioral data share the same frame indices within each trial.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
```

iii. The AI set `off_start = 0.0` and `off_end = None` in the metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is the native imaging frame period: 1000 / (scan_rate / n_planes) ms, which is approximately 64.48 ms (~15.5 Hz). No temporal rebinning is applied.

ii.
```python
d['fs'] = d['scan_rate'] / len(planes)  # per-plane sampling rate
# ...
'time_bin_size': float(1000.0 / infos[0]['fs']),  # ms
```

iii. The AI documented that the native frame rate is used without rebinning, consistent with the paper's statement that all time series were sampled at ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` of the behavioral time series (specifically `position.timestamps`), which are the imaging frame times.

ii.
```python
d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
# ...
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. The timestamps are on the imaging frame clock, which is the same clock as the neural data.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first frame in the trial is subtracted from all timestamps within the trial to get time relative to trial start.

ii.
```python
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. Simple subtraction of trial start time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices within each trial (both are indexed by the same `slice(s-1, e-1)`), so no additional alignment is needed.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
# ...
t_rel = time_v[sl] - time_v[s - 1]
```

iii. Both use the same slice of frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
d['env'] = np.asarray(beh['environment'].data[:], dtype=np.float64)
# ...
ev = env[sl]
ev = ev[ev >= 0]
env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
```

iii. The environment variable is 0 for ENV1 and 1 for ENV2, matching the binary encoding specified.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the median of valid (>= 0) environment samples within the trial is taken, then rounded to the nearest integer. This handles edge cases where the environment might have -1 values (outside laps). The value is broadcast as constant across all timepoints in the trial.

ii.
```python
ev = env[sl]
ev = ev[ev >= 0]
env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
# ...
inp[1] = env_trial[i]
```

iii. Taking the median handles the day-8 environment switch sessions where environment changes at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (0-based), derived from the loop counter over trials. The trial boundaries come from `trial_start` and `teleport`.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    # ...
    inp[2] = i
```

iii. The trial number is a sequential index within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. Note that lick-error trials and short trials are skipped but the original trial index `i` is still used (not re-indexed after filtering).

ii.
```python
inp[2] = i
```

iii. The trial number reflects the original trial index in the session, not the index after filtering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series (timestamps of reward delivery events) and the `reward_zone` behavioral time series. A trial is considered rewarded if a reward was delivered within the trial AND the reward zone was entered.

ii.
```python
d['reward_times'] = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
# ...
rew_frames = np.searchsorted(time_v, raw['reward_times'])
# ...
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. This matches the reference code's `behavior.get_trial_types` logic where `isreward = reward > 0 AND rzone > 0`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial outcome is looked up from the `isreward` array using `isreward[i-1]`. For the first trial (i=0), it is set to 0. The value is constant across all timepoints in the trial. Note: the "previous trial" is the trial with the previous index in the same session, regardless of whether it was a lick-error trial.

ii.
```python
inp[3] = isreward[i - 1] if i > 0 else 0
```

iii. Conservative choice for the first trial (no prior information available).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries. The reward zone for each trial is determined from the scene name in the NWB `identifier` field using `get_reward_zone_labels()`, which replicates `reward_relative.behavior.get_reward_zones`. Zone boundaries are: A=[80,130], B=[200,250], C=[320,370] cm.

ii.
```python
d['scene'] = nwb.identifier.split('/')[-1]
# ...
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
# ...
z0, z1 = ZONE_DICT[zlab]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. The AI uses the scene-based approach from the reference code's `behavior.get_reward_zones`, with zone switching at trial 30 on switch days.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance to the nearest edge of the reward zone is computed: negative if before the zone, 0 if inside, positive if past. The continuous distance is then discretized into 7 categories.

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    cat = np.zeros(pos.shape, dtype=np.int64)
    cat[d < -50] = 0
    cat[(d >= -50) & (d < -10)] = 1
    cat[(d >= -10) & (d < 0)] = 2
    cat[d == 0] = 3
    cat[(d > 0) & (d <= 10)] = 4
    cat[(d > 10) & (d <= 50)] = 5
    cat[d > 50] = 6
    return cat, d
```

iii. The distance computation follows the standard definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized using explicit conditional logic into 7 bins: 0 (< -50), 1 (-50 to -10), 2 (-10 to 0), 3 (exactly 0), 4 (>0 to 10), 5 (10 to 50), 6 (>50).

ii.
```python
cat[d < -50] = 0
cat[(d >= -50) & (d < -10)] = 1
cat[(d >= -10) & (d < 0)] = 2
cat[d == 0] = 3
cat[(d > 0) & (d <= 10)] = 4
cat[(d > 10) & (d <= 50)] = 5
cat[d > 50] = 6
```

iii. The bin edges match the decoder task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices within each trial (same `slice(s-1, e-1)`), so no additional alignment is needed.

ii.
```python
sl = slice(s - 1, e - 1)
p = pos[sl]
ev_trial = events[:, sl]
```

iii. Both indexed by the same slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
# ...
p = pos[sl]
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
```

iii. The `position` variable records position in cm along the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing. Position is discretized using `np.digitize` with edges [90, 180, 270, 360], producing 5 bins (0-4).

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
out[1] = pos_cat
```

iii. Five equal 90 cm bins spanning the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges [90, 180, 270, 360] produces 5 bins: 0 (< 90), 1 (90-180), 2 (180-270), 3 (270-360), 4 (> 360).

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
```

iii. Matches the specification of 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii. Same `sl = slice(s-1, e-1)` indexing.

iii. Verified by the AI through timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
d['lick'] = np.asarray(beh['lick'].data[:], dtype=np.float64)
# ...
lk = lick[sl]
lick_cat = (lk > 0).astype(np.int64)
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_cat = (lk > 0).astype(np.int64)
out[3] = lick_cat
```

iii. Instructions specify binary output (no/yes). Raw lick values can be > 1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii. Same `sl = slice(s-1, e-1)` indexing.

iii. Same frame clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `identifier` field in the NWB file, which contains the scene name (e.g., `Env1_LocationA`, `Env1_LocationA_to_C`). The `get_reward_zone_labels()` function parses the scene name to determine the zone (A, B, or C) and handles switch sessions by changing at trial 30.

ii.
```python
d['scene'] = nwb.identifier.split('/')[-1]
# ...
def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    if scene.endswith('LocationA'):
        return ['A'] * ntrials
    if scene.endswith('LocationB'):
        return ['B'] * ntrials
    if scene.endswith('LocationC'):
        return ['C'] * ntrials
    for first in ('A', 'B', 'C'):
        if f'{first}_to' in scene:
            second = scene[-1]
            n0 = min(change_trial, ntrials)
            return [first] * n0 + [second] * (ntrials - n0)
```

iii. This replicates `reward_relative.behavior.get_reward_zones` from the reference code. The AI verified that this approach matches the observed `reward_zone > 0` positions on every trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine zone identity. On switch sessions, the zone label changes at trial 30 (the `CHANGE_TRIAL` constant). The zone label is encoded as 0=A, 1=B, 2=C, and is constant within each trial.

ii.
```python
zone_code = {'A': 0, 'B': 1, 'C': 2}
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
# ...
out[4] = zone_code[zlab]
```

iii. Follows the reference code's approach exactly.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps and `reward_zone` behavioral time series. A trial is rewarded if a reward delivery event occurred within the trial AND the reward zone was entered during the trial.

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
# ...
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. Matches the reference code's `behavior.get_trial_types` logic.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward delivery timestamp falls within the trial frame range AND if the reward zone was entered. The value is constant across all timepoints in the trial. Binary: 0 = omitted, 1 = rewarded.

ii.
```python
isreward[i] = int(bool(has_rew) and bool(in_zone))
# ...
out[5] = isreward[i]
```

iii. The 15.8% omission rate matches the paper's ~15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled: (1) In 10 sessions (2-plane m17/m18), behavioral series are 1 sample shorter than ophys -- all streams are truncated to the common length. (2) Lick-sensor error trials (81 total) are dropped. (3) Trials with < 2 time points are dropped. (4) Trials with non-finite neural data are dropped. (5) Sessions with < 2 usable trials are dropped. (6) `trial_starts` are clamped to >= 1 to handle the `[start-1]` indexing convention.

ii.
```python
nB = len(d['time'])
nF = d['F'].shape[1]
n = min(nB, nF)
d['n_trunc'] = max(nB, nF) - n
if d['n_trunc'] > 0:
    for k in ('time', 'pos', 'speed', 'lick', 'rzone', 'env', 'trialnum', 'scanning'):
        d[k] = d[k][:n]
    d['F'] = d['F'][:, :n]
    d['Fneu'] = d['Fneu'][:, :n]
# ...
starts = np.maximum(starts, 1)
```

iii. The AI documented each edge case and verified that no trials are affected by the truncation.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files via pynwb (0.3-1.5 s/session, I/O bound), (2) dF/F computation + OASIS deconvolution (0.3-12 s/session depending on FOV size), (3) Saving the pickle file. Total wall clock with 12-worker parallelism: ~131 seconds for all 152 sessions.

ii. N/A

iii. The AI profiled timing per session and reported timing breakdowns.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially for input/output construction. Some discretization operations could be applied to full session arrays before splitting. The interneuron correlation was already vectorized (matrix multiply instead of per-cell loop).

ii. N/A

iii. The per-trial loop is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded only once per session (in `read_nwb_session`). There is no survey/pre-pass step that loads files separately -- the AI's code loads all data in a single pass per session. However, `nansmooth` is called multiple times within the dF/F computation per trial (once for baseline smoothing, once for dF/F smoothing).

ii. N/A

iii. The single-pass design avoids the repeated file loading seen in the reference solution.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `dff` (the full dF/F trace) but only uses the deconvolved `events` for the output. The dF/F is only used temporarily for interneuron detection (speed correlation) and then discarded. The code also reads `scanning` and `trialnum` from the NWB file but does not use them for the final output.

ii.
```python
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'], keep_teleports=keep_teleports)
# dff is used only for interneuron detection, then discarded
```

iii. The dF/F is a necessary intermediate for computing events and for interneuron detection. The extra fields are minor overhead.
