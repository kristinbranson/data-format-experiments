# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `/app/data/*/\*.nwb`. Each NWB file is opened with `h5py` (not `pynwb`). Behavior time series are read from `processing/behavior/BehavioralTimeSeries`, neural data from `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. Sessions are processed in parallel using `multiprocessing.Pool` with spawn context, writing per-session pickle files, then assembled into the final dataset.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
# ...
with h5py.File(fn, 'r') as f:
    ident = f['identifier'][()].decode()
    scene = ident.split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    b = f['processing/behavior/BehavioralTimeSeries']
    get = lambda k: b[k + '/data'][()]
    pos = get('position')
    speed = get('speed')
    # ...
```

iii. From trajectory step 61: "Write /app/convert_data.py that processes each NWB session in parallel and writes per-session pickles, then assembles /app/converted_data.pkl." The AI chose h5py over pynwb for direct HDF5 access, and used multiprocessing for speed.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from each NWB file's `general/subject/subject_id` field. During assembly, unique subjects are collected in order of first appearance.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
# ...
if m['subject'] not in subjects:
    subjects.append(m['subject'])
```

iii. The AI reads the subject ID directly from the NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from the NWB `general/session_id` field. Sessions are sorted by file path (alphabetical within each subject directory).

ii.
```python
session_id = f['general/session_id'][()].decode()
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
```

iii. From trajectory step 33: "All sessions are single-plane, ~80 trials each, 15.5 Hz sampling, 152 sessions, 11 mice."

## 1-d. How are the data split into trials?

i. Trial starts are identified where `trial_start > 0`. Trial ends are identified where `teleport > 0`. An assertion checks that starts and stops have equal length and that stops come after starts.

ii.
```python
starts = np.where(trial_start > 0)[0]
stops = np.where(teleport > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
```

iii. From trajectory step 49: "trials = trial_start→teleport". The AI verified trial boundary conventions through data inspection.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: (1) The first trial of each session is dropped because "previous trial outcome" is undefined for it. (2) Trials with lick-sensor errors are dropped: if more than 30% of imaging samples in the trial have a cumulative lick count > 2, the trial is considered to have a faulty lick sensor and is excluded.

ii.
```python
lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH  # LICK_ERR_THRESH = 0.3
# ...
if t == 0 or tr['lick_err']:
    continue  # no previous-trial outcome / unusable lick data
```

iii. From trajectory step 52: "lick sensor error trials = 81 (matches paper)". Step 78: "Conversion matches paper: 81 lick-error trials (paper: 81)". The AI also justified dropping the first trial because the decoder input "previous trial outcome" is undefined for it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) traces, not from the NWB's `Deconvolved` field. The AI explicitly recognized that the NWB `Deconvolved` is suite2p's own deconvolution, not the paper's signal.

ii.
```python
planes = sorted(f['processing/ophys/Fluorescence'].keys())
F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
```

iii. From trajectory step 34: "NWB Deconvolved appears to be suite2p's raw-F deconvolution (values ~300, no NaNs outside trials), not the paper's custom dF/F→OASIS events."

## 2-b. How is the `neural` data processed?

i. The AI implements a `compute_events` function that follows the paper's `preprocessing.dff` pipeline per trial: neuropil subtraction (coefficient 0.7), add back trial mean neuropil, maximin baseline (Gaussian smooth sigma=15, 300-sample min filter, 300-sample max filter), compute dF/F = (F - baseline) / |baseline|, Gaussian smooth with sigma=2, then OASIS deconvolution (tau=0.7). Cells from multiple planes are concatenated.

ii.
```python
def compute_events(F, Fneu, starts, stops, fs):
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
        fneu = Fneu[:, s:e].astype(np.float64)
        f = f - NEU_COEF * fneu + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
        flow = gaussian_filter1d(f, BASELINE_SMOOTH, axis=1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
```

iii. From trajectory step 42: "paper pipeline: trials from trial_start to teleport, dF/F computed per trial (neuropil subtract 0.7, maximin baseline, 2-sample smoothing), deconvolved via OASIS -> 'events' used as neural data."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Cells are restricted to those with `iscell == 1` (suite2p's manual curation). (2) Putative interneurons are excluded: cells whose dF/F correlates with running speed at r > 0.5 are dropped.

ii.
```python
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][:, 0] > 0
F = F[iscell]
Fneu = Fneu[iscell]
# ...
r = np.divide((d_c * s_c).sum(axis=1), denom, out=np.zeros(d_.shape[0]), where=denom > 0)
keep_cells = r <= INT_R_THRESH  # INT_R_THRESH = 0.5
events = events[keep_cells]
```

iii. From trajectory step 78: "0.29% interneurons excluded (paper: 0.42 +/- 0.85%)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. No additional processing is needed beyond splitting into trials using the trial_start and teleport indices.

ii.
```python
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. From trajectory step 49: "Temporally align based on start of the trial" - the trial boundaries define the alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native imaging rate (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median of timestamp differences across sessions.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
fs = 1.0 / dt
# ...
data['metadata'] = dict(
    time_bin_size=float(np.mean(dts)) * 1000.0,  # ms
```

iii. From trajectory step 33: "15.5 Hz sampling". The AI verified consistent rates across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the imaging frame index within the trial and the median timestamp interval `dt`.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
# ...
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. The AI uses computed dt rather than actual timestamps for time calculation.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, an array `[0, dt, 2*dt, ..., (T-1)*dt]` is created, where T is the number of timepoints in the trial and dt is the median inter-frame interval.

ii.
```python
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. This creates a uniformly-spaced time array starting at 0 for each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices. Both are indexed by the same `s:e` (start:end) range for each trial.

ii.
```python
s, e = tr['s'], tr['e']
T = e - s
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. Alignment is implicit since both neural and behavioral variables use the same trial indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
env = get('environment')
# ...
trials.append(dict(..., env=int(np.round(np.median(env[s:e])))))
```

iii. The environment variable records which virtual environment the mouse was in for each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the environment values within each trial is computed and rounded to an integer. This gives a single per-trial value.

ii.
```python
env=int(np.round(np.median(env[s:e])))
# ...
np.full(T, tr['env'], dtype=np.float32),
```

iii. The AI takes the median rather than a single sample, which is robust to potential edge effects.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop index `t` over trials within a session.

ii.
```python
for t in range(n_trials):
    # ...
    np.full(T, t, dtype=np.float32),
```

iii. The trial number is the sequential 0-based index of the trial within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop counter. The value is constant across all timepoints within a trial.

ii.
```python
np.full(T, t, dtype=np.float32),
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file. Reward event times are mapped to imaging frame indices using `np.searchsorted`. For each trial, whether any reward event occurred is determined.

ii.
```python
reward_times = b['Reward/timestamps'][()]
reward_idx = np.searchsorted(tstamps, reward_times)
# ...
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
```

iii. The Reward time series has its own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial `t > 0`, the previous trial's `rewarded` flag is used. For `t == 0`, the trial is dropped entirely (not included in the dataset).

ii.
```python
if t == 0 or tr['lick_err']:
    continue
# ...
np.full(T, trials[t - 1]['rewarded'], dtype=np.float32),
```

iii. The AI drops the first trial because the previous trial outcome is undefined, rather than setting it to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone is determined from the NWB `identifier` field (scene name), parsed via `scene_zones()`, with a switch at trial index 30 (CHANGE_TRIAL).

ii.
```python
def scene_zones(scene):
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]
# ...
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
```

iii. From trajectory step 20: "Scene name is in NWB identifier, allowing reward zone label determination." Step 52: "switch always at trial 30 (scene-based)."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from animal position to the nearest edge of the reward zone: negative before the zone, 0 inside, positive after. Uses zone boundaries from `ZONES` dictionary.

ii.
```python
lo, hi = ZONES[tr['zone']]
d = np.zeros(T)
d[p < lo] = p[p < lo] - lo
d[p > hi] = p[p > hi] - hi
```

iii. The distance is 0 inside the reward zone, negative when the animal is before the zone start, and positive after the zone end.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit boolean indexing rather than `np.digitize`:
- 0: d < -50
- 1: -50 <= d < -10
- 2: -10 <= d < 0
- 3: d == 0 (default, inside zone)
- 4: 0 < d <= 10
- 5: 10 < d <= 50
- 6: d > 50

ii.
```python
rd = np.full(T, 3, dtype=np.int64)
rd[d < -50] = 0
rd[(d >= -50) & (d < -10)] = 1
rd[(d >= -10) & (d < 0)] = 2
rd[(d > 0) & (d <= 10)] = 4
rd[(d > 10) & (d <= 50)] = 5
rd[d > 50] = 6
```

iii. The default value 3 corresponds to "0 cm" (inside the reward zone), and the bins match the instruction specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data — both use the same `s:e` trial slice.

ii.
```python
s, e = tr['s'], tr['e']
p = pos[s:e]
```

iii. Alignment is implicit through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = get('position')
# ...
p = pos[s:e]
```

iii. The position variable records the animal's location in cm along the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450 - epsilon] then discretized into 5 bins using `np.digitize` with edges [90, 180, 270, 360].

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. Clipping ensures all positions fall within the 5 expected bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm each: 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360. The `np.digitize` with edges [90, 180, 270, 360] naturally produces bins 0-4.

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. Matches the instruction specification of 5 equal bins spanning 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
p = pos[s:e]
```

iii. Shared indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = get('lick')
# ...
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. Shared indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (scene name). The scene name encodes the reward zone configuration (e.g., `Env1_LocationB_to_A`). The `scene_zones()` function parses the last character of each part around `_to_` to get zones before and after a switch.

ii.
```python
def scene_zones(scene):
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]
# ...
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
```

iii. From trajectory step 20-21: The AI examined the `behavior.py` code to understand reward zone derivation from scene names and the switch at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A, B, or C) is mapped to an integer index (0, 1, 2) using a dictionary lookup.

ii.
```python
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
# ...
np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64),
```

iii. Simple label-to-index mapping.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file. Reward event times are mapped to behavior frame indices using `np.searchsorted`.

ii.
```python
reward_times = b['Reward/timestamps'][()]
reward_idx = np.searchsorted(tstamps, reward_times)
# ...
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
```

iii. The Reward time series has separate timestamps indicating when reward was delivered.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward event timestamp falls within the trial's time range [s, e). If yes, reward=1; otherwise reward=0.

ii.
```python
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
# ...
np.full(T, tr['rewarded'], dtype=np.int64),
```

iii. Binary per-trial reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Imaging and behavior are truncated to the common length (a few 2-plane sessions have one extra imaging frame).
- **Lick sensor errors**: Trials with >30% of samples having cumulative lick count >2 are excluded (81 trials total, matching the paper).
- **First trial**: Dropped because previous trial outcome is undefined.
- **Sessions with <2 trials**: Skipped during assembly.
- **NaN in dF/F**: Replaced with 0 via `np.nan_to_num`.

ii.
```python
n_common = min(F.shape[1], len(pos))
F = F[:, :n_common]
Fneu = Fneu[:, :n_common]
# ...
d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
# ...
if len(r['neural']) < 2:
    print('skipping session with <2 trials:', pkl)
    continue
```

iii. From trajectory step 72: "Some sessions have a 1-frame mismatch between imaging and behavior lengths."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** (h5py reads of large arrays) - I/O bound
2. **Computing dF/F and OASIS deconvolution** per trial - CPU bound
3. **Multiprocessing overhead** (spawn context, per-session pickle I/O)

ii. N/A

iii. The AI uses multiprocessing with 12 worker processes and spawn context to parallelize session processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `compute_events` iterates over each trial for baseline computation and deconvolution. The interneuron correlation computation iterates per cell. The trial assembly loop iterates over trials for building decoder arrays.

ii.
```python
for s, e in zip(starts, stops):
    # per-trial dF/F and deconvolution
# ...
for c in range(dff.shape[0]):
    # per-cell speed correlation (though AI vectorized this)
```

iii. The per-trial processing is inherently sequential due to per-trial baselines, but the interneuron detection was partially vectorized by the AI.

## 13-c. What processing does the code repeat multiple times?

i. Session processing reads each NWB file once (unlike a survey+conversion approach). However, positions within each trial are accessed during both the zone-mismatch sanity check loop and the final output building loop.

ii. N/A

iii. The parallel processing design avoids redundant file I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes full dF/F traces (stored in the `dff` array) which are only used for interneuron detection (speed correlation). After that check, the dF/F is discarded and only the deconvolved events are kept. The zone mismatch sanity check loop also does work that doesn't contribute to the final output.

ii.
```python
dff, events = compute_events(F, Fneu, starts, stops, fs)
# dff used only for interneuron check, then discarded
del dff, d_, d_c
```

iii. The dF/F is needed for interneuron detection but is not part of the final output.
