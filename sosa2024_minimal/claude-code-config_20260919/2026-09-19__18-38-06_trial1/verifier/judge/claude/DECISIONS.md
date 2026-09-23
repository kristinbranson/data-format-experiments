# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `/app/data/sub-*/sub-*.nwb`, sorts them by subject number, and loads each file using `h5py` (not `pynwb`). Each NWB file corresponds to one session. All 11 subjects and 152 sessions are loaded.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')),
               key=lambda p: (int(os.path.basename(p).split('_')[0][5:]),
                              os.path.basename(p)))
...
with h5py.File(path, 'r') as f:
    ...
    beh = f['processing/behavior/BehavioralTimeSeries']
    pos = beh['position/data'][:]
    speed = beh['speed/data'][:]
    ...
```

iii. The agent explored the data directory structure and NWB file format using h5py to inspect keys and datasets before writing the conversion script. It found 152 NWB files across 11 mouse subdirectories, consistent with the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file metadata (`general/subject/subject_id`). The subject list is built by collecting unique subject IDs across all results and sorting by numeric suffix.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(s[1:]))
```

iii. The agent read the subject ID from each NWB file's metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in parallel using `ProcessPoolExecutor` with spawn context. The experiment day is read from `general/session_id`.

ii.
```python
day = int(f['general/session_id'][()].decode())
...
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
        results.append(res)
```

iii. The agent determined that each NWB file represents a single session and used parallel processing for efficiency.

## 1-d. How are the data split into trials?

i. Trial boundaries are found from `trial_start` (nonzero indices) and `teleport` (nonzero indices). The AI then shifts both by -1 to match the reference `dff` function's indexing convention: `starts = trial_starts - 1`, `stops = teleports - 1`. Each trial's data is the slice `[starts[i], stops[i])`.

ii.
```python
trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
teleports = np.nonzero(beh['teleport/data'][:])[0]
# trial windows, matching preprocessing.dff: [start-1, stop-1)
starts = trial_starts - 1
stops = teleports - 1
```

iii. The agent studied the reference `preprocessing.dff` code and noted that it slices `[start-1, stop-1)`, so it pre-shifted the indices to match. The agent explicitly comments this matches the reference code's windowing.

## 1-e. How are trials filtered based on quality controls?

i. Two types of trials are filtered: (1) the first trial of each session is dropped because "previous trial outcome" is undefined, and (2) trials with a stuck lick sensor (>30% of samples with cumulative lick count > 2) are dropped. This flags exactly 81 trials, matching the paper's reported count.

ii.
```python
lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC
...
for i in range(ntrials):
    if i == 0:                      # previous trial outcome undefined
        continue
    if lick_error[i]:               # stuck lick sensor
        continue
```

iii. The agent noted that since licking is a decoder output, trials with stuck lick sensors should be entirely removed rather than NaN'd. The first trial removal is justified by the previous trial outcome input being undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) traces read from the NWB file. The NWB's `Deconvolved` field is explicitly NOT used — the agent identified it as suite2p's own deconvolution, not the paper's processing.

ii.
```python
F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
Fneu_list.append(f['processing/ophys/Neuropil'][plane]['data'][:, :].T[keep])
```

iii. The agent investigated the NWB file structure and found that the `Deconvolved` series contains non-zero values during the inter-trial interval, meaning it cannot be from the paper's within-trial dF/F processing.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's dF/F and deconvolution pipeline: (1) subtract 0.7 * Fneu from F, (2) add back mean neuropil per trial, (3) compute maximin baseline (Gaussian smooth sigma=15, min filter 300, max filter 300), (4) dF/F = (F - baseline) / |baseline|, (5) smooth with Gaussian sigma=2, (6) OASIS deconvolution with tau=0.7 and per-plane frame rate.

ii.
```python
def compute_dff_and_events(F, Fneu, starts, stops, frame_rate):
    for start, stop in zip(starts, stops):
        f = F[:, start:stop] - NEU_COEF * Fneu[:, start:stop]
        f = f + NEU_COEF * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
        flow = gaussian_filter1d(f, BASELINE_SIG, axis=-1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SIG, axis=-1)
        dff[:, start:stop] = d
        events[:, start:stop] = dcnv.oasis(d, 2000, TAU, frame_rate)
```

iii. The agent studied the reference code's `preprocessing.dff` and replicated the pipeline. Key differences from the reference implementation: (1) uses `gaussian_filter1d` instead of the reference's `nansmooth` (nan-aware Gaussian), (2) does NOT handle `keep_teleports` (the reference conditionally extends baseline windows across inter-trial periods based on `teleport_metadata.py`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) only suite2p-curated cells (`iscell == 1`) are kept, (2) putative interneurons are excluded based on dF/F-speed correlation > 0.5.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
keep = iscell[rois]
F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
...
# interneuron exclusion
speed_corr = (dff_c @ spd_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
```

iii. The agent implemented both the `iscell` filter and the interneuron exclusion from the paper's Methods, using vectorized correlation computation instead of the reference's per-cell loop.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. No additional alignment is needed since the trial slicing already starts at the trial start index. The `off_start` is set to 0.0.

ii.
```python
s, t = starts[i], stops[i]
...
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
...
'off_start': 0.0,
```

iii. Since each trial's data begins at the trial start frame, alignment is implicit in the slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate is used. The time bin size is computed as the median inter-timestamp interval (~64.484 ms). The AI verifies all sessions have the same dt.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
...
dts = {round(r['dt'], 6) for r in results}
assert len(dts) == 1, f'inconsistent time bins: {dts}'
...
'time_bin_size': float(round(list(dts)[0] * 1000, 4)),
```

iii. The agent found that all sessions share the same sampling rate once multi-plane sessions are accounted for.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the sample index within the trial and the median inter-timestamp interval `dt`. Not from explicit timestamps.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
...
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. The agent computed time as `sample_index * dt` rather than using the stored timestamps directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as `np.arange(T) * dt` where T is the number of samples in the trial and dt is the median timestamp interval. This assumes uniform sampling.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. This creates evenly-spaced time values starting at 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (same indexing into the data arrays), so alignment is implicit. The AI verifies the frame count matches.

ii.
```python
nframes = len(pos)
assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
```

iii. Both neural and behavioral data use the same sampling, verified by assertion.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = beh['environment/data'][:]
...
env_vals = np.unique(env[s:t])
assert len(env_vals) == 1
envs[i] = int(env_vals[0])
...
inp[1] = envs[i]
```

iii. The agent verified each trial has a single environment value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No processing beyond reading the value and casting to int. An assertion verifies the environment is constant within each trial.

ii.
```python
env_vals = np.unique(env[s:t])
assert len(env_vals) == 1
envs[i] = int(env_vals[0])
```

iii. Environment is already binary (0 or 1) in the raw data.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` over all trials in the session (including dropped ones), i.e. the original trial index within the session.

ii.
```python
inp[2] = i
```

iii. The trial number preserves the original trial index rather than a renumbered index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — just the loop counter. The value is constant across all timepoints in the trial.

ii.
```python
inp[2] = i
```

iii. Simple assignment of the trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior time series. The AI defines "rewarded" as both being in the reward zone AND receiving a reward delivery.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
...
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
...
inp[3] = rewarded[i - 1]
```

iii. The agent used the paper's definition of rewarded trials (reward in zone), as in `behavior.get_trial_types`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, `rewarded[i]` is 1 if the reward zone flag was active AND a reward timestamp fell within the trial's time window. The previous trial outcome is then `rewarded[i-1]`. The first trial is skipped entirely (not included in output).

ii.
```python
for i in range(ntrials):
    if i == 0:  # previous trial outcome undefined
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. Rather than setting the first trial's previous outcome to 0 (as the reference does), the AI drops the first trial entirely since the value is truly undefined.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone boundaries (A: 80-130, B: 200-250, C: 320-370 cm). The reward zone for each trial is determined from the `reward_zone` flag in the behavior data, with a switch placed at trial 30 on switch sessions.

ii.
```python
REWARD_ZONES = {0: (80.0, 130.0), 1: (200.0, 250.0), 2: (320.0, 370.0)}
...
in_zone = rzone[s:t] > 0
if in_zone.any():
    observed_zone[i] = zone_label(pos[s:t][in_zone].min())
...
z0, z1 = REWARD_ZONES[zone_of_trial[i]]
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
```

iii. The agent used the paper's reward zone dictionary and a simple threshold-based zone labeling (`zone_label` function) rather than the reference's Viterbi algorithm approach.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to nearest edge of reward zone: negative before, 0 inside, positive after. Uses `np.where` for vectorized computation.

ii.
```python
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using the `discretize_distance` function with explicit conditional assignments matching the instruction bins.

ii.
```python
def discretize_distance(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The bin edges match the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data — both use the same `[starts[i], stops[i])` slice.

ii.
```python
s, t = starts[i], stops[i]
p = pos[s:t]
...
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
```

iii. Implicit alignment via shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:t]
```

iii. Direct read from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond slicing per trial.

ii.
```python
p = pos[s:t]
out[1] = np.digitize(p, POS_EDGES)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges `[90, 180, 270, 360]`. This produces values 0-4 directly (no -1 adjustment needed since the edges don't include -inf/inf).

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. The 5 equal bins span the 450 cm track at 90 cm each.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
p = pos[s:t]
```

iii. Implicit alignment via shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh['lick/data'][:]
...
lk = (lick[s:t] > 0).astype(np.int64)
```

iii. Direct read from NWB file.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
...
out[3] = lk
```

iii. Instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
```

iii. Implicit alignment via shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position`. The zone for each trial is determined by where the reward zone flag is active, then classified using a simple position-threshold function (`zone_label`). On switch sessions, the switch is placed at trial 30.

ii.
```python
def zone_label(zone_start):
    if zone_start < 150:
        return 0
    if zone_start < 280:
        return 1
    return 2
...
if in_zone.any():
    observed_zone[i] = zone_label(pos[s:t][in_zone].min())
...
if np.all(labels == labels[0]):
    zone_of_trial[:] = int(labels[0])
else:
    ...
    zone_of_trial[:switch_trial] = int(labels[0])
    zone_of_trial[switch_trial:] = int(labels[-1])
```

iii. The agent used a simpler threshold-based approach compared to the reference's Viterbi algorithm, and hardcodes the switch at trial 30 from the paper's protocol.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Zone is identified per trial from position when reward_zone flag > 0, classified by thresholding the minimum position. For switch sessions, the switch boundary at trial 30 is used. The value is mapped to 0=A, 1=B, 2=C and is constant per trial.

ii.
```python
out[4] = zone_of_trial[i]
```

iii. The zone label is a per-trial constant.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior time series.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
...
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
...
out[5] = rewarded[i]
```

iii. The agent requires both being in the reward zone AND receiving a reward delivery, matching the paper's `get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if reward delivered while in reward zone during this trial, 0 otherwise. The value is constant per trial.

ii.
```python
rewarded[i] = int(in_zone.any() and got_reward)
out[5] = rewarded[i]
```

iii. Matches the paper's definition where reward outcome requires both zone presence and reward delivery.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Neural/behavior length mismatch: if F has one more frame than behavior data, it is cropped to match (asserted to be at most 1 frame difference). (2) Stuck lick sensor trials are dropped entirely. (3) Trials with no observed reward zone get NaN and are filled by the switch logic (using the zone from surrounding trials). (4) First trial of each session is dropped.

ii.
```python
nframes = len(pos)
assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
F = F[:, :nframes]
Fneu = Fneu[:, :nframes]
...
lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC
```

iii. The agent handled the known data issues found during exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files and reading large arrays (I/O bound), (2) Computing dF/F and OASIS deconvolution for each session, (3) Writing the large pickle file. The AI mitigates this with parallel processing using `ProcessPoolExecutor`.

ii.
```python
ctx = multiprocessing.get_context('spawn')
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
```

iii. The agent chose parallelism to speed up the I/O and compute-bound steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for building input/output arrays could theoretically be vectorized, but variable trial lengths make this awkward. The interneuron exclusion is already vectorized (matrix multiply for correlation). The `compute_dff_and_events` iterates per trial for the baseline computation, which is inherent to per-trial processing.

ii. N/A

iii. The per-trial loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Each session is only loaded once (unlike the reference which loads twice: survey + conversion). The AI's parallelized approach processes each session independently in a single pass.

ii. N/A

iii. The AI's design avoids the reference's duplicate NWB file loading.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes full dF/F traces but only uses the deconvolved events for the neural output. The dF/F is used for interneuron exclusion (speed correlation) and then discarded. Some per-session metadata (scene, date, switch_trial info) is computed and stored but not used by the decoder.

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
del F, Fneu
...
del dff, dff_v, dff_c  # only events are kept
```

iii. The dF/F computation is necessary as an intermediate step for both the events and the interneuron filter, even though only events are kept.
