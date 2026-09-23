# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script loads all data by globbing every `*.nwb` file one directory below `/app/data`, treating each file as one session. It reads each NWB file directly with `h5py`, processes each session independently in `process_session`, writes a per-session pickle to `/app/sessions_out`, and then reassembles those per-session outputs into `/app/converted_data.pkl`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
...
with h5py.File(fn, 'r') as f:
    ident = f['identifier'][()].decode()
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
...
with ctx.Pool(nproc, maxtasksperchild=1) as p:
    for _ in p.imap_unordered(process_session, files):
        pass
assemble()
```

iii. In the trajectory, the agent first enumerated `/app/data/*/*.nwb`, concluded there were 152 session files across 11 mice, and decided to process each file independently. It explicitly switched to direct `h5py` access and parallel per-session conversion after exploring the NWB structure and profiling runtime.

## 1-b. How are the data split into subjects?

i. Subjects are determined from each NWB file's stored subject identifier, and the final `subjects` list is built from the unique subject IDs encountered while assembling processed sessions.

ii.
```python
with h5py.File(fn, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
...
subjects = []
...
if m['subject'] not in subjects:
    subjects.append(m['subject'])
...
data['subject_idx'].append(subjects.index(m['subject']))
```

iii. In the trajectory, the agent noted that the dataset is organized one directory per mouse and that the same subject identity is also stored inside each NWB file. It chose to trust the NWB metadata during assembly.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script reads the session ID from the file metadata and stores one entry in the output lists per processed file.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
...
with h5py.File(fn, 'r') as f:
    session_id = f['general/session_id'][()].decode()
...
for fn in files:
    pkl = os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl'))
    ...
    data['neural'].append(r['neural'])
    data['input'].append(r['input'])
    data['output'].append(r['output'])
```

iii. In the trajectory, the agent repeatedly summarized the dataset as "152 sessions" and described the work unit as one NWB file per session, which drove both processing and assembly.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport`. The script takes every index where `trial_start > 0` as a trial start and every index where `teleport > 0` as a trial stop, then slices each trial as `[start:stop)`.

ii.
```python
trial_start = get('trial_start')
teleport = get('teleport')
...
starts = np.where(trial_start > 0)[0]
stops = np.where(teleport > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
...
for t in range(n_trials):
    s, e = starts[t], stops[t]
```

iii. In the trajectory, the agent stated several times that the paper pipeline uses "trials = trial_start→teleport" and that it verified trial boundary conventions by checking session structure and position traces.

## 1-e. How are trials filtered based on quality controls?

i. The script drops two kinds of trials: the first trial of every session, because it treats previous-trial outcome as undefined there, and any trial with a lick-sensor error, defined as more than 30% of samples in the trial having `lick > 2`. It does not implement a minimum-trial-length filter.

ii.
```python
LICK_ERR_THRESH = 0.3
...
lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
...
for t in range(n_trials):
    tr = trials[t]
    if t == 0 or tr['lick_err']:
        continue
```

iii. In the trajectory, the agent justified dropping lick-error trials by reference to the paper's lick-sensor correction logic and by its own dataset scan, where it found 81 such trials. It also decided to drop first trials because it preferred not to encode an undefined previous-trial outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the raw `Fluorescence` and `Neuropil` datasets in the NWB file, not from the NWB `Deconvolved` dataset. Only ROIs marked as `iscell` are kept before further processing.

ii.
```python
planes = sorted(f['processing/ophys/Fluorescence'].keys())
F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = ps['iscell'][:, 0] > 0
...
F = F[iscell]
Fneu = Fneu[iscell]
```

iii. In the trajectory, the agent explicitly decided that the NWB `Deconvolved` array was not the paper's analysis signal, after inspecting its values and comparing them to the paper code. It therefore chose to recompute the paper-style events from `Fluorescence` and `Neuropil`.

## 2-b. How is the `neural` data processed?

i. The script pools all imaging planes, then for each trial computes a per-trial dF/F-like signal from `F` and `Fneu`: subtract `0.7 * Fneu`, add back each trial's mean neuropil, smooth with a Gaussian (`sigma=15` samples), apply a 300-sample minimum filter and then a 300-sample maximum filter to form the maximin baseline, compute `(F - baseline) / |baseline|`, smooth again with `sigma=2`, convert NaNs/infs to zero, and deconvolve with `suite2p.extraction.dcnv.oasis` using `tau=0.7` and `fs = 1 / median(diff(position_timestamps))`.

ii.
```python
def compute_events(F, Fneu, starts, stops, fs):
    ...
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

iii. In the trajectory, the agent repeatedly described this as the paper pipeline: dF/F with maximin baseline plus OASIS deconvolution, using `tau=0.7` and neuropil subtraction 0.7. It chose this after reading the paper code and decoder notebook and after rejecting the stored `Deconvolved` series as the wrong signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only `iscell` ROIs, then removes putative interneurons by computing each cell's correlation with running speed across in-trial samples and dropping cells with correlation greater than `0.5`.

ii.
```python
iscell = ps['iscell'][:, 0] > 0
...
F = F[iscell]
Fneu = Fneu[iscell]
...
sp_ = speed[in_trial]
d_ = dff[:, in_trial]
...
r = np.divide((d_c * s_c).sum(axis=1), denom, out=np.zeros(d_.shape[0]), where=denom > 0)
keep_cells = r <= INT_R_THRESH
events = events[keep_cells]
```

iii. In the trajectory, the agent explicitly called out two QC steps from the paper: `iscell==1` manual curation and putative-interneuron exclusion at `r > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by the way trials are sliced. For each kept trial, the neural matrix is simply `events[:, start:stop]`, so timepoint 0 in each trial corresponds to `trial_start`.

ii.
```python
for t in range(n_trials):
    ...
    s, e = tr['s'], tr['e']
    T = e - s
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. In the trajectory, the agent described the alignment event as "trial start" and treated trial segmentation itself as the alignment mechanism, rather than adding any extra temporal shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data keep the native frame rate. The time step is computed as the median spacing of the behavioral timestamps (`dt = median(diff(position/timestamps))`), and no temporal rebinning or resampling is applied.

ii.
```python
tstamps = b['position/timestamps'][()]
...
dt = float(np.median(np.diff(tstamps)))
fs = 1.0 / dt
...
time_s = np.arange(T, dtype=np.float32) * dt
...
'time_bin_size': float(np.mean(dts)) * 1000.0,
```

iii. In the trajectory, the agent decided to keep the native sampling rate after estimating dataset size and decoder feasibility, and it explicitly rejected adding coarser time binning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This variable is derived from the trial boundaries and the behavioral timestamp spacing. The code reads `position/timestamps`, computes a single session-level `dt`, and then uses the length of each trial to generate `0, dt, 2dt, ...`.

ii.
```python
tstamps = b['position/timestamps'][()]
...
dt = float(np.median(np.diff(tstamps)))
...
T = e - s
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. In the trajectory, the agent reasoned that the behavioral and neural data are already frame-aligned at a constant rate, so using the native frame spacing was sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code does not subtract the actual first timestamp in each trial. Instead, it creates a synthetic per-trial clock by multiplying the sample index within the trial by the session-level `dt`.

ii.
```python
T = e - s
time_s = np.arange(T, dtype=np.float32) * dt
inp.append(np.stack([
    time_s,
    ...
]).astype(np.float32))
```

iii. In the trajectory, the agent justified this by treating the timestamps as uniformly sampled and already aligned to imaging frames.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction because it has exactly one entry per neural frame in the same `[start:stop)` trial slice. The trial length `T` comes from the same indices used to slice the neural matrix.

ii.
```python
s, e = tr['s'], tr['e']
T = e - s
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. In the trajectory, the agent repeatedly stated that the neural and behavior streams share the same frame grid once the off-by-one mismatch is cropped away.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env = get('environment')
...
trials.append(dict(..., env=int(np.round(np.median(env[s:e])))))
```

iii. In the trajectory, the agent examined per-trial environment values and concluded that environment is effectively constant within a trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code takes the median of `environment[s:e]`, rounds it to an integer, and then fills the entire trial with that constant value.

ii.
```python
trials.append(dict(..., env=int(np.round(np.median(env[s:e])))))
...
np.full(T, tr['env'], dtype=np.float32)
```

iii. In the trajectory, the agent justified this as a per-trial variable and described environment as stable within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial loop index `t` after trials have been segmented from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    np.full(T, t, dtype=np.float32),
```

iii. In the trajectory, the agent summarized trial number as the within-session trial index after defining trials from `trial_start` to `teleport`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No further processing is applied. The within-session trial index is broadcast across all timepoints in that trial.

ii.
```python
inp.append(np.stack([
    time_s,
    np.full(T, tr['env'], dtype=np.float32),
    np.full(T, t, dtype=np.float32),
    ...
]).astype(np.float32))
```

iii. In the trajectory, the agent treated trial number as a simple sequential covariate and did not describe any transformation beyond the loop index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/timestamps` series together with the trial boundaries. Reward timestamps are mapped onto the behavioral frame grid using `np.searchsorted`.

ii.
```python
reward_times = b['Reward/timestamps'][()]
...
reward_idx = np.searchsorted(tstamps, reward_times)
...
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
```

iii. In the trajectory, the agent explicitly described omission/reward detection as coming from reward timestamps and trial windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial after the first, the code asks whether any mapped reward timestamp fell within the previous trial. It then fills the current trial with that previous trial's binary reward outcome. The first trial is not assigned a value; it is dropped entirely.

ii.
```python
trials.append(dict(..., rewarded=rewarded, ...))
...
if t == 0 or tr['lick_err']:
    continue
...
np.full(T, trials[t - 1]['rewarded'], dtype=np.float32),
```

iii. In the trajectory, the agent stated that the first trial would be excluded because previous-trial outcome is undefined there.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavioral `position` time series and a per-trial reward-zone label inferred from the scene name in the NWB `identifier`. The `reward_zone` behavioral time series is read only for a sanity check and is not used to assign the per-trial zone label.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
def scene_zones(scene):
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]
...
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
...
p = pos[s:e]
lo, hi = ZONES[tr['zone']]
```

iii. In the trajectory, the agent first explored the `reward_zone` timeseries and then concluded that the paper code assigns reward zones from the scene name plus a switch at trial 30. It said it verified this rule against the reward-zone entry positions across sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the code computes signed distance to the current trial's reward zone: negative before the zone, zero inside it, positive after it. It does this by subtracting the lower edge when position is before the zone and subtracting the upper edge when position is after the zone.

ii.
```python
p = pos[s:e]
lo, hi = ZONES[tr['zone']]
d = np.zeros(T)
d[p < lo] = p[p < lo] - lo
d[p > hi] = p[p > hi] - hi
```

iii. In the trajectory, the agent described this as a deliberate deviation from the paper's circular reward-relative coordinates to satisfy the task's requested signed linear distance in centimeters.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is done with explicit comparisons into seven bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. In the trajectory, the agent explicitly noted that the task required the seven linear distance bins from the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial `[start:stop)` slice as the neural data, so there is one distance category per neural frame.

ii.
```python
s, e = tr['s'], tr['e']
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
...
p = pos[s:e]
...
out.append(np.stack([
    rd, pbin, sbin, lk,
    ...
]).astype(np.int64))
```

iii. In the trajectory, the agent consistently described all behavioral outputs as already sampled on the same frame grid as neural activity within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
pos = get('position')
...
p = pos[s:e]
```

iii. In the trajectory, the agent inspected the behavioral streams and treated `position` as the direct VR corridor position signal.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position trace is clipped to `[0, 450)` and then discretized into 90 cm bins using `np.digitize`.

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. In the trajectory, the agent described this as using the 450 cm track and the five equal-width bins required by the task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholding uses the four internal edges `90`, `180`, `270`, and `360` cm, producing five bins.

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. In the trajectory, the agent tied these thresholds directly to the decoder specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position[start:stop]` on the same indices used for the neural trial slice.

ii.
```python
s, e = tr['s'], tr['e']
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
...
p = pos[s:e]
```

iii. In the trajectory, the agent described position as already frame-aligned with the imaging stream within each trial.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = get('lick')
...
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. In the trajectory, the agent explored the `lick` values and used them both for the binary lick output and for detecting lick-sensor-error trials.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes the lick trace frame-by-frame: values greater than zero become `1`, and zero becomes `0`.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. In the trajectory, the agent treated the task's lick output as a binary presence/absence variable.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the lick trace on the same `[start:stop)` indices as the neural trial.

ii.
```python
s, e = tr['s'], tr['e']
...
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. In the trajectory, the agent treated lick as already sampled on the same imaging-rate time base as the other behavioral series.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` string, which contains the scene name, plus the fixed rule that the zone changes at trial 30 for switch sessions. The behavioral `reward_zone` signal is not used to assign the label.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
...
np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64)
```

iii. In the trajectory, the agent explicitly said it was using the scene name and the paper's `change_trial=30` logic after verifying this against the data.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses the scene string into an initial and final reward zone label, applies the first label for trials `t < 30` and the second for trials `t >= 30`, maps `A/B/C` to `0/1/2`, and broadcasts the per-trial zone label across all frames in that trial.

ii.
```python
def scene_zones(scene):
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]
...
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
...
np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64)
```

iii. In the trajectory, the agent justified this as the paper-consistent rule implemented in `behavior.get_reward_zones`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, mapped onto the behavioral time base and checked against each trial's `[start:stop)` interval.

ii.
```python
reward_times = b['Reward/timestamps'][()]
reward_idx = np.searchsorted(tstamps, reward_times)
...
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
```

iii. In the trajectory, the agent described omission/reward detection as "trials with no reward event versus trials with a reward event".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code uses `np.searchsorted` to place reward timestamps onto the imaging/behavior frame grid, then labels each trial as rewarded if any mapped reward event falls inside that trial. The result is constant within each trial.

ii.
```python
reward_idx = np.searchsorted(tstamps, reward_times)
...
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
...
np.full(T, tr['rewarded'], dtype=np.int64),
```

iii. In the trajectory, the agent justified this as matching the binary rewarded-versus-omitted trial structure in the task and the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles one mismatch explicitly: if imaging has one extra frame relative to behavior, it truncates `F` and `Fneu` to the common length. During neural preprocessing it also replaces NaN and infinite dF/F values with zero before deconvolution. It does not implement broader missing-data handling such as short-trial removal or fallback reward-zone inference.

ii.
```python
# a few sessions have one extra imaging frame relative to the VR timeseries;
# truncate to the common length so imaging and behavior stay aligned
n_common = min(F.shape[1], len(pos))
F = F[:, :n_common]
Fneu = Fneu[:, :n_common]
...
d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. In the trajectory, the agent discovered off-by-one imaging/behavior mismatches in a few sessions, patched the code to crop to the common length, and then reran the full conversion. It did not describe any more general missing-data strategy.

## 13-a. What are the most time-consuming steps of the code?

i. The code's main expensive steps are reading the large fluorescence/neuropil arrays from every NWB file, running the per-trial dF/F and OASIS deconvolution in `compute_events`, and writing/reading the intermediate per-session pickles before the final assembly.

ii.
```python
F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
...
dff, events = compute_events(F, Fneu, starts, stops, fs)
...
with open(os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl')), 'wb') as fo:
    pickle.dump(res, fo, protocol=4)
```

iii. In the trajectory, the agent profiled the pipeline, tested single-session runtime, and explicitly discussed I/O cost, multiprocessing overhead, and the cost of the per-trial neural preprocessing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial loop in `compute_events`, the loop that marks `in_trial`, and the per-trial loop that constructs `trials`, `neural`, `inp`, and `out`. These loops repeatedly slice session arrays trial-by-trial and could only partly be vectorized.

ii.
```python
for s, e in zip(starts, stops):
    ...
for s, e in zip(starts, stops):
    in_trial[s:e] = True
...
for t in range(n_trials):
    ...
    trials.append(...)
...
for t in range(n_trials):
    ...
    neural.append(...)
    inp.append(...)
    out.append(...)
```

iii. In the trajectory, the agent accepted the per-trial structure because trial-wise preprocessing and variable trial lengths are central to the design, but it did profile the code and think about efficiency tradeoffs.

## 13-c. What processing does the code repeat multiple times?

i. The script repeats session-level disk I/O by first writing each processed session to `/app/sessions_out/*.pkl` and then immediately reading those files back in `assemble`. Within `process_session`, it also loops over the trial boundaries once to build trial metadata and again to build the final decoder arrays.

ii.
```python
with open(os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl')), 'wb') as fo:
    pickle.dump(res, fo, protocol=4)
...
for fn in files:
    pkl = os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl'))
    ...
    with open(pkl, 'rb') as fi:
        r = pickle.load(fi)
...
for t in range(n_trials):
    ...
    trials.append(...)
...
for t in range(n_trials):
    ...
    neural.append(...)
```

iii. In the trajectory, the agent chose the per-session-pickle design to make parallel processing and final assembly simpler, even though it meant rereading processed session outputs.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores the full-session `dff` array even though the final dataset only keeps deconvolved `events`; `dff` is used transiently for interneuron filtering and then discarded. It also reads `reward_zone` and computes `n_zone_mismatch` only as a sanity check, without using that signal in the final outputs.

ii.
```python
dff = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
events = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
...
dff, events = compute_events(F, Fneu, starts, stops, fs)
...
d_ = dff[:, in_trial]
...
del dff, d_, d_c
...
rzone_ts = get('reward_zone')
...
n_zone_mismatch = 0
...
if not (lo - 15 <= p_in.min() <= hi + 15):
    n_zone_mismatch += 1
```

iii. In the trajectory, the agent said it used `reward_zone` only to verify that the scene-derived labels matched the data, and the code shows that the check is retained only in metadata rather than in downstream model inputs or outputs.
