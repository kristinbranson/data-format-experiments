# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans every subject directory under `/app/data`, collects every `.nwb` file, sorts them by subject and session number, and converts each file as one session. It loads the NWB contents directly with `h5py` rather than `pynwb`.

ii. 
```python
files = []
for sub in sorted(os.listdir(DATA_ROOT)):
    d = os.path.join(DATA_ROOT, sub)
    if os.path.isdir(d):
        files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                          int(re.search(r'ses-(\d+)', p).group(1))))

with h5py.File(path, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
    exp_day = int(f['general/session_id'][()].decode())
```

iii. The trajectory’s final summary says the dataset includes the “full DANDI:001361 switch cohort” with 152 sessions and 11 mice, so the intent was to include all available session files and not subsample the dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the directory / file naming pattern `sub-m<N>`, and the final `subjects` list is the sorted set of those IDs.

ii. 
```python
subjects = sorted({re.search(r'sub-(m\d+)', p).group(1) for p in files},
                  key=lambda s: int(s[1:]))
data['subjects'] = subjects
```

iii. The trajectory’s final summary reports “11 mice,” which matches the decision to treat each `sub-m*` directory / file prefix as one subject.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session, ordered by subject number and session number (`ses-XX`).

ii. 
```python
files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                          int(re.search(r'ses-(\d+)', p).group(1))))
...
for path in files:
    cache = results.get(path)
    if cache is None:
        continue
    with open(cache, 'rb') as fh:
        res = pickle.load(fh)
    data['neural'].append(res['neural'])
```

iii. The trajectory’s final summary reports “152 sessions,” which is consistent with using one NWB file per session.

## 1-d. How are the data split into trials?

i. Trials are treated as laps. Trial starts are frames where `trial_start > 0`; trial ends are the frames where `teleport > 0`. Each emitted trial is sliced as `[trial_start - 1, teleport - 1)`, which is one-sample shifted relative to the raw markers so it matches the AI’s reading of the paper’s `dff()` slicing.

ii. 
```python
trial_starts = np.flatnonzero(beh['trial_start/data'][:] > 0)
teleports = np.flatnonzero(beh['teleport/data'][:] > 0)
...
for i in range(ntrials):
    a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
    if b - a < 2:
        continue
    neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
```

iii. The trajectory explicitly says “Trials are laps, sliced `[trial_start−1, teleport−1)` — exactly the slice the reference `dff()` uses, which excludes the interpolated teleport frame.”

## 1-e. How are trials filtered based on quality controls?

i. The AI drops four kinds of trials / sessions: the first lap of every session, lick-sensor artifact laps, laps shorter than 2 bins, and laps whose neural slice contains non-finite values. Entire sessions are also dropped if fewer than 2 usable trials remain.

ii. 
```python
for i in range(ntrials):
    if i == 0:
        continue
    if lick_error[i]:
        continue
    a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
    if b - a < 2:
        continue
    neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
    if not np.all(np.isfinite(neu)):
        continue
...
if len(neural_trials) < 2:
    return None
```

iii. The trajectory gives two explicit justifications: lick-artifact laps are dropped because the paper’s lick-sensor error criterion flags exactly 81 such laps and “since lick is a required output I drop the lap,” and the first imaged lap is dropped because the preceding warm-up lap is unobserved so `previous_trial_outcome` is genuinely unknown.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw fluorescence and neuropil traces, not from the NWB `Deconvolved` dataset.

ii. 
```python
plane_names = sorted(f['processing/ophys/Fluorescence'].keys())
F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                    for p in plane_names], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil'][p]['data'][:nframes, :]
                       for p in plane_names], axis=1).T
```

iii. The trajectory explicitly says “The NWB `Deconvolved` field is suite2p's own `spks` from raw F. The paper instead computes its own dF/F and deconvolves that.”

## 2-b. How is the `neural` data processed?

i. The AI computes per-session dF/F and deconvolved events from `F` and `Fneu`: neuropil subtraction with coefficient 0.7, add back the trial mean neuropil, Gaussian smoothing before a maximin baseline, compute `(F - baseline) / |baseline|`, smooth dF/F with a 2-sample Gaussian, and deconvolve with OASIS. The baseline segments either span trial only or trial-plus-teleport depending on `TELEPORT_SESSIONS`.

ii. 
```python
f_ -= NEU_COEF * fneu_
...
f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
seg = ndi.gaussian_filter1d(f_[:, a:b], BASELINE_SMOOTH_SIG, axis=1)
seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
flow[:, a:b] = seg
...
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
...
dff[:, a:b] = ndi.gaussian_filter1d(dff[:, a:b], DFF_SMOOTH_SIG, axis=1)
events[:, a:b] = dcnv.oasis(np.ascontiguousarray(dff[:, a:b], dtype=np.float32),
                            2000, TAU, FRAME_RATE)
```

iii. The trajectory’s final summary lists this processing almost verbatim and says it was chosen to reproduce `reward_relative.preprocessing.dff` from the paper’s repository.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are first filtered by suite2p manual curation (`iscell`), then putative interneurons are removed if their dF/F is too correlated with running speed (`r > 0.5`).

ii. 
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
...
F = np.ascontiguousarray(F[iscell], dtype=np.float64)
Fneu = np.ascontiguousarray(Fneu[iscell], dtype=np.float64)
...
speed_corr = (dv @ spc) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
```

iii. The trajectory explicitly justifies this as “suite2p `iscell`, then putative interneurons with speed–dF/F Pearson r > 0.5,” matching the paper’s cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial relative to the lap boundaries; no extra interpolation or temporal shift is applied after trial extraction.

ii. 
```python
for i in range(ntrials):
    a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
    neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
    neural_trials.append(neu)
```

iii. The trajectory says the alignment event is lap start and that trials are emitted as lap-aligned slices `[trial_start−1, teleport−1)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The AI uses a fixed frame rate of 15.5078125 Hz, so the time bin size is `1000 / 15.5078125 = 64.484 ms`.

ii. 
```python
FRAME_RATE = 15.5078125
DT = 1.0 / FRAME_RATE
...
'time_bin_size': 1000.0 / FRAME_RATE,
```

iii. The trajectory explicitly says “Variable length is kept ... bin size is constant at 64.484 ms.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `position/timestamps`.

ii. 
```python
ts = beh['position/timestamps'][:]
...
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
inp[0] = t_rel
```

iii. The trajectory does not separately justify choosing `position/timestamps`; it only states that trials are frame-aligned and use the constant imaging bin size.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the first sample is subtracted so the trial starts at 0 seconds.

ii. 
```python
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
inp[0] = t_rel
```

iii. The trajectory does not discuss this separately; the processing is implicit in the trial-alignment choice.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the exact same slice `[a:b]` as the neural data for each trial, so the timestamps and neural bins stay frame-aligned.

ii. 
```python
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
...
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
```

iii. The trajectory frames the whole conversion around lap-aligned imaging frames and does not mention any extra alignment step.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii. 
```python
env_ts = beh['environment/data'][:]
...
env_vals = np.unique(env_ts[a:b])
environment[i] = int(env_vals[0])
```

iii. The trajectory’s final summary describes the task as occurring in “one of two visually distinct environments (ENV 1 / ENV 2),” which is the rationale for keeping environment as a per-trial decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI asserts that the environment is constant within each trial, converts it to an integer, stores one value per trial, and broadcasts that scalar across all timepoints in the input matrix.

ii. 
```python
env_vals = np.unique(env_ts[a:b])
assert len(env_vals) == 1, f'{path}: trial {i} has environments {env_vals}'
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. The trajectory does not discuss this separately beyond treating environment as a per-trial variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over extracted laps, i.e. the ordinal position of the lap within the session after trial boundaries are found.

ii. 
```python
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The trajectory says "`trial_number` still carries the true ordinal (starts at 1)" because the script drops the first imaged lap before emitting trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond assigning the per-lap index and repeating that scalar over all time bins in the trial.

ii. 
```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[2] = i
```

iii. The trajectory’s only explicit justification is that the value should preserve the “true ordinal.”

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps, which are mapped onto imaging / behavior frames and summarized as a per-lap rewarded / not rewarded flag.

ii. 
```python
reward_ts = beh['Reward/timestamps'][:]
...
rew_frames = np.searchsorted(ts, reward_ts)
rewarded = np.zeros(ntrials, dtype=np.int64)
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. The trajectory explicitly says “Reward outcome uses the `Reward` event timestamps mapped to imaging frames.”

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes whether each lap was rewarded, then sets `previous_trial_outcome` for lap `i` to `rewarded[i - 1]`. It does not fabricate a value for the first imaged lap; instead, it drops that lap entirely.

ii. 
```python
for i in range(ntrials):
    if i == 0:
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. The trajectory explicitly justifies dropping the first lap because “the preceding lap is one of the un-imaged warm-up laps, so `previous_trial_outcome` is genuinely unknown.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s position (`position/data`) and a per-trial reward-zone identity inferred from the session’s scene name rather than from the `reward_zone` timeseries.

ii. 
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
...
p = pos[a:b]
zstart, zstop = REWARD_ZONES[zone_labels[i]]
out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. The trajectory explicitly says “Reward zone per lap comes from the scene name plus the 30-lap switch point,” and says this was validated against rewarded-lap zone entries with “zero mismatches.”

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside it, positive after it. It then converts that signed distance into 7 discrete classes.

ii. 
```python
def reward_zone_distance_bin(pos, zstart, zstop):
    d = np.zeros_like(pos)
    before = pos < zstart
    after = pos > zstop
    d[before] = pos[before] - zstart
    d[after] = pos[after] - zstop
    out = np.full(pos.shape, 3, dtype=np.int64)
```

iii. The trajectory’s final summary describes this as decoding “the signed distance to the reward zone (7 bins).”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded into 7 categories with explicit comparisons implementing the instruction bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii. 
```python
out[d < -50.0] = 0
out[(d >= -50.0) & (d < -10.0)] = 1
out[(d >= -10.0) & (d < 0.0)] = 2
out[(d > 0.0) & (d <= 10.0)] = 4
out[(d > 10.0) & (d <= 50.0)] = 5
out[d > 50.0] = 6
```

iii. The trajectory does not justify the implementation details separately; it only states that the output is the signed distance in 7 bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same trial slice `[a:b]` as the neural data and evaluating position frame-by-frame within that slice.

ii. 
```python
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
...
p = pos[a:b]
out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. The trajectory does not describe any extra alignment step beyond lap-based slicing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii. 
```python
pos = beh['position/data'][:]
...
p = pos[a:b]
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The trajectory’s final summary describes the task as decoding “absolute track position (5 bins).”

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the raw position trace per trial and discretizes it into 5 bins spanning the 450 cm track.

ii. 
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
p = pos[a:b]
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The trajectory does not provide a separate justification beyond following the decoder task specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It uses `np.digitize` with edges `[90, 180, 270, 360]`, which yields categories 0-4 corresponding to `<90`, `90-180`, `180-270`, `270-360`, and `>=360`.

ii. 
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The trajectory does not separately justify the thresholding implementation.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned to neural data by taking the exact same trial slice `[a:b]`.

ii. 
```python
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
p = pos[a:b]
```

iii. The trajectory treats all time-varying outputs as frame-aligned within each lap.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii. 
```python
lick = beh['lick/data'][:]
...
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The trajectory explicitly mentions licking as one of the decoded framewise outputs.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick signal is binarized so any positive value becomes 1 and zero stays 0.

ii. 
```python
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The trajectory says lick is a required binary output, which is why artifact laps are dropped instead of leaving corrupted lick values in the dataset.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing `lick[a:b]` with the same indices used for the neural activity.

ii. 
```python
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The trajectory does not mention any further alignment step.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `identifier` / scene name, which is parsed into per-trial zone labels using hard-coded zone-switch rules.

ii. 
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
...
out[4] = ZONE_LABELS.index(zone_labels[i])
```

iii. The trajectory explicitly says reward-zone identity comes from the scene name and the 30-lap switch point, validated against the data.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses scene names of the forms `Env*_LocationX`, `Env*_LocationX_to_Y`, or `Env*_X_to_Env*_Y`. If the scene encodes a switch, it assigns the first reward zone for the first 30 trials and the second zone after that, then converts `A/B/C` to integers 0/1/2.

ii. 
```python
def scene_reward_zones(scene, ntrials):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return [m.group(1)] * ntrials
    m = (re.fullmatch(r'Env\d_Location([ABC])_to_([ABC])', scene)
         or re.fullmatch(r'Env\d_([ABC])_to_Env\d_([ABC])', scene))
    if m:
        return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
...
out[4] = ZONE_LABELS.index(zone_labels[i])
```

iii. The trajectory explicitly justifies this by saying it mirrors `behavior.get_reward_zones` and matched rewarded-lap entries with “zero mismatches.”

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` event timestamps.

ii. 
```python
reward_ts = beh['Reward/timestamps'][:]
rew_frames = np.searchsorted(ts, reward_ts)
...
rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
...
out[5] = rewarded[i]
```

iii. The trajectory explicitly notes that the script uses `Reward` timestamps because the `Reward/data` values are all zeros and therefore not useful.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices with `np.searchsorted`. Each trial is marked rewarded if any mapped reward event falls within that trial’s raw `[trial_start, teleport)` frame interval. The resulting per-trial binary label is then broadcast across the whole emitted trial.

ii. 
```python
rew_frames = np.searchsorted(ts, reward_ts)
rewarded = np.zeros(ntrials, dtype=np.int64)
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
...
out[5] = rewarded[i]
```

iii. The trajectory justifies this by noting that the mean rewarded fraction is consistent with the task’s programmed omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles problems by asserting invariants and skipping bad trials / sessions rather than repairing them. It asserts equal frame counts and one environment per trial, skips trials with too few bins or non-finite neural values, skips sessions with zero neurons or fewer than two usable trials, and catches worker exceptions so failed sessions are omitted.

ii. 
```python
assert F.shape[1] == nframes and len(trial_starts) == len(teleports)
...
assert len(env_vals) == 1, f'{path}: trial {i} has environments {env_vals}'
...
if b - a < 2:
    continue
...
if not np.all(np.isfinite(neu)):
    continue
...
if n_neurons == 0:
    return None
if len(neural_trials) < 2:
    return None
...
try:
    res = convert_session(path)
except Exception:
    return path, None, traceback.format_exc()
```

iii. The trajectory explicitly justifies two of these skips: lick-artifact trials are dropped because lick is a required output, and first laps are dropped because the previous outcome is unknown. It does not provide separate justification for the other assertions / skips.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work in this script is session-level conversion: reading large behavior / fluorescence arrays from HDF5, computing dF/F and OASIS events, and pickling per-session results plus the final aggregated dataset. The use of multiprocessing and a session cache implies the AI viewed those steps as the runtime bottlenecks.

ii. 
```python
with h5py.File(path, 'r') as f:
    ...
    F = np.concatenate([...], axis=1).T
    Fneu = np.concatenate([...], axis=1).T
...
dff, events = compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports)
...
with ctx.Pool(args.workers) as pool:
    for n, (path, cache, err) in enumerate(pool.imap_unordered(_worker, files), 1):
...
with open(tmp, 'wb') as fh:
    pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory does not say this explicitly, but the presence of `--workers`, a per-session cache, and the final report of writing a 9.5 GB pickle all imply that conversion and serialization were treated as the expensive parts.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining loop targets are the per-segment loops in `compute_dff_and_events`, the per-trial loop that computes reward / environment / lick-artifact summaries, and the per-trial assembly loop that slices each trial and writes inputs / outputs.

ii. 
```python
for a, b in segments:
    f_[:, a:b] = F[:, a:b]
    fneu_[:, a:b] = Fneu[:, a:b]
...
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
    ...
for i in range(ntrials):
    ...
    neural_trials.append(neu)
    input_trials.append(inp)
    output_trials.append(out)
```

iii. The trajectory does not discuss vectorization explicitly.

## 13-c. What processing does the code repeat multiple times?

i. The code avoids a separate survey pass, but it still repeats some work: each session is converted and pickled once, then re-opened from the cache during final aggregation; trial slicing logic is repeated across neural, input, and output construction; and trial boundaries are used in multiple loops.

ii. 
```python
cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
...
with open(tmp, 'wb') as fh:
    pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
...
with open(cache, 'rb') as fh:
    res = pickle.load(fh)
...
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    ...
for i in range(ntrials):
    ...
```

iii. The trajectory does not discuss repeated processing explicitly; the cache suggests the AI was trying to avoid costlier repeated NWB reads.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is not much obviously unnecessary work. The main intermediate that is computed and then discarded is the full dF/F array, which is only kept long enough to classify putative interneurons before only the deconvolved events are saved. The script also builds detailed session metadata that are not needed by the decoder itself.

ii. 
```python
dff, events = compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports)
...
valid = ~np.isnan(dff[0, :])
...
events = events[keep_cells]
del dff, dv
...
info = {
    'subject': subject,
    'experiment_day': exp_day,
    'scene': scene,
    ...
}
```

iii. The trajectory does not call this out as wasteful; the dF/F intermediate is justified there as necessary for matching the paper’s interneuron filtering.
