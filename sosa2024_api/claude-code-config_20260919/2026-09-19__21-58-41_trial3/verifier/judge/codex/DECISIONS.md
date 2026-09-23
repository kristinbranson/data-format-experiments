# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads every NWB file under `/app/data/sub-*/*.nwb`, then reads each file with `pynwb.NWBHDF5IO` inside `read_session()`. Each file becomes one converted session.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. `CONVERSION_NOTES.md` Step 2 says the dataset layout is one NWB file per mouse-day under `sub-<mouse>/`, with 152 files total, and Step 6 says `read_session` is the intended `pynwb` loader.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `nwb.subject.subject_id` in each file, then uniqued and numerically sorted when assembling the final dataset.

ii.
```python
subject = nwb.subject.subject_id
```

```python
subjects = sorted({r['info']['subject'] for r in results},
                  key=lambda s: int(s[1:]))
...
'subject_idx': np.array([subjects.index(r['info']['subject']) for r in results],
                        dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 2 documents that NWB subject IDs are values like `m11`, and the final assembly uses those IDs rather than directory names.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. The session label is read from `nwb.session_id`, but session membership is determined by the file list itself.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
```

```python
session_id = nwb.session_id
...
return dict(path=path, scene=scene, subject=subject, session_id=session_id, ...)
```

iii. `CONVERSION_NOTES.md` Step 2 explicitly describes `/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` as one NWB file per mouse-day.

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` and `teleport` behavioral streams. The code finds `trial_start == 1` and `teleport == 1`, then uses the frame window `[trial_start - 1, teleport - 1)` for every trial.

ii.
```python
si = np.where(raw['trial_start'] == 1)[0]
ti = np.where(raw['teleport'] == 1)[0]
...
sl = slice(s - 1, e - 1)
```

```python
# keep_teleports=False -> only on-track samples, window [start-1, stop-1)
slices = [slice(s - 1, t - 1) for s, t in zip(trial_starts, teleports)]
```

iii. The module docstring and `CONVERSION_NOTES.md` Step 5 say this was chosen to mirror the reference code window and to exclude the teleport frame.

## 1-e. How are trials filtered based on quality controls?

i. The code drops trials flagged as lick-sensor-error trials, trials with `scanning != 1`, trials whose activity block has fewer than 2 samples, trials with non-finite neural values, and any trial whose `[start-1, end-1)` window would run out of bounds.

ii.
```python
keep = (si >= 1) & (ti <= len(ts))
...
lick_error[i] = (np.sum(L > LICK_ERROR_COUNT_THR) / len(L)) > LICK_ERROR_FRAC_THR
scan_bad[i] = np.any(scanning[s - 1:e - 1] != 1)
...
if lick_error[i]:
    n_dropped_lick += 1
    continue
if scan_bad[i]:
    n_dropped_scan += 1
    continue
...
if act.shape[1] < 2 or not np.all(np.isfinite(act)):
    n_dropped_nan += 1
    continue
```

iii. `CONVERSION_NOTES.md` Steps 4-6 justify dropping lick-error trials because `lick` is a decoder output, note that `scanning` should be all ones, and document the out-of-range and non-finite checks as edge-case handling.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural signal is derived from raw `Fluorescence` and `Neuropil` ROI response series, not from the NWB `Deconvolved` field.

ii.
```python
plane_names = sorted(oph['Fluorescence'].roi_response_series.keys())
...
rrs = oph['Fluorescence'].roi_response_series[pn]
...
nrs = oph['Neuropil'].roi_response_series[pn]
...
f = np.concatenate(f_list, axis=0)
f_neu = np.concatenate(fneu_list, axis=0)
```

iii. The module docstring and `CONVERSION_NOTES.md` Steps 1, 4, and 5 explicitly say the NWB `Deconvolved` array is not the paper’s analyzed signal, so the agent recomputed the paper’s signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The code recomputes per-trial dF/F with the paper’s `preprocessing.dff` logic: subtract `0.7 * Fneu`, add back the per-trial neuropil mean, estimate a maximin baseline, compute `(F - baseline)/abs(baseline)`, smooth with a 2-sample Gaussian, and also run OASIS deconvolution. However, the stored default neural signal is `dff`, not the deconvolved `events`.

ii.
```python
def dff_and_events(f, f_neu, trial_starts, teleports, frame_rate,
                   neu_coef=NEU_COEF, tau=OASIS_TAU):
    ...
    f_ -= neu_coef * f_neu_
    ...
    f_[:, sl] = f_[:, sl] + neu_coef * np.nanmean(f_neu_[:, sl], axis=1, keepdims=True)
    base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
    base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
    base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
    ...
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    spks[:, sl] = dcnv.oasis(np.ascontiguousarray(smoothed, dtype=np.float32),
                             OASIS_BATCH, tau, frame_rate)
```

```python
ap.add_argument('--signal', choices=['dff', 'events'], default='dff')
...
act = (events if signal == 'events' else dff)[:, sl]
```

iii. `CONVERSION_NOTES.md` Step 8 says the agent compared `events` against `dff` on sample decoder runs and switched the default to `dff` because it decoded better for this task; trajectory step 129 states that switch explicitly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only manually curated `iscell` ROIs at load time, then removes putative interneurons whose dF/F is correlated with running speed above 0.5.

ii.
```python
iscell = np.asarray(seg['iscell'].data[:])[:, 0].astype(bool)
...
sel = iscell[rid]
...
sp_v = speed[valid].astype(np.float32)
d_v = dff[:, valid]
...
r_speed = (d_c @ sp_c) / denom
...
keep_cells = r_speed <= INTERNEURON_SPEED_CORR_THR
dff = dff[keep_cells]
events = events[keep_cells]
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 cite the paper’s neuron-curation rules: `iscell == 1` and exclusion of putative interneurons with `r(dF/F, speed) > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by trial slicing itself: each trial uses `[trial_start - 1, teleport - 1)`, so the saved neural block begins one frame before the `trial_start` event.

ii.
```python
slices = [slice(s - 1, t - 1) for s, t in zip(trial_starts, teleports)]
...
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
```

iii. `CONVERSION_NOTES.md` Step 5 says the alignment event is trial start, but also states the chosen trial window is `[si[i]-1, ti[i]-1)` because the agent believed that matched the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native imaging/behavior frame grid. The time bin is the median behavioral timestamp step, about 64.48 ms, with no further rebinning.

ii.
```python
dt = float(np.median(np.diff(ts)))
frame_rate = 1.0 / dt
```

```python
'time_bin_size': float(np.median(dts) * 1000.0),
```

iii. `CONVERSION_NOTES.md` Steps 2, 3, 5, and 9 repeatedly state that the NWB behavior is already aligned to imaging frames and that the native frame period is 64.4836 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the common behavioral timestamp vector, specifically `b['position'].timestamps`.

ii.
```python
b = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
```

```python
inp[0] = ts[sl] - ts[s]
```

iii. `CONVERSION_NOTES.md` Step 2 says all behavioral time series share a common timestamp vector, so the code uses one of those aligned timestamp streams directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the timestamp at `trial_start` (`ts[s]`) from every timestamp in the saved trial slice. Because the slice itself starts at `s - 1`, the first saved value is negative by one frame.

ii.
```python
sl = slice(s - 1, e - 1)
...
inp[0] = ts[sl] - ts[s]                 # t = 0 at the trial_start frame
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly notes that `time_from_trial_start` starts at about `-0.0645 s` because the stored window includes the pre-start sample.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by using the exact same per-trial slice `sl` as the neural activity block.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
...
inp[0] = ts[sl] - ts[s]
```

iii. The code constructs `act`, `p`, `sp_t`, `lk`, and `inp[0]` from the same `sl`, so the time axis and neural matrix have the same length and sample boundaries.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
morph = raw['environment']
```

```python
mvals = np.unique(morph[s:e])
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
```

iii. `CONVERSION_NOTES.md` Step 5 maps `environment`/`morph` directly to the decoder input `environment`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code reduces the within-trial environment samples to a single per-trial integer: it uses the unique value if the trial is constant, otherwise it falls back to the rounded median. That scalar is then broadcast across all saved timepoints in the trial.

ii.
```python
mvals = np.unique(morph[s:e])
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
...
inp[1] = trial_morph[i]
```

iii. `CONVERSION_NOTES.md` Step 5 says environment is treated as a per-trial variable and broadcast over time; the fallback median is simply defensive handling for unexpected within-trial variation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not taken from the NWB `trial number` stream. Instead it is derived from the loop index over detected trials after splitting on `trial_start`/`teleport`.

ii.
```python
for i, (s, e) in enumerate(zip(si, ti)):
    ...
    inp[2] = i
```

iii. `CONVERSION_NOTES.md` Step 5 says `trial_number` is the 0-based within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied: the code writes the detected trial index `i` and broadcasts it across the trial’s timepoints.

ii.
```python
inp[2] = i
```

iii. The agent’s notes describe this as a per-trial scalar input stored in time-varying `(d, T)` form by broadcasting.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps (`Reward.timestamps`) and the `reward_zone` flag. The code first builds a per-trial `isreward` using both signals, then shifts that trial-level vector by one trial.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
...
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
prev_reward = np.zeros(ntrials_all, dtype=np.int64)
prev_reward[1:] = isreward[:-1]
```

iii. `CONVERSION_NOTES.md` Steps 4 and 5 say this follows `behavior.get_trial_types`, and note that in these NWB files `reward_zone` is only raised on rewarded trials, so the conjunction reduces to the same rewarded/omitted distinction.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code computes `isreward` for every trial, then shifts it forward by one trial so trial `i` gets the previous trial’s reward outcome; the first trial is set to 0.

ii.
```python
prev_reward = np.zeros(ntrials_all, dtype=np.int64)
prev_reward[1:] = isreward[:-1]
...
inp[3] = prev_reward[i]
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says the first trial is undefined and therefore set to 0, and that `prev_reward` is computed before trial dropping so filtering does not corrupt the sequence.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from behavioral `position` plus per-trial reward-zone boundaries inferred from the NWB scene name (`nwb.identifier`), not from the `reward_zone` binary trace.

ii.
```python
scene = nwb.identifier.split('/')[-1]
...
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
zone_start = np.array([REWARD_ZONE_DICT[z][0] for z in zone_lab])
zone_end = np.array([REWARD_ZONE_DICT[z][1] for z in zone_lab])
```

```python
p = pos[sl]
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. `CONVERSION_NOTES.md` Steps 4 and 5 say this mirrors `behavior.get_reward_zones` from the paper’s code and was cross-checked against measured reward-zone-entry positions on 10,394/10,394 trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes a signed distance from position to the nearest point in the active reward zone: 0 inside the zone, negative before it, positive after it. It then discretizes that continuous distance into decoder classes.

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

```python
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. `CONVERSION_NOTES.md` Step 5 maps this variable exactly that way and notes that the discretization is a new step imposed by the decoder specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses seven hard-coded classes: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
DIST_EDGES = (-50.0, -10.0, 0.0, 10.0, 50.0)
...
out = np.full(d.shape, 3, dtype=np.int64)
out[d < 0] = 2
out[d < DIST_EDGES[1]] = 1
out[d < DIST_EDGES[0]] = 0
out[d > 0] = 4
out[d > DIST_EDGES[3]] = 5
out[d > DIST_EDGES[4]] = 6
```

iii. The code comments and `CONVERSION_NOTES.md` Step 5 both say these bins were chosen to match the decoder task exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance output is aligned by using the same per-trial slice `sl` and the same number of timepoints as the saved neural matrix.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
...
p = pos[sl]
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. The output is computed from `p = pos[sl]`, where `sl` is the same slice used for `act`, so distance and neural activity are sample-aligned inside each saved trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
pos = raw['position']
...
p = pos[sl]
```

iii. `CONVERSION_NOTES.md` Steps 2 and 5 identify `position` as the behavioral track-coordinate signal in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code takes the trial slice of raw position and discretizes it into five bins spanning the track.

ii.
```python
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. `CONVERSION_NOTES.md` Step 5 says the 450 cm track is split into five equal 90 cm bins as required by the decoder task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded with `np.digitize` at 90, 180, 270, and 360 cm, yielding five classes.

ii.
```python
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. `CONVERSION_NOTES.md` Step 5 states these are exactly the task-specified equal-width bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned with neural data by slicing both from the same trial window `sl`.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
...
p = pos[sl]
out[1] = np.digitize(p, POS_EDGES)
```

iii. The same `sl` drives both the neural and position blocks, so they share sample-for-sample alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = raw['lick']
...
lk = lick[sl]
```

iii. `CONVERSION_NOTES.md` Step 2 describes `lick` as the cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes the lick trace: any value greater than zero becomes 1, otherwise 0.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 cites the paper’s `behavior.lickrate` behavior of first binarizing licks with `licks[licks>0]=1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same saved trial slice `sl` as the neural activity block.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
...
lk = lick[sl]
out[3] = (lk > 0).astype(np.int64)
```

iii. Lick and neural activity are both indexed by `sl`, so they share the same per-trial sample grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB scene name in `nwb.identifier`, parsed into labels A/B/C through `zone_labels_from_scene`.

ii.
```python
scene = nwb.identifier.split('/')[-1]
...
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)
```

```python
out[4] = zone_code[i]
```

iii. `CONVERSION_NOTES.md` Steps 4 and 5 say this was chosen because it directly mirrors `behavior.get_reward_zones` in the paper’s code and matched observed zone-entry positions perfectly in the data scan.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses fixed scenes ending in `LocationA/B/C` and switch scenes containing `<zone1>_to` and ending in the final zone letter. On switch scenes it assigns the first zone for the first 30 trials and the final zone afterward, then maps A/B/C to integer classes 0/1/2.

ii.
```python
def zone_labels_from_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    for z in ZONE_ORDER:
        if scene.endswith('Location' + z):
            return np.array([z] * ntrials)
    ...
    n0 = min(change_trial, ntrials)
    return np.array([first] * n0 + [last] * (ntrials - n0))
```

```python
zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)
...
out[4] = zone_code[i]
```

iii. `CONVERSION_NOTES.md` Step 5 says this is a direct port of `behavior.get_reward_zones` and uses the paper’s documented switch-after-30-trials rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward delivery timestamps (`Reward.timestamps`) together with the `reward_zone` flag inside each trial.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
...
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
```

iii. `CONVERSION_NOTES.md` Step 4 says this mirrors `behavior.get_trial_types`; the notes also state the `reward_zone` flag is only active when reward is available, so the combination reduces to rewarded versus omitted trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial the code computes a binary `isreward` once, then broadcasts that trial-level value across all timepoints in the saved output block.

ii.
```python
isreward = np.zeros(ntrials_all, dtype=np.int64)
...
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
out[5] = isreward[i]
```

iii. `CONVERSION_NOTES.md` Step 5 maps reward outcome to the decoder output as a per-trial variable stored in time-varying form, consistent with how the code broadcasts other per-trial variables.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code trims mismatched neural/behavior stream lengths to the shorter stream, skips any out-of-range trial windows, drops lick-error trials, drops trials with bad `scanning`, drops trials with non-finite activity or fewer than 2 samples, and uses a rounded median if the environment signal is unexpectedly not constant within a trial.

ii.
```python
n = min(len(ts), f.shape[1])
n_trimmed = max(len(ts), f.shape[1]) - n
ts = ts[:n]
beh = {k: v[:n] for k, v in beh.items()}
f, f_neu = f[:, :n], f_neu[:, :n]
```

```python
keep = (si >= 1) & (ti <= len(ts))
...
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
...
if act.shape[1] < 2 or not np.all(np.isfinite(act)):
    n_dropped_nan += 1
    continue
```

iii. `CONVERSION_NOTES.md` Step 6 lists these edge cases explicitly, including the 10 sessions trimmed by one frame and the defensive fallbacks for out-of-range or non-finite trial windows.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant cost is dF/F plus OASIS computation; NWB reading is the next-largest cost, and pickling the final dataset is a smaller but still visible final step.

ii.
```python
t0 = time.time()
raw = read_session(path)
t_read = time.time() - t0
...
t1 = time.time()
dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
t_dff = time.time() - t1
...
print(f"... read {i['t_read']:.1f}s dff {i['t_dff']:.1f}s total {i['t_total']:.1f}s ...")
```

iii. `CONVERSION_NOTES.md` Step 6 says dF/F is the dominant cost (~10-16 s/session for ~1000 cells), reading `Fluorescence`/`Neuropil` is ~1-3.5 s/session, and Step 9 reports ~14 s to write the 9.63 GB pickle.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes the cellwise speed-correlation filter, but still contains per-trial loops in `dff_and_events` and in trial assembly. Those remaining loops are the obvious vectorization targets, although they follow the reference pipeline closely.

ii.
```python
for sl in slices:
    f_[:, sl] = f[:, sl]
    f_neu_[:, sl] = f_neu[:, sl]
...
for sl in slices:
    ...
    spks[:, sl] = dcnv.oasis(...)
```

```python
sp_v = speed[valid].astype(np.float32)
d_v = dff[:, valid]
...
r_speed = (d_c @ sp_c) / denom
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says a speedup was added by vectorizing the interneuron correlation across all cells, while dF/F’s per-trial filters and OASIS remain the dominant serial work.

## 13-c. What processing does the code repeat multiple times?

i. The code traverses trial slices multiple times: once to copy trial frames into the NaN-masked buffers, again to compute per-trial baselines and OASIS outputs, and again to assemble saved per-trial neural/input/output blocks. It also computes both `dff` and `events` every time even when only one will be stored.

ii.
```python
for sl in slices:
    f_[:, sl] = f[:, sl]
    f_neu_[:, sl] = f_neu[:, sl]
...
for sl in slices:
    ...
    spks[:, sl] = dcnv.oasis(...)
```

```python
dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
...
act = (events if signal == 'events' else dff)[:, sl]
```

iii. `CONVERSION_NOTES.md` Step 6 highlights dF/F plus OASIS as the main cost, and Step 8 explains that both signals are computed so the script can optionally store either one and compare decoder performance.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is computing both `dff` and `events` even though the final dataset stores only one signal. The per-session `plane_of_cell` array is also returned from `convert_session()` but never written into the final pickle, and the `info` diagnostics are used only for metadata/printing.

ii.
```python
dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
...
act = (events if signal == 'events' else dff)[:, sl]
```

```python
return dict(neural=neural_trials, input=input_trials, output=output_trials,
            brain_region_idx=brain_region_idx, info=info,
            plane_of_cell=plane_of_cell)
```

iii. The agent’s own justification is implicit rather than explicit: `CONVERSION_NOTES.md` Step 8 says it wanted the ability to compare `dff` and `events`, and the extra per-session diagnostics support the summary tables and validation output.
