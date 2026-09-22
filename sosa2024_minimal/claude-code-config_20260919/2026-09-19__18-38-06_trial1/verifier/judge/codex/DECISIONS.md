# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively gathers every NWB file under `/app/data/sub-*`, sorts them by numeric mouse ID and filename, and loads each complete session directly with `h5py`. Sessions are converted in parallel, with an optional `--nsessions` limit only for testing.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')),
               key=lambda p: (int(os.path.basename(p).split('_')[0][5:]),
                              os.path.basename(p)))
with h5py.File(path, 'r') as f:
    ...
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
        results.append(res)
```

iii. The trajectory says the directory inspection found 11 mice and 152 NWB sessions. Direct HDF5 access and session-level multiprocessing were chosen to make full conversion practical; spawn was selected after forked OASIS/Numba workers failed.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, unique IDs are numerically sorted, and every session receives the corresponding `subject_idx`.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
'subject_idx': np.array([subjects.index(r['subject']) for r in results])
```

iii. The agent treated the NWB subject field as authoritative and checked it against the `sub-*` organization.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session; `load_session` returns one session record and the output lists preserve sorted file order.

ii.
```python
def load_session(path):
    """Read one NWB session and return the per-trial neural/input/output arrays."""
...
'neural': [r['neural'] for r in results]
```

iii. The trajectory identified one `ses-*` NWB file as one experimental session.

## 1-d. How are the data split into trials?

i. Trial starts are every nonzero `trial_start` sample and trial ends are every nonzero `teleport` sample. The stored lap window is `[trial_start-1, teleport-1)`, excluding the teleport period/sample.

ii.
```python
trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
teleports = np.nonzero(beh['teleport/data'][:])[0]
starts = trial_starts - 1
stops = teleports - 1
...
neural.append(events[:, s:t])
```

iii. The agent justified this as the exact indexing used by the paper’s `preprocessing.dff`, with teleport excluded because track variables are undefined there and imaging was usually blanked.

## 1-e. How are trials filtered based on quality controls?

i. The first trial of each session is removed because previous outcome is unavailable. Trials with a stuck lick sensor are also removed when more than 30% of lap samples have cumulative lick count greater than 2. No short-trial or speed filter is applied.

ii.
```python
lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC
...
if i == 0:
    continue
if lick_error[i]:
    continue
```

iii. The trajectory ties the 30% rule to the paper and reports 81 flagged trials, matching the paper. It drops rather than NaNs them because lick is a required decoder target, and drops trial zero rather than inventing a previous outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from raw `Fluorescence` (F), `Neuropil` (Fneu), ROI `iscell`, and plane membership—not the NWB `Deconvolved` series. Speed is additionally used for interneuron filtering.

ii.
```python
F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
Fneu_list.append(f['processing/ophys/Neuropil'][plane]['data'][:, :].T[keep])
```

iii. The agent found that the paper recomputed dF/F and events, whereas the stored Deconvolved field was Suite2p output and nonzero during ITIs.

## 2-b. How is the `neural` data processed?

i. For each trial, the code subtracts `0.7*Fneu`, adds back the trial mean neuropil, computes a sigma-15/300-sample maximin baseline, calculates `(F-baseline)/abs(baseline)`, smooths at sigma 2, and OASIS-deconvolves with tau 0.7 at the per-plane frame rate. Planes are pooled.

ii.
```python
f = F[:, start:stop] - NEU_COEF * Fneu[:, start:stop]
f += NEU_COEF * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
flow = gaussian_filter1d(f, BASELINE_SIG, axis=-1)
flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
d = gaussian_filter1d((f - flow) / np.abs(flow), DFF_SIG, axis=-1)
events[:, start:stop] = dcnv.oasis(d, 2000, TAU, frame_rate)
```

iii. The trajectory attributes every parameter to the Methods or repository. It intentionally reproduces the analyzed “events” signal rather than accepting the NWB’s precomputed signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p-curated `iscell` ROIs are retained, then putative interneurons with dF/F–speed Pearson correlation above 0.5 are removed.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
events = events[~is_int]
```

iii. Both filters were selected from the paper’s calcium-processing methods; the trajectory also checked resulting cell counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each matrix begins at the chosen lap-start sample, so column zero is trial start; no interpolation is performed.

ii.
```python
s, t = starts[i], stops[i]
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The agent regarded splitting at trial start as sufficient alignment and recorded `off_start = 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained without rebinning, about 64.484 ms (15.508 Hz) per sample. Two-plane scanner rate is divided by plane count.

ii.
```python
frame_rate = float(imaging_rate) / len(planes)
dt = float(np.median(np.diff(tstamps)))
'time_bin_size': float(round(list(dts)[0] * 1000, 4))
```

iii. The agent verified a common behavioral `dt` across sessions and used native aligned samples to avoid unnecessary resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from position-series timestamps, summarized as their median sample interval.

ii.
```python
tstamps = beh['position/timestamps'][:]
dt = float(np.median(np.diff(tstamps)))
```

iii. The trajectory found behavioral streams already frame-aligned and regularly sampled.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based time vector is generated as sample index times `dt`.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. This makes the aligning frame exactly zero and preserves native temporal spacing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `T=t-s` values and is constructed alongside the same neural slice.

ii.
```python
T = t - s
inp = np.empty((4, T), dtype=np.float32)
neural.append(events[:, s:t])
```

iii. The agent relied on the NWB’s existing frame alignment and checked/cropped a possible single trailing imaging frame.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes directly from the behavioral `environment` series.

ii.
```python
env = beh['environment/data'][:]
env_vals = np.unique(env[s:t])
```

iii. The agent verified one binary environment value per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Uniqueness within each trial is asserted, the value is cast to integer, and broadcast over time.

ii.
```python
assert len(env_vals) == 1
envs[i] = int(env_vals[0])
inp[1] = envs[i]
```

iii. No transformation was considered necessary because raw values already encode ENV1/ENV2 as 0/1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index over trial boundaries derived from `trial_start`/`teleport`.

ii.
```python
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The agent interpreted trial number as within-session sequential trial identity.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The index is simply broadcast to all timepoints; retained trials keep original numbering despite exclusions.

ii.
```python
inp[2] = i
trial_ids.append(i)
```

iii. This preserves protocol trial number, including the trial-30 switch point.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the preceding trial’s computed reward outcome, itself based on Reward timestamps and reward-zone occupancy.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
```

iii. The agent used actual reward delivery and required in-zone context to represent rewarded versus omitted trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i`, `rewarded[i-1]` is broadcast across time. Trial zero is excluded because no previous trial exists.

ii.
```python
if i == 0:
    continue
inp[3] = rewarded[i - 1]
```

iii. The agent preferred removing an undefined datum over assigning an arbitrary zero.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral position plus per-trial zone identity inferred from `reward_zone`, position while the flag is active, and the known trial-30 switch protocol.

ii.
```python
in_zone = rzone[s:t] > 0
observed_zone[i] = zone_label(pos[s:t][in_zone].min())
...
z0, z1 = REWARD_ZONES[zone_of_trial[i]]
```

iii. The agent used known A/B/C boundaries and verified inferred labels against every trial with an observed zone flag.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the zone, zero inside, and positive after it, measured to the nearest boundary.

ii.
```python
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
```

iii. This directly implements “distance to any location in the reward zone.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven specified intervals, preserving exact zero as its own category.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[d == 0] = 3
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The explicit masks were chosen to match boundary wording exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and zone-derived distance use the identical `[s:t]` samples as neural activity.

ii.
```python
p = pos[s:t]
neural.append(events[:, s:t])
```

iii. No resampling is needed because behavior is already aligned to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavioral `position/data`.

ii.
```python
pos = beh['position/data'][:]
p = pos[s:t]
```

iii. The raw variable is already corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is discretized; no smoothing or normalization is applied.

ii.
```python
out[1] = np.digitize(p, POS_EDGES)
```

iii. The requested decoder output is categorical, so only instructed binning is necessary.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` uses edges 90, 180, 270, and 360 cm to form five classes.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.digitize(p, POS_EDGES)
```

iii. These are five equal 90-cm bins over the 450-cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same `[s:t]` indices as neural data.

ii.
```python
p = pos[s:t]
neural.append(events[:, s:t])
```

iii. The NWB behavior samples are already imaging-frame aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavioral `lick/data`.

ii.
```python
lick = beh['lick/data'][:]
```

iii. The agent identified this as a cumulative/count-like lick signal requiring binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Each sample is 1 when raw lick is positive and 0 otherwise; separately, corrupted trials are excluded by the stuck-sensor rule.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
out[3] = lk
```

iii. Thresholding satisfies the requested binary output; trial exclusion prevents sensor failures becoming labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks use the same `[s:t]` slice as neural samples.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
neural.append(events[:, s:t])
```

iii. The agent relied on native frame alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone/data`, position during active-zone samples, and the known switch schedule.

ii.
```python
in_zone = rzone[s:t] > 0
observed_zone[i] = zone_label(pos[s:t][in_zone].min())
```

iii. The trajectory reports that direct observations bracketed the protocol switch and validated all filled labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Observed zone starts are mapped to A/B/C (0/1/2). Constant sessions use one label; switch sessions fill labels before/after trial 30, with a data-derived fallback, and broadcast the label over time.

ii.
```python
if np.all(labels == labels[0]):
    zone_of_trial[:] = int(labels[0])
else:
    switch_trial = (SWITCH_TRIAL if last_before < SWITCH_TRIAL <= first_after
                    else first_after)
...
out[4] = zone_of_trial[i]
```

iii. This uses protocol knowledge while asserting consistency with every observed trial.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses Reward event timestamps, behavior timestamps, and the trial’s reward-zone flag.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
```

iii. The agent followed the paper’s rewarded-trial concept: reward delivered while the trial contains the reward zone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if a reward timestamp lies in its timestamp interval and the zone flag occurs, otherwise 0; the result is broadcast across trial time.

ii.
```python
rewarded[i] = int(in_zone.any() and got_reward)
...
out[5] = rewarded[i]
```

iii. This represents the requested rewarded/omitted categorical outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code permits and crops one extra trailing imaging frame, asserts all other structural assumptions, fills missing per-trial zone observations using constant/switch-session logic, treats undefined correlations as non-interneurons, and excludes corrupted lick trials and undefined-first-history trials.

ii.
```python
assert 0 <= F.shape[1] - nframes <= 1
F = F[:, :nframes]
...
speed_corr = np.nan_to_num(speed_corr, nan=0.0)
assert np.all(zone_of_trial[known] == observed_zone[known])
```

iii. The trajectory investigated the one-frame alignment issue and missing zone flags, then added constrained corrections plus assertions rather than silently accepting broader inconsistencies.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large fluorescence arrays, per-trial filtering/baseline calculation, OASIS deconvolution, and serializing the large pickle dominate runtime.

ii.
```python
dff, events = compute_dff_and_events(...)
events[:, start:stop] = dcnv.oasis(...)
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory’s timing experiments showed conversion was expensive enough to require session-level parallelism; excessive workers also caused memory/process failures.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial reward/environment/quality computation and output construction could partly be computed session-wide, though variable trial lengths still require splitting. The agent already vectorized neuron–speed correlations across cells.

ii.
```python
for i, (s, t) in enumerate(zip(starts, stops)):
    ...
for i in range(ntrials):
    ...
speed_corr = (dff_c @ spd_c) / denom
```

iii. The chosen loops keep variable-length lap logic clear; the trajectory prioritized parallel sessions and vectorized the expensive correlation calculation.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials once for dF/F, again for trial metadata, and again to build arrays. Position, lick, zone, and reward values are therefore sliced repeatedly, and maximin/OASIS is invoked separately for every trial.

ii.
```python
for start, stop in zip(starts, stops):  # neural processing
...
for i, (s, t) in enumerate(zip(starts, stops)):  # metadata
...
for i in range(ntrials):  # output assembly
```

iii. The agent kept stages separate to control memory and mirror the paper’s per-trial processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes full-session dF/F only to use it for interneuron filtering and then deletes it; it also computes metadata such as dates, scenes, reward fractions, and trial IDs that the decoder itself does not consume. Neural events for subsequently excluded trials are also computed.

ii.
```python
dff, events = compute_dff_and_events(...)
...
del dff, dff_v, dff_c
...
'date': date, 'scene': scene, 'rewarded_fraction': float(np.mean(rewarded))
```

iii. dF/F is required for the paper’s interneuron criterion, while extra metadata supports auditing. Computing before trial exclusions simplifies correct session-wide filtering but spends work on dropped trials.
