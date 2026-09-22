# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively gathers every NWB file matching `/app/data/sub-*/*.nwb`, sorts the filenames, and processes every file. It uses `h5py` directly rather than `pynwb`, and parallelizes session processing with a 12-worker spawn-based process pool.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
    for fn, res in zip(files, ex.map(process_session, files)):
        results.append(res)
```

iii. The trajectory says the glob found all 152 sessions from 11 mice. The agent chose direct HDF5 access for speed and later used `spawn` because `fork` conflicted with OpenMP.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file's `general/subject/subject_id`; unique IDs are numerically sorted and each session receives an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r[3]['subject'] for r in results}, key=lambda s: int(s[1:]))
'subject_idx': np.array([subjects.index(r[3]['subject']) for r in results])
```

iii. The agent treated NWB subject metadata as authoritative and confirmed 11 unique mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; `process_session(fn)` returns one nested session entry and reads `general/session_id` for metadata.

ii.
```python
session_id = f['general/session_id'][()].decode()
'neural': [r[0] for r in results]
```

iii. The trajectory states that all 152 NWB files were treated as sessions, matching the file organization.

## 1-d. How are the data split into trials?

i. Trial starts are every positive sample in `trial_start`; ends are every positive sample in `teleport`. Starts and ends are positionally paired, cropped to the shorter list, and trials use half-open slices `[start, teleport)`.

ii.
```python
tstart = np.where(f[B + 'trial_start/data'][:] > 0)[0]
teleport = np.where(f[B + 'teleport/data'][:] > 0)[0]
npairs = min(len(tstart), len(teleport))
tstart, teleport = tstart[:npairs], teleport[:npairs]
for s, e in zip(tstart, teleport):
    ... F[:, s:e] ...
```

iii. The agent described a trial as one on-track lap from `trial_start` to `teleport`, excluding the teleport/ITI.

## 1-e. How are trials filtered based on quality controls?

i. It drops the first trial (previous outcome unknown), trials with lick-sensor errors (>30% of frames having lick value >2), invalid environment labels, trials shorter than two samples, and trials whose neural slice is nonfinite.

ii.
```python
for i in range(1, ntrials):
    if lick_error[i]: continue
    if env_trial[i] not in (0, 1): continue
    if T < 2: continue
    if not np.all(np.isfinite(ev)): continue
```

iii. The trajectory says the lick criterion reproduces the paper's licking-analysis exclusion and that trial one was removed because previous-trial outcome was undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from all planes of the raw `Fluorescence` (F) and `Neuropil` (Fneu) ROI response series plus the `iscell` ROI labels; it does not use the NWB `Deconvolved` field.

ii.
```python
F = np.concatenate([f['processing/ophys/Fluorescence/' + p + '/data'][:].T for p in planes], axis=0)
Fneu = np.concatenate([f['processing/ophys/Neuropil/' + p + '/data'][:].T for p in planes], axis=0)
```

iii. The agent reasoned that the paper computes its analyzed “events” from F and Fneu and that the stored Deconvolved signal is Suite2p's different preprocessing.

## 2-b. How is the `neural` data processed?

i. After pooling planes and selecting cells, each trial independently receives 0.7 neuropil subtraction, trial-mean neuropil restoration, Gaussian smoothing (sigma 15), 300-frame minimum then maximum baseline filters, `(F-F0)/abs(F0)`, Gaussian smoothing (sigma 2), and OASIS deconvolution with tau 0.7 at the per-plane sampling rate. Unlike the reference, it always confines baseline estimation to individual trials.

ii.
```python
fseg = F[:, s:e] - NEU_COEF * Fneu[:, s:e]
fseg += NEU_COEF * np.mean(Fneu[:, s:e], axis=1, keepdims=True)
flow = ndi.maximum_filter1d(ndi.minimum_filter1d(
    ndi.gaussian_filter1d(fseg, BASELINE_SMOOTH, axis=1), BASELINE_WIN, axis=-1),
    BASELINE_WIN, axis=-1)
d = ndi.gaussian_filter1d((fseg - flow) / np.abs(flow), DFF_SMOOTH, axis=1)
events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
```

iii. The agent cited the Methods and repository parameters. Its trajectory claimed on-track-only trial processing matched the paper, but did not account for the reference's session-specific `keep_teleports` baseline windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps Suite2p-curated `iscell` ROIs, then excludes putative interneurons whose session-wide on-track dF/F has Pearson correlation with speed greater than 0.5.

ii.
```python
F = F[iscell]
...
r = (dsub @ spc) / denom
is_int = np.nan_to_num(r, nan=0.0) > INT_R_THRESH
events = events[~is_int]
```

iii. The agent identified both filters in the Methods and reported exclusion rates consistent with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural matrix begins at the `trial_start` sample and ends immediately before `teleport`, so time column zero is the trial-start alignment event.

ii.
```python
s, e = tstart[i], teleport[i]
ev = events[:, s:e]
neural_trials.append(ev.astype(np.float32))
```

iii. The agent stated no resynchronization was needed because imaging and behavioral arrays share sample indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The effective rate is scanner rate divided by plane count (about 15.5078 Hz), and metadata records about 64.48 ms per bin.

ii.
```python
fs = rate / nplanes
'time_bin_size': 1000.0 / (15.5078125)
```

iii. The agent recognized that dual-plane stored rate is the scanner rate and validated the common effective sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is synthesized from the trial slice length and imaging sampling rate, not from raw timestamps (although position timestamps are loaded for reward matching).

ii.
```python
T = e - s
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. The agent assumed uniform sampling at `rate/nplanes` and used that common neural clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Integer offsets 0 through T−1 are divided by the per-plane sampling frequency, producing a zero-based seconds vector.

ii.
```python
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. The implied justification was that this exactly matches uniformly sampled neural columns.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is constructed with exactly one value per neural column after applying the identical `[s:e]` boundaries.

ii.
```python
ev = events[:, s:e]
t_in_trial = np.arange(e - s, dtype=np.float32) / fs
```

iii. The trajectory's sanity check confirmed matching neural, input, and output shapes.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It derives from the behavioral `environment/data` time series.

ii.
```python
env = f[B + 'environment/data'][:]
env_trial[i] = int(np.round(np.median(env[s:e])))
```

iii. The agent interpreted environment as a binary trial context.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. It takes the median within each trial, rounds and converts it to integer, rejects labels outside {0,1}, then repeats the scalar across all trial timepoints.

ii.
```python
env_trial[i] = int(np.round(np.median(env[s:e])))
np.full(T, env_trial[i], dtype=np.float32)
```

iii. The agent treated the signal as constant per trial and added validation against malformed labels.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of paired `trial_start`/`teleport` boundaries, not the raw NWB trial-number series.

ii.
```python
for i in range(1, ntrials):
    ... np.full(T, i, dtype=np.float32) ...
```

iii. The trial index naturally reflects within-session order; the retained sequence begins at 1 because trial zero is dropped.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond broadcasting the loop index across the trial.

ii.
```python
np.full(T, i, dtype=np.float32)
```

iii. The agent considered trial number a per-trial constant.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, position timestamps used as trial time bounds, and the `reward_zone` flag.

ii.
```python
reward_ts = f[B + 'Reward/timestamps'][:]
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. The agent said this mirrors the repository's rewarded-trial definition: delivered reward while the zone flag is active.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. After computing every trial's binary outcome, it broadcasts `rewarded[i-1]` over trial i. Trial zero is removed instead of assigning its unknown predecessor to zero.

ii.
```python
for i in range(1, ntrials):
    ... np.full(T, rewarded[i - 1], dtype=np.float32) ...
```

iii. The agent explicitly justified dropping the first trial because its prior outcome is unknown.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses the behavioral `position` series and reward-zone boundaries inferred from the NWB `identifier` scene name; switch scenes change from the first named zone to the second after trial 30.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
zone_lab = zone_labels_for_session(scene, ntrials)
dist = signed_distance_to_zone(pos[s:e], REWARD_ZONES[zone_lab[i]])
```

iii. The agent cited Methods zone coordinates and stated scene-derived labels were checked against the reward-zone flag with no observed mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the zone, zero inside its inclusive boundaries, and positive after it, measured to the nearest zone edge; it is then categorized.

ii.
```python
d[pos < start] = pos[pos < start] - start
d[pos > stop] = pos[pos > stop] - stop
```

iii. The agent followed the task's reward-relative distance definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit Boolean masks implement the requested cutoffs: `<−50`, `−50..<−10`, `−10..<0`, exactly 0, `0<..≤10`, `10<..≤50`, and `>50`.

ii.
```python
out[(d >= -10) & (d < 0)] = 2
out[d == 0] = 3
out[(d > 0) & (d <= 10)] = 4
```

iii. The agent's comments map each class directly to the instruction labels.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and events are sliced with identical `[s:e]` indices, so each distance class corresponds to the same neural column.

ii.
```python
ev = events[:, s:e]
p = pos[s:e]
dist = signed_distance_to_zone(p, zone)
```

iii. The agent relied on the shared sample axis and validated equal shapes.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position/data`.

ii.
```python
pos = f[B + 'position/data'][:]
p = pos[s:e]
```

iii. The agent treated this as the recorded position in centimeters on the 450-cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is passed directly to a binning helper with no smoothing, interpolation, or clipping.

ii.
```python
def bin_position(pos):
    edges = np.array([90.0, 180.0, 270.0, 360.0])
    return np.digitize(pos, edges).astype(np.int16)
```

iii. The agent used four boundaries to form the five equal 90-cm track bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` at 90, 180, 270, and 360 cm returns classes 0–4.

ii.
```python
np.digitize(pos, np.array([90.0, 180.0, 270.0, 360.0]))
```

iii. The cutoffs were taken directly from the task.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial indices slice position and events.

ii.
```python
ev = events[:, s:e]
p = pos[s:e]
```

iii. Shared indexing supplies one position label per neural timepoint.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It derives from behavioral `lick/data`.

ii.
```python
lick = f[B + 'lick/data'][:]
```

iii. The agent used the recorded time-varying lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero become 1 and all others become 0. Separately, severely corrupted trials are excluded using the >2 / 30% rule.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int16)
lick_error[i] = (np.sum(seg > 2) / len(seg)) > LICK_ERROR_FRAC
```

iii. Binary thresholding follows the requested no/yes output; the trajectory attributes the error exclusion to the paper's licking analyses.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural events use the same `[s:e]` slice.

ii.
```python
ev = events[:, s:e]
lk = (lick[s:e] > 0).astype(np.int16)
```

iii. Equal sample indices and output-shape checks were the alignment basis.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name in `identifier`, the session's trial count/index, and fixed A/B/C coordinate definitions; the raw reward-zone flag is only used as a check and in reward-outcome logic.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
zone_lab = zone_labels_for_session(scene, ntrials)
```

iii. The agent reasoned that scene names encode zone identity and switch scenes change after the Methods-specified 30 trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Constant scenes use their final character; switch scenes parse pre/post characters and switch after trial 30. A/B/C are mapped to 0/1/2 and broadcast over time.

ii.
```python
labels = [z0] * min(SWITCH_TRIAL, ntrials) + [z1] * max(0, ntrials - SWITCH_TRIAL)
np.full(T, ZONE_IDX[zone_lab[i]], dtype=np.int16)
```

iii. The agent reported validating these labels against observed reward-zone entries.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from `Reward/timestamps`, behavioral position timestamps delimiting the trial, and `reward_zone/data`.

ii.
```python
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. The agent invoked the paper repository's rewarded-trial criterion rather than using reward timestamps alone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 only if a reward timestamp falls inclusively between start and end timestamps and the reward-zone flag occurs in the trial; otherwise 0. The scalar is repeated over all timepoints.

ii.
```python
rewarded[i] = int(got_reward and in_zone)
np.full(T, rewarded[i], dtype=np.int16)
```

iii. The agent stated this implements `behavior.get_trial_types` and the binary requested output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Imaging and behavior arrays are cropped to their common minimum length; unmatched start/end lists are cropped to pairs; trials ending beyond available frames are removed; malformed environment, too-short, nonfinite-neural, and lick-error trials are skipped. Correlation NaNs are treated as zero. There is no explicit recovery of missing reward-zone labels because labels come from scene metadata.

ii.
```python
nframes = min(F.shape[1], len(pos))
keep_tr = teleport < nframes
if env_trial[i] not in (0, 1): continue
if not np.all(np.isfinite(ev)): continue
```

iii. The trajectory records discovering one-frame dual-plane mismatches and patching by truncating to samples with both imaging and behavior.

## 13-a. What are the most time-consuming steps of the code?

i. The code's heavy work is loading large F/Fneu arrays, per-trial Gaussian/min/max filtering and OASIS deconvolution, collecting a roughly 9.4-GB in-memory result, and pickling it. Sessions are parallelized to reduce elapsed time.

ii.
```python
with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
    ... ex.map(process_session, files) ...
events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
pickle.dump(data, f, protocol=4)
```

iii. The trajectory observed 0.4–1.4 seconds in sample sessions, then processed 152 sessions and wrote a 9.4-GB pickle; it identified multiprocessing/OpenMP startup as an operational bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-wise dF/F/deconvolution must mostly remain segmented, but trial reward/error/environment summaries and final per-trial construction could be partly vectorized or precomputed. The agent already vectorized the cell-speed correlations instead of looping over cells.

ii.
```python
for s, e in zip(tstart, teleport): ...
for i, (s, e) in enumerate(zip(tstart, teleport)): ...
r = (dsub @ spc) / denom
```

iii. The trajectory did not explicitly discuss vectorization; its main efficiency decision was session-level multiprocessing.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials once for neural preprocessing, again for reward/lick/environment summaries, and a third time to assemble outputs. It also slices the same behavioral arrays repeatedly and evaluates reward timestamps against each trial separately.

ii.
```python
for s, e in zip(tstart, teleport): ...
for i, (s, e) in enumerate(zip(tstart, teleport)): ...
for i in range(1, ntrials): ...
```

iii. The agent did not justify this repetition directly; the organization separates preprocessing, trial summaries, and final serialization.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `trials_kept` is populated but never returned or saved; reward-zone validation data are loaded even though zone labels come from scene metadata (the flag remains needed for reward outcome); full dF/F is retained only until interneuron filtering; several metadata values are diagnostic only. `TRACK_LENGTH` is unused.

ii.
```python
trials_kept = []
trials_kept.append(i)
TRACK_LENGTH = 450.0
```

iii. The trajectory emphasized diagnostics and sanity checks, but did not identify these as discarded work.
