# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers `data_structure_*.mat` and matching `motionEnergy_*.mat` only under `data/Ephys_Behavior`, sorts them, then keeps sessions containing both autowater states. It supports HDF5 and classic MAT loading, but the final run skips unsupported sessions and does not load `RandomizedDelay_Ephys_Behavior`.

ii.
```python
base = Path('data/Ephys_Behavior')
data_files = sorted(base.glob('data_structure_*.mat'))
...
sessions = select_context_sessions(sessions)
```

iii. Notes identify the two-context subset as the intended cohort, but acknowledge that selecting merely by both autowater states yields 21 sessions rather than the paper's curated 12 and that reference-matching curation remains unresolved.

## 1-b. How are the data split into subjects?

i. Subject and date are parsed from each filename; unique subject strings are sorted and each retained session receives an index.

ii.
```python
subj, day = m.group(1), m.group(2)
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
```

iii. The notes planned to obtain subject metadata from filenames or `obj.meta`; filenames were used consistently.

## 1-c. How are the data split into sessions?

i. Each discovered data-structure file is one session. Sessions are prefiltered for both context values, then appended only if building succeeds and at least two valid trials remain.

ii.
```python
sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
...
if len(neural) < 2: continue
data['neural'].append(neural)
```

iii. The agent intended to isolate the paper's two-context sessions, but explicitly recorded that its 21-session result does not match the paper/reference cohort.

## 1-d. How are the data split into trials?

i. The number of trials is the length of `bp.ev.goCue`; boolean trial fields are flattened to that indexing scheme, and accepted zero-based indices select spikes, video, motion energy, labels, and inputs.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
trial_idx = np.where(valid)[0]
for tr in trial_idx:
```

iii. The rationale was that Bpod fields and go-cue entries define trial identity directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only when they are not early, are hit or miss (thus dropping ignores), are not stimulated, have exactly one of R/L true, and have a binary autowater value.

ii.
```python
valid &= ~early
valid &= (hit | miss)
valid &= ~stim_enable
valid &= (right ^ left)
valid &= np.isin(autowater.astype(int), [0, 1])
```

iii. Notes cite paper exclusion of early-lick and ignore trials and preserve misses so outcome remains nondegenerate. This differs from the human conversion, which keeps ignores as a requested third outcome class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from the first resolved `obj.clu` probe's per-cluster `trial` and `trialtm` arrays. Although `goCue` is loaded, it is not used in neural extraction.

ii.
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]
...
tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
```

iii. The plan stated that spikes would be aligned to `goCue` following `alignSpikes.m`, but the implementation never completed that subtraction.

## 2-b. How is the `neural` data processed?

i. Per trial and cluster, raw spike times are histogrammed into 75 ms bins and stored as counts. There is no conversion to Hz, smoothing, normalization, or baseline correction.

ii.
```python
spikes = tm_arr[tr_arr == tr1]
mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. Notes proposed low-rate filtering and reference alignment, and chose 75 ms based on decoder scripts, but do not justify omitting firing-rate conversion and Gaussian smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by cluster quality, firing rate, or minimum units. Every cluster in the first supported probe is retained; unsupported cluster layouts cause the whole session to be skipped.

ii.
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    ...
```

iii. Notes correctly identify the paper's greater-than-1-Hz criterion and at-least-10-unit session rule as requirements, but they were not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not actually aligned. `trialtm` is histogrammed directly rather than subtracting `goCue[trial]`, despite constructing a nominal −1.5 to +1.5 s axis.

ii.
```python
spikes = tm_arr[tr_arr == tr1]
mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The documentation claims go-cue alignment matching `alignSpikes`, so the code contradicts the intended decision.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The stored grid has 41 centers from −1.5 to +1.5 s at 75 ms spacing. Spikes are directly counted in those bins; video and motion traces are interpolated to the centers.

ii.
```python
BIN_SIZE_S = 0.075
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The agent chose 75 ms because the paper's context decoder scripts use `rez.binSize = 75`; the human conversion instead preserves the common reference processing at 5 ms over −2.5 to +2.5 s.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic grid from the constants `T_START`, `T_END`, and `BIN_SIZE_S`, conceptually relative to raw `bp.ev.goCue` but not numerically derived from each cue.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
inp = time_bins[None, :].astype(np.float32)
```

iii. Notes say the decoder input requires a time-from-go-cue vector and should share the alignment grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The fixed grid is generated once per session build and copied into each trial with a leading input dimension.

ii.
```python
time_bins = np.arange(-1.5, 1.5 + 1e-9, 0.075, dtype=np.float32)
input_trials.append(time_bins[None, :].astype(np.float32))
```

iii. The agent considered this the decoder's continuous aligned time coordinate; no raw-data processing was considered necessary.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Input centers and neural histogram edges are constructed from the same `time_bins`, but because spike times are not go-cue shifted, the semantic alignment is wrong.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2,
                           [time_bins[-1] + BIN_SIZE_S/2]])
```

iii. The intended justification is a shared go-cue grid; only the shared array geometry was implemented.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses the per-trial `bp.R` flag (and uses `bp.L` only for validity checking), not hit/miss jointly with instructed side.

ii.
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
lick_dir = 1 if bool(right[tr]) else 0
```

iii. Notes explicitly planned to use R/L trial labels. They did not account for a miss meaning the actual lick is opposite the instructed direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. R becomes right/1 and otherwise left/0, broadcast over time. Ignore/no-lick trials are removed rather than represented as class 2.

ii.
```python
np.full(time_bins.shape, lick_dir, dtype=np.int64)
```

iii. The notes treat direction labels as the choice itself and prioritize a binary decoder output, contrary to the requested/reference three-class derivation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived directly from `bp.autowater`.

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
```

iii. The agent found that reference context decoders define WC versus DR using autowater and used that mapping.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater true is WC/0 and false is DR/1; the per-trial value is broadcast over all 41 bins.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
np.full(time_bins.shape, context, dtype=np.int64)
```

iii. Notes explicitly selected the task-required convention `WC=0`, `DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit` and `bp.miss`, although after filtering every retained trial is one or the other.

ii.
```python
hit = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
valid &= (hit | miss)
```

iii. Notes identify hit/miss/ignore as the outcome sources but deliberately remove ignore trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit becomes correct/1 and every retained non-hit becomes incorrect/0, broadcast in time. No ignore/2 class exists in data or `output_values`.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
```

iii. The agent kept misses to avoid a degenerate outcome decoder, but followed paper analysis exclusions rather than the explicit target's three outcome categories.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. For HDF5 sessions it searches both trajectory views for several tongue feature names and reads `ts` x/y coordinates plus `frameTimes`; it also reads `goCue`.

ii.
```python
extract_hdf5_traj_velocity(obj, tr,
 ['top_tongue', ..., 'tongue', 'left_tongue', 'right_tongue'], time_bins)
```

iii. Notes describe trajectory/DLC-derived tongue speed and iterative debugging of feature-name and sparsity issues.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The first view containing a candidate is used. Framewise speed is the Euclidean first difference divided by frame-time difference, then linearly interpolated to bin centers; internal missing values are nearest-linearly filled. There is no likelihood cutoff, position smoothing, valid-run handling, view normalization, or two-view averaging.

ii.
```python
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
speed = np.sqrt(dx*dx + dy*dy) / dt
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The notes say nearest-fill was added to make sample output balanced; this optimizes class statistics but departs from reference visibility-aware processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session median over all finite retained-trial bins defines low/0 versus high/1. NaNs remain initialized as low/0, so no not-visible/2 category is produced.

ii.
```python
tongue_thr = np.nanpercentile(tongue_all, 50)
tmp = np.zeros(tong.shape, dtype=np.int64)
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
```

iii. The median follows the prompt; notes acknowledge sparse visibility but resolve it through filling rather than the specified third class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Raw `frameTimes - goCue` are interpolated to the nominal grid. The session video-to-behavior bitcode offset is not calculated or subtracted, while neural spikes are not go-cue shifted either.

ii.
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The plan cited `loadKinData` alignment but did not document or implement the reference clock correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses the same HDF5 trajectory fields, searching both views and several paw feature names.

ii.
```python
extract_hdf5_traj_velocity(obj, tr,
 ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
```

iii. Notes planned DLC-derived paw speed and later identified a usable feature after an initially degenerate result.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It uses the same raw finite-difference, interpolation, and missing-value filling as tongue velocity, selecting the first matching view/feature rather than the reference's reliable bottom-camera `top_paw` specifically.

ii.
```python
speed = np.sqrt(dx*dx + dy*dy) / dt
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The agent justified the generic search pragmatically to obtain a nonconstant output, not as a match to reference feature selection and quality processing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is split at the per-session finite-value median into 0/1; missing bins default to low/0 and class 2 is absent.

ii.
```python
paw_thr = np.nanpercentile(paw_all, 50)
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
```

iii. The median is instruction-driven, but the notes do not justify violating the required not-visible category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It uses `frameTimes - goCue` and interpolation to the same nominal centers, without video clock correction.

ii.
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The intended rationale was common go-cue alignment, but the necessary reference offset is omitted.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from the session's `motionEnergy_*.mat` top-level `me`, recursively unwrapped into a per-trial trace; missing files yield no trace.

ii.
```python
x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
me = x['me'].flat[0]
data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. Notes correctly recognized the standalone `me.data` sidecar and added nested-struct handling after a full-run failure.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each raw trace is assigned an artificial evenly spaced −1.5-to-1.5 axis and interpolated onto 41 centers. There is no use of camera frame timestamps or averaging within bins.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. Notes planned reference interpolation/alignment but the implementation uses a length-based surrogate time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session median is computed after replacing NaNs with the pooled median; finite bins become low/0 or high/1 and missing bins remain low/0. No-video/2 is absent.

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) ...])
thr = np.nanpercentile(all_me, 50)
tmp[finite] = (meb[finite] >= thr).astype(np.int64)
```

iii. The median follows the requested threshold, but documentation does not justify erasing the required no-video state.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is only by rescaling every trace across the analysis window, not by side-camera `frameTimes`, video clock offset, and per-trial go cue. Consequently it is not physically aligned to neural data.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
np.interp(time_bins, x_old, raw)
```

iii. The agent intended common alignment but provides no rationale for assuming each trace spans exactly the target window.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many extraction exceptions return all-NaN traces; trajectory gaps are interpolated across; missing motion trials get a one-element NaN trace; NaN output bins ultimately default to low/0. Unsupported neural layouts and other build failures skip whole sessions.

ii.
```python
except Exception:
    return np.full(time_bins.shape, np.nan, dtype=float)
...
vals = np.interp(idx, idx[good], vals[good])
...
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}')
```

iii. Notes present these as engineering fallbacks and describe nearest filling as fixing imbalance. Unlike the reference, missing visibility/video is not preserved as category 2.

## 11-a. What are the most time-consuming steps of the code?

i. Trajectory extraction dominates: every trial searches HDF5 references and computes velocities repeatedly. Full logs show many session builds taking roughly 6–19 seconds while file opening takes about 0.01 seconds.

ii.
```python
for tr in trial_idx:
    tongue_vals = extract_hdf5_traj_velocity(...)
    paw_vals = extract_hdf5_traj_velocity(...)
```

iii. Notes include runtime estimates but never complete the efficiency section; observed timings contradict the reference implementation's loading-dominated profile.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Cluster-by-trial spike histograms could be computed per cluster over all trials, and threshold application could operate on stacked arrays. Ragged HDF5 trajectory reading still reasonably requires trial iteration.

ii.
```python
for tr in trial_idx:
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The notes leave “Code inefficiencies identified” blank, so no explicit justification was supplied.

## 11-c. What processing does the code repeat multiple times?

i. Tongue and paw velocities are computed once in the first trial loop but discarded, recomputed for concatenated threshold arrays, then recomputed a third time to assign classes. Each call also rescans views, feature names, and HDF5 references.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(...)  # discarded
...
tongue_all = np.concatenate([extract_hdf5_traj_velocity(...) ...])
...
tong = extract_hdf5_traj_velocity(...)
```

iii. Notes do not acknowledge this repetition; they only chronicle debugging and output balance.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first-loop `tongue_vals` and `paw_vals` are never used. Session discovery also opens every candidate file to inspect autowater and then reopens selected files during conversion; HDF5 files are generally left open.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(...)
paw_vals = extract_hdf5_traj_velocity(...)
output_trials.append(...)  # neither value is inserted here
```

iii. The agent supplied no rationale; its efficiency-note placeholders remained unfinished.
