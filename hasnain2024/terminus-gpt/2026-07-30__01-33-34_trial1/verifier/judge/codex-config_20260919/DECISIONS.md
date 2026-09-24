# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 19 randomized-delay session IDs and loads each paired `data_structure` and `motionEnergy` file. It supports HDF5/v7.3 session files with `h5py`, falls back to `scipy.io.loadmat` for older files, and always uses SciPy for motion energy. It omits the 25 fixed-delay ephys sessions used by the reference.

ii.
```python
RAND_SESSIONS = ['JEB11_2022-05-10', ..., 'JEB24_2023-11-03']
base = Path('data/RandomizedDelay_Ephys_Behavior')
for key in keys:
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The notes say the list follows the paper's reference loading scripts and focus on the randomized-delay ALM sessions. They also justify dual readers because mixed MATLAB formats were observed. No justification is given for excluding the fixed-delay reference sessions.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from the session ID prefix. Subjects are added in first-seen order and each session receives the corresponding index; this yields JEB11, JEB12, JEB23, and JEB24.

ii.
```python
subj = key.split('_')[0]
if subj not in subject_names:
    subject_names.append(subj)
subject_idx.append(subject_names.index(subj))
```

iii. The README reports these four subjects. The trajectory/notes recognize that session loader names encode animal and date.

## 1-c. How are the data split into sessions?

i. Every hard-coded `<animal>_<date>` key is one session and one element of each top-level session list. Only the randomized-delay directory is used.

ii.
```python
df = base / f'data_structure_{key}.mat'
mf = base / f'motionEnergy_{key}.mat'
all_neural.append(neural)
all_input.append(inp)
all_output.append(out)
```

iii. The AI states that per-session MAT files and reference loader scripts define session boundaries.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` sets the number of trials. Behavioral arrays, trajectory references, and motion-energy elements are indexed by zero-based trial position; spike `trial` values are matched to one-based trial numbers. Each retained trial becomes one matrix/list element.

ii.
```python
n_trials = bp['Ntrials']
for ti in range(n_trials):
    ts, ft = getter(ti)
for trial_idx in range(1, n_trials + 1):
    mask = tr == trial_idx
```

iii. The notes identify event arrays and motion-energy arrays as per-trial and report 365 trials in a representative session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only when `early == 0`, `no == 0`, and the neural matrix is not entirely zero. Photostimulation trials are not filtered. Movement thresholds are computed before trial filtering.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The README explicitly says early, no-response, and all-zero-neural trials are excluded. It gives no paper-based justification for dropping ignores or retaining photostimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the first `obj.clu` probe's `trialtm` and `trial` fields. Cluster quality, probe selection, and the go-cue values are not used in neural construction.

ii.
```python
clu = deref(f, f['obj/clu'][()].flat[0])
trialtm = [... for r in clu['trialtm'][()].flat]
trial = [... for r in clu['trial'][()].flat]
```

iii. The notes identify `obj.clu` and `alignSpikes.m` as relevant, but do not justify selecting only the first probe or omitting quality fields.

## 2-b. How is the `neural` data processed?

i. Raw spike times are histogrammed into 75 ms bins per unit and trial. Values remain spike counts; there is no conversion to Hz, Gaussian smoothing, normalization, or baseline correction.

ii.
```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)
per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The notes cite a 75 ms bin in the paper's decoding scripts. They do not explain the omission of the reference 5 ms rate representation and 14 ms smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No unit-level filtering is performed. All units from the selected probe are kept regardless of manual quality or mean firing rate; only whole trials with all-zero neural matrices are removed.

ii.
```python
n_units = len(trialtm)
brain_region_idx = np.zeros((n_units,), dtype=int)
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The documentation does not mention unit quality filtering or the paper's greater-than-1-Hz criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Although metadata claims go-cue alignment, neural spike times are binned directly against edges from -2.5 to 2.5 seconds. The code never subtracts that trial's `bp.goCue`, so it does not implement go-cue alignment unless `trialtm` were already aligned (the reference establishes it is relative to trial start).

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
```

iii. The notes correctly identify `alignSpikes.m` and go-cue event alignment, but the claimed alignment is not carried into the neural code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The intended resolution is 75 ms and all streams are rebinned by histograms or bin means. Because `np.arange(-2.5, 2.5 + .075, .075)` does not divide the five-second interval evenly, it creates 67 bins and extends the final edge to 2.525 s.

ii.
```python
BIN_SIZE = 0.075
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
```

iii. The AI cites `rez.binSize = 75` ms in decoding code. The reference conversion instead preserves the paper preprocessing grid at 5 ms; the AI does not discuss the overshooting final bin.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured -2.5 to 2.5 s window and 75 ms bin edges, not directly from a raw variable. `bp.goCue` is conceptually the origin but is not used to form this common vector.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. The notes say go cue is the required/common alignment event and that the time input uses that aligned grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent bin edges are averaged to obtain centers, copied into a `(1, 67)` float32 array for every trial.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. The AI treats the bin centers as a continuous time-varying decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has the same number of bins and nominal edges as neural data, but it labels those bins as time from go cue even though neural spikes were not shifted by go cue. Thus shape alignment exists but physical temporal alignment does not.

ii.
```python
neural = bin_unit_spikes_for_trials(..., time_edges)
input_trials = make_time_input(n_trials, time_centers)
```

iii. Documentation asserts all data are aligned to go cue, without addressing the missing neural subtraction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived only from the instructed-side flags `bp.R` and `bp.L`; outcome flags are not used.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The notes do not give a specific rationale. The README exposes only left/right classes because no-response trials are removed.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Trials instructed right are code 1 and all others code 0. Incorrect trials are therefore labeled by instructed direction rather than actual lick direction, and no-lick code 2 is absent because ignores are discarded.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. No justification is recorded for equating instructed side with actual lick side on misses.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp.autowater`.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. The notes connect water-cued/automatic-water trials with behavioral condition groupings.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Positive autowater is relabeled WC (0); otherwise it is DR (1), then repeated across all time bins.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. This direct categorical mapping matches the output labels documented by the AI.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses only `bp.hit`; misses and ignores both initially map to incorrect, though ignores are later filtered out via `bp.no`.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
valid = (bp['early'] == 0) & (bp['no'] == 0)
```

iii. The README advertises only incorrect/correct and explicitly says no-response trials are excluded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits are correct (1), all other raw trials incorrect (0), and the value is repeated through time. The requested/reference ignore class 2 is not represented.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. No justification is supplied for omitting the explicitly requested ignore category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses DLC `obj.traj` feature positions and `frameTimes`. It searches bottom-view tongue feature names first, then side-view names, and chooses the first view with any finite binned value. `bp.goCue` is subtracted from frame times.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', ...])
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes identify the paper's two trajectory feature groups. They do not justify selecting one camera rather than combining both.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For every tracked feature, adjacent x/y differences are divided by frame-time differences to produce speed at midpoint times. Speeds are averaged into 75 ms bins. There is no likelihood threshold, positional smoothing, valid-run handling, or cross-view normalization/averaging.

ii.
```python
dxy = np.diff(xy, axis=2)
spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
tmid = (frame_times[:-1] + frame_times[1:]) / 2
```

iii. The documentation only says velocity is median-thresholded; it does not justify departures from the reference kinematic preprocessing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median over all finite trial/bin values is used: values below it are 0 and values at/above it are 1. Missing values are initialized and left as 0 rather than assigned requested code 2.

ii.
```python
thr = np.nanmedian(allv) if allv.size else np.nan
b = np.zeros_like(a, dtype=np.int64)
b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The AI correctly cites per-session median thresholding, but its README defines only low/high and omits “not visible.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame midpoints have per-trial go cue subtracted and are averaged on the nominal common edges. The required session video-to-behavior clock offset is never computed, while neural data also lacks its go-cue subtraction; consequently the streams are not correctly co-aligned.

ii.
```python
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes claim all streams are go-cue aligned but do not document or implement `findVideoOffset` logic.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity uses bottom-camera trajectory positions and frame times, preferring `top_paw` and falling back to `bottom_paw`.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
ts, ft = get_stream1(ti)
```

iii. The notes identify paw DLC features, but do not discuss why a fallback is valid; the reference deliberately uses reliable `top_paw` only.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw processing uses the same raw adjacent-frame Euclidean speed and 75 ms bin means as tongue, without confidence filtering or Gaussian smoothing.

ii.
```python
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The AI gives no detailed processing rationale beyond identifying DLC features and median discretization.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses a per-session median of all finite binned values. Low is 0, high is 1, and missing is incorrectly also 0 rather than category 2.

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. The notes cite the required median threshold but document only low/high categories.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw speed midpoint times have the go cue subtracted and are binned on the common edge array, but no video-clock offset is applied and the neural stream itself is not go-cue shifted.

ii.
```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The AI states common go-cue alignment without addressing the distinct video and behavior clocks.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's standalone `motionEnergy_*.mat` `me` data, paired with side-camera `frameTimes`; the loader unwraps nested object/struct wrappers and also reads but does not use `moveThresh`.

ii.
```python
motion_data, move_thresh = load_motion_energy(motion_path)
me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
_, ft0 = get_stream0(ti)
```

iii. The notes observed paired files, mixed wrappers, and a representative `moveThresh`, motivating a flexible loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Existing per-frame scalar motion energy is averaged into 75 ms bins if its length exactly matches side-camera frame times; otherwise the entire trial is marked missing. No additional smoothing is used.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full(..., np.nan)
```

iii. This treats the file as already spatially reduced; the AI does not explicitly explain the all-or-nothing length policy.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A per-session median over finite binned values defines low (0) and high (1). Missing/no-video bins are incorrectly encoded low (0), not requested code 2.

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The notes cite median thresholding but the README/output metadata omit the no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times have the behavioral go cue subtracted and motion-energy values are binned on the common edge grid. The video clock offset is omitted, and neural spikes are independently not shifted to go cue.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The AI's claim of go-cue alignment is unsupported by clock-offset handling.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Empty nested motion values become empty arrays; nonpositive time differences become NaN; bins without finite samples remain NaN during continuous processing. Missing movement bins ultimately become category 0. Motion/frame length mismatch makes the entire trial missing. Trials with all-zero neural activity are dropped. There is little explicit bounds/error handling for missing trajectory trials or inconsistent arrays.

ii.
```python
dt[dt <= 0] = np.nan
out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
b = np.zeros_like(a, dtype=np.int64)
```

iii. The notes emphasize mixed-format robustness but do not justify conflating missing behavior with low movement; this conflicts with the requested missingness categories.

## 11-a. What are the most time-consuming steps of the code?

i. The likely hotspots are repeated unit-by-trial spike masking/histograms, repeated per-trial DLC extraction and speed calculation (including calculating all feature speeds), and Python-loop temporal binning. File loading is also substantial. The AI did not provide profiling.

ii.
```python
for ui, (ttm, tr) in enumerate(...):
    for trial_idx in range(1, n_trials + 1):
for ti in range(n_trials):
    spd, tmid = speed_from_ts(ts, ft)
```

iii. Notes focus on data exploration and validation rather than runtime measurement, so this assessment is inferred from code.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested unit/trial spike loop could be replaced by a joint trial-by-time histogram per unit (as in the reference). `nanbin_mean` loops over every bin and could use `bincount` or grouped reductions. Output construction and repeated subject lookup could also be vectorized or mapped, while ragged trial file extraction reasonably remains iterative.

ii.
```python
for trial_idx in range(1, n_trials + 1):
    mask = tr == trial_idx
for bi in range(len(out)):
    m = idx == bi
```

iii. The AI offers no explicit vectorization justification.

## 11-c. What processing does the code repeat multiple times?

i. `get_stream1(ti)` and `speed_from_ts` are called once while trying bottom-camera tongue and again for paw; side-stream frame times are also fetched for tongue and motion energy. The same time input and constant per-trial labels are copied for every trial, and each binner repeatedly scans all frame-bin indices.

ii.
```python
spd, tmid = speed_from_ts(ts, ft)  # tongue loop
spd, tmid = speed_from_ts(ts, ft)  # paw block
```

iii. No rationale is documented; caching per-trial stream/speed results would avoid this repetition.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused behavioral fields (`bitRand`, `bitStart`, `sample`, `delay`, `reward`, and effectively `miss`), computes speed for every DLC feature even though only tongue/paw rows are selected, reads `moveThresh` only for metadata, computes outputs for all trials before filtering, and accepts CLI flags `--full`/`--show-processing` that do not change processing.

ii.
```python
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
xy = ts[:, :2, :]
output_trials = []
for ti in range(n_trials):
```

iii. The AI does not document these as deliberate; they appear to be convenience or incomplete implementation artifacts.
