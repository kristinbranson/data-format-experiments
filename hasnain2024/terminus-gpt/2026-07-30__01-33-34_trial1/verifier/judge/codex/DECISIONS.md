# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not load the full paper dataset. It hard-codes a 19-session randomized-delay subset in `RAND_SESSIONS`, reads only `data/RandomizedDelay_Ephys_Behavior`, and for each session loads `data_structure_<session>.mat` plus `motionEnergy_<session>.mat`. `data_structure` files are read with `h5py` first and `scipy.io.loadmat` as a fallback; motion-energy files are always read with `scipy.io.loadmat` and unwrapped.

ii.
```python
RAND_SESSIONS = [
    'JEB11_2022-05-10', ... ,'JEB24_2023-11-03'
]

base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS

df = base / f'data_structure_{key}.mat'
mf = base / f'motionEnergy_{key}.mat'
neural, inp, out, bri, info = process_session(df, mf)
```

```python
try:
    with h5py.File(data_path, 'r') as f:
        bp = extract_bp_h5(f)
        ...
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
```

iii. `README.md` says "19 randomized-delay ALM sessions." `CONVERSION_NOTES.md` says the randomized-delay paper subset had 19 sessions and should follow the explicit loader session list, and trajectory Step 43 says it would "initially target the randomized-delay directory only."

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session filename prefix before the underscore, e.g. `JEB11_2022-05-10 -> JEB11`. The `subjects` list is built in first-seen order from the chosen sessions, and `subject_idx` stores the index of each session's subject in that list.

ii.
```python
for key in keys:
    subj = key.split('_')[0]
    if subj not in subject_names:
        subject_names.append(subj)
    ...
    subject_idx.append(subject_names.index(subj))
```

iii. This follows the filename-based subject identification documented in the notes. The code never reads a subject id from inside the MAT files.

## 1-c. How are the data split into sessions?

i. One session is one entry in `RAND_SESSIONS`, corresponding to one `data_structure_*.mat` and one `motionEnergy_*.mat` file in `RandomizedDelay_Ephys_Behavior`. Each processed session becomes one element of `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`.

ii.
```python
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS

for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
```

iii. `README.md` and `CONVERSION_NOTES.md` both frame the output as the 19-session randomized-delay subset rather than the full 44-session ephys dataset.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `bp['Ntrials']`. The script builds one neural array, one input array, and one output array for each trial index `0..Ntrials-1`, then drops some trial indices afterward with `keep_idx`.

ii.
```python
n_trials = bp['Ntrials']
...
for ti in range(n_trials):
    ...
    tongue_vals.append(tongue_bin)
    paw_vals.append(paw_bin)
    me_vals.append(me_bin)
```

```python
per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
```

iii. The notes record that `obj.bp.ev.goCue` is a per-trial array and that `obj.clu.trial` indexes spikes by trial, so the agent used those native trial indices rather than reconstructing trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only when `early == 0` and `no == 0`, so early-lick and no-response/ignore trials are excluded. After that, any kept trial whose neural matrix is all zeros is dropped. There is no explicit photostimulation filter.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. `README.md` says "Early and no-response trials excluded." `CONVERSION_NOTES.md` also says to exclude early and ignore/no-go trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived from `obj.clu[*].trialtm` and `obj.clu[*].trial`. The code extracts per-unit spike times within trials and the trial number of each spike. It does not use cluster quality labels or `goCue` when constructing neural bins.

ii.
```python
trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32)
           for r in clu['trialtm'][()].flat]
trial = [np.asarray(deref(f, r)[()]).ravel().astype(int)
         for r in clu['trial'][()].flat]
```

iii. Trajectory Step 43 says `clu.tm`, `clu.trial`, and `clu.trialtm` were enough to reconstruct per-trial spike trains, and the final code uses `trial` and `trialtm`.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, raw `trialtm` values are histogrammed into a fixed window from `-2.5` to `2.5` with 75 ms bins. The resulting counts are stored directly as `float32`. There is no conversion to firing rate, no Gaussian smoothing, and no z-scoring.

ii.
```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
counts, _ = np.histogram(ttm[mask], bins=time_edges)
per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the agent chose 75 ms because the decoding scripts use `rez.binSize = 75`, and the code keeps that coarse binning all the way through.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no neuron-level quality control. It keeps every extracted unit, assigns all units to ALM, and only removes trials with all-zero neural activity.

ii.
```python
trialtm, trial, n_units = extract_clu_h5(f)
...
brain_region_idx = np.zeros((n_units,), dtype=int)
...
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. There is no code or note implementing `quality`-based filtering or a minimum firing-rate threshold; `README.md` only mentions trial exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Operationally, the code bins raw `trialtm` values directly into the `[-2.5, 2.5]` window and therefore implicitly treats `trialtm` as already aligned. It does not subtract `bp['goCue']` when forming neural bins, even though the metadata and README claim go-cue alignment.

ii.
```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    ...
    counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

```python
'metadata': {
    'task_description': 'Neural decoding aligned to go cue in randomized-delay ALM sessions',
    'temporal_alignment_event': 'go cue onset',
}
```

iii. The justification in the notes and README is that the dataset should be go-cue aligned, but the implementation never performs the subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 75 ms bins over `[-2.5, 2.5]`, yielding 67 time bins. This is a direct coarse binning choice; there is no additional rebinning after bin construction.

ii.
```python
BIN_SIZE = 0.075
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. `CONVERSION_NOTES.md` explicitly says the decoding scripts' `rez.binSize = 75` ms suggested a 75 ms analysis bin, and `README.md` repeats "Time bin size: 75 ms."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not computed from a trial-specific raw array. It is a synthetic vector of shared bin centers defined by `T_START`, `T_END`, and `BIN_SIZE`; conceptually it is intended to represent time relative to the go cue.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2

def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. The notes say to use go-cue alignment and 75 ms bins, so the time input was made from that common analysis grid rather than from a dedicated raw variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes evenly spaced bin centers and copies the same `1 x T` vector into every trial. There is no per-trial transformation beyond replication.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
input_trials = make_time_input(n_trials, time_centers)
```

iii. This follows the agent's general design choice of a single shared time grid for all trials and streams.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time_edges`/`time_centers` grid as the neural histograms, so each input bin corresponds to the same nominal interval used for neural binning.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
counts, _ = np.histogram(ttm[mask], bins=time_edges)
input_trials = make_time_input(n_trials, time_centers)
```

iii. The agent's code is built around one shared grid, even though the neural stream is not actually shifted by `goCue`.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The script derives `lick_direction` only from the trial-side flags `bp['R']` and `bp['L']`. It does not use `hit`, `miss`, or `no` to infer the animal's actual lick.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. There is no separate recorded lick-direction variable in the code, so the agent appears to have treated instructed side as a proxy for lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It is encoded as a binary trial label: `1` when `R > L`, else `0`. The value is then repeated across all time bins for the trial. Misses are not flipped, and no-response trials are removed upstream rather than assigned a separate class.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. `README.md` advertises only left/right classes, consistent with this simplified binary encoding.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `behavioral_context` is derived from `bp['autowater']`.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly maps `autowater` to WC and `~autowater` to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater flag is relabeled into the requested classes: WC = 0, DR = 1. The value is repeated across all time bins of the trial.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
...
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. Both the notes and the final `output_values` reflect this binary WC/DR mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The script derives `outcome` only from `bp['hit']`. It does not use `miss` except indirectly through the binary complement after trial filtering.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. Because no-response trials are dropped before output assembly, the code effectively treats remaining non-hit trials as incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is binary: correct = 1 on hit trials, incorrect = 0 otherwise. The value is repeated across all time bins. Ignore/no-response trials are excluded earlier rather than kept as a third class.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
...
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. `README.md` lists only `incorrect/correct`, and the notes say early and ignore/no-go trials should be excluded.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` feature arrays (`ts`) and `frameTimes` from the two camera streams, plus `bp['goCue']` for trial alignment. The code looks for tongue-like feature names in both streams and uses the first stream that yields finite values for a trial.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
...
ts, ft = getter(ti)
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. Trajectory Step 49 says the agent resolved the two-stream mapping and planned to use a tongue feature from either stream, aligned relative to `goCue`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code computes frame-to-frame speed from x/y coordinates, converts it to mid-frame timestamps, bins those speeds into the session time grid, and keeps the first camera view that produces finite binned values for the trial. It does not apply a likelihood cutoff, does not smooth positions, and does not combine or normalize both tongue views.

ii.
```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
```

```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ...
    candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
    if np.any(np.isfinite(candidate)):
        tongue_bin = candidate
        break
```

iii. The trajectory shows the agent intentionally implemented a simpler "get a working end-to-end sample converter" path for randomized-delay sessions, rather than reproducing the paper's full camera processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All finite tongue-velocity bins from the session are pooled, a session median is computed, and bins are labeled `1` if they are at or above the median and `0` otherwise. Non-finite bins remain at the default value `0`, so missingness is collapsed into the low class.

ii.
```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    ...
    b = np.zeros_like(a, dtype=np.int64)
    if np.isfinite(thr):
        valid = np.isfinite(a)
        b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. `README.md` says the movement outputs were "discretized per session using median thresholding." No separate handling for unseen frames is documented.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue speeds are aligned by subtracting the trial's `goCue` from mid-frame times and binning into the same fixed session grid. The code does not compute or subtract any video-to-behavior clock offset.

ii.
```python
go = bp['goCue'][ti]
...
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes and README emphasize go-cue alignment, but there is no implementation of the video offset logic described in the reference workflow.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` `ts` and `frameTimes` in stream 1, using `top_paw` if present and falling back to `bottom_paw`. `bp['goCue']` is then used for relative timing.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
...
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Trajectory Step 49 says the agent identified stream 1 as the paw-containing stream and preferred `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code computes frame-to-frame x/y speed for the selected paw feature, bins the resulting speeds, and stores the binned values. It does not apply likelihood filtering, smoothing, or any camera-scale normalization.

ii.
```python
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. This mirrors the simplified kinematic pipeline the agent adopted for the randomized-delay subset.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, paw velocity is median-split per session with `threshold_session_bins`, and NaN bins are left at class `0`.

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. The README documents only median thresholding, not a separate missing-data class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw speeds are aligned by subtracting `goCue` from mid-frame times and binning onto the same fixed grid as the neural arrays. No video-clock offset is applied.

ii.
```python
go = bp['goCue'][ti]
...
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The intended alignment event is go cue, but the implementation omits the session-level video offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, after unwrapping `me.data`, and is time-indexed using stream-0 `frameTimes` plus `bp['goCue']`.

ii.
```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    ...
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
```

```python
me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The notes document `me.data` and `moveThresh`, and the trajectory shows the agent treating the standalone motion-energy file as the motion-energy source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is not differentiated or smoothed. The code bins each per-frame motion-energy trace into the session grid with `nanbin_mean`, then later median-thresholds the pooled session values.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The agent treated the saved motion-energy trace as the already-computed signal of interest and only rebinned it.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is pooled across the session, split at the median, and encoded as `0/1`. Missing or mismatched traces end up as NaN during binning and then default to `0`.

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. `README.md` says motion energy is discretized by a per-session median threshold; the code does exactly that and does not preserve a missing-data class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The code subtracts `goCue` from stream-0 frame times and bins motion-energy values onto the shared analysis grid. It does not apply any video offset between camera and behavior clocks.

ii.
```python
go = bp['goCue'][ti]
...
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The notes state that the target alignment event is go cue, but there is no implementation of the reference's video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles MATLAB wrapper/layout variation explicitly, but missing numeric data are mostly collapsed rather than represented separately. Motion-energy traces are unwrapped recursively; frame/trace length mismatches are turned into all-NaN bins; NaN bins then become class `0` in `threshold_session_bins`. Trials with all-zero neural matrices are dropped.

ii.
```python
while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
    raw = raw.data
...
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
b = np.zeros_like(a, dtype=np.int64)
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The trajectory and notes focus on mixed MAT-file formats and wrapper unwrapping. There is no documented strategy for preserving missing-video bins as a distinct class.

## 11-a. What are the most time-consuming steps of the code?

i. The likely hotspots are session loading plus the Python-level nested loops used for spike binning and per-trial kinematic/motion-energy binning. In particular, `bin_unit_spikes_for_trials` loops over every unit and every trial, and `extract_binned_behavior_generic` loops over every trial and repeatedly re-reads trajectory arrays.

ii.
```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

```python
for ti in range(n_trials):
    ...
    ts, ft = getter(ti)
    ...
    ts, ft = get_stream1(ti)
```

iii. The agent did not document a final runtime analysis, but the structure of the code makes these loops the main computational work beyond file I/O.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the unit-by-trial spike histogram loop in `bin_unit_spikes_for_trials`, the per-bin loop inside `nanbin_mean`, and repeated per-trial trajectory extraction in `extract_binned_behavior_generic`.

ii.
```python
for bi in range(len(out)):
    m = idx == bi
    if np.any(m):
        vv = values[m]
        if np.any(np.isfinite(vv)):
            out[bi] = np.nanmean(vv)
```

iii. The human reference used a single `histogram2d` call for spike binning; this implementation leaves much more work in explicit Python loops.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some work. For each trial it may read stream 1 twice, once while testing tongue features and again for paw velocity; it also recomputes binned summaries separately for tongue, paw, and motion energy rather than caching shared timing information. Trial-level labels are also expanded into full-length time series for every output.

ii.
```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ts, ft = getter(ti)
    ...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
```

```python
output_trials.append(np.vstack([
    np.full((len(time_centers),), lick_dir[ti], dtype=np.int64),
    np.full((len(time_centers),), context[ti], dtype=np.int64),
    np.full((len(time_centers),), outcome[ti], dtype=np.int64),
    tongue_bin[ti], paw_bin[ti], me_bin[ti],
]))
```

iii. None of this repetition is called out in the notes; it is visible from the final code structure.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several fields it never uses (`bitStart`, `sample`, `delay`, `reward`, `bitRand`, `moveThresh` except for metadata). It also computes whole-trial time-varying copies of constant per-trial labels and may examine candidate tongue features/cameras that are later discarded.

ii.
```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()
```

```python
move_thresh = getattr(me, 'moveThresh', None)
...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. The notes mention `moveThresh` and several event fields during exploration, but the final converter does not use them for the decoded outputs.
