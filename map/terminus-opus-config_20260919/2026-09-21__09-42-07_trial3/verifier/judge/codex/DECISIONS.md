# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script finds all session files with a glob over `/app/data/sub-*/*.nwb`, then processes each NWB file independently in `process_session`. It opens each file with `h5py.File` rather than `pynwb`, and reads trials from `intervals/trials`, events from `acquisition/BehavioralEvents`, units from `units`, and tongue/video from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Full conversion is parallelized with a multiprocessing pool.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with Pool(min(args.nproc, len(tasks))) as pool:
    for i, res in enumerate(pool.imap(process_session, tasks)):
```

```python
with h5py.File(fname, 'r') as f:
    tr = f['intervals/trials']
    ...
    be = f['acquisition/BehavioralEvents']
    ...
    u = f['units']
    ...
    tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md`, the AI says the DANDI release is organized as one NWB/HDF5 file per session under `sub-*`, so a file glob gives the complete dataset. It also says opening each file once and reading arrays in bulk is the efficient way to process the 50 GB dataset.

## 1-b. How are the data split into subjects?

i. The AI script treats each unique `sub-<id>` prefix from the session filenames as one mouse. It then builds `subjects` as the sorted unique ids and `subject_idx` as the per-session index into that list.

ii.
```python
session = {
    'file': os.path.basename(fname),
    'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
    'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
    ...
}
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI records that the dataset is laid out as 28 `sub-*` folders and repeatedly refers to `general/subject/subject_id` and the folder name as matching mouse identity. That explains why it chose the filename-derived subject id.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The file list is sorted once, and each file is processed independently. The stored `session_id` is parsed from the filename’s `ses-...` component rather than from the NWB identifier.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
session = {
    'file': os.path.basename(fname),
    'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
    'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
    ...
}
```

iii. The notes explicitly say the DANDI dandiset consists of 174 NWB files and describe the filename pattern `sub-<animalid>_ses-<yyyymmddThhmmss>_...nwb`, so the AI used the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`, one row per behavioral trial. The script reads trial-level columns from that table, reads `go_start_times`, asserts there is one go cue per trial, and thereafter indexes trials by a boolean keep mask.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
outcome = decode_array(tr['outcome'][:])
early_lick = decode_array(tr['early_lick'][:])
instruction = decode_array(tr['trial_instruction'][:])
...
ntrials_all = len(start_time)
```

```python
be = f['acquisition/BehavioralEvents']
go = be['go_start_times/timestamps'][:]
assert len(go) == ntrials_all, 'go cue count != trial count'
...
trials = np.where(keep)[0]
```

iii. In the notes, the AI says `go_start_times` has exactly one entry per trial in all sessions, which made the trials-table rows the canonical trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial-level filters. It computes a session-level stability mask from `units/obs_intervals` and `units/is_good_trials`, keeps only trials where at least 90% of good units are both observed and marked good, removes `auto_water` and `free_water` trials, removes trials whose full `[-2.5, 1.5]` window does not have at least one tongue-video frame in every 50 ms bin, and finally removes retained trials whose binned neural matrix is all zeros. It also rejects sessions with fewer than 10 retained trials.

ii.
```python
usable = observed & igt
trial_stable = usable.mean(axis=0) >= IGT_TRIAL_FRAC
...
keep = (auto_water == 0) & (free_water == 0) & trial_stable
```

```python
vidx = np.searchsorted(vts, edges_abs)
frames_per_bin = np.diff(vidx, axis=1)
video_ok = np.all(frames_per_bin > 0, axis=1)
keep = keep & video_ok
...
trials = np.where(keep)[0]
if len(trials) < 10:
    info['reject'] = 'fewer than 10 usable trials'
    return None, info
```

```python
nonempty = rates.sum(axis=(0, 2)) > 0
if not nonempty.all():
    info['n_trials_drop_nospikes'] = int((~nonempty).sum())
    rates = rates[:, nonempty, :]
    trials = trials[nonempty]
```

iii. The notes justify these filters as follows: `obs_intervals` and `is_good_trials` were needed because some sessions otherwise produced all-zero neural trials; auto/free-water trials were considered non-task trials; full video coverage was required so “not visible” could be distinguished from “no video”; and a minimum number of trials was kept for decoder usability.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, using per-trial go-cue times from `BehavioralEvents/go_start_times` to place the bin edges. Only units with `classification == 'good'` and that survive later stability filtering are used.

ii.
```python
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
```

```python
st_index = u['spike_times_index'][:]
st_all = u['spike_times'][:]
starts = np.concatenate([[0], st_index[:-1]])
edges_kept = edges_abs[trials]
```

iii. In the notes, the AI states that this is an ephys dataset, so spike times are the raw neural signal, and that `classification == 'good'` is the NWB encoding of the white-paper QC classifier output.

## 2-b. How is the `neural` data processed?

i. Spikes are converted to firing rates in Hz. The AI builds 50 ms bin edges around each trial’s go cue, uses `np.searchsorted` into each unit’s spike train to get cumulative spike counts at all edges, differences adjacent counts to get bin counts, and divides by bin width. There is no smoothing or normalization.

ii.
```python
edges_kept = edges_abs[trials]
flat_edges = edges_kept.ravel()
rates = np.zeros((len(good), nkept, NBINS), dtype=np.float32)
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32)
rates /= BIN_SIZE
```

iii. The notes say this matches the reference pipeline’s “sliding histogram -> Hz” logic, with the one intentional change that the decoder task required 50 ms non-overlapping bins instead of the paper’s 40 ms / 3.4 ms sliding grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only `units/classification == 'good'`. It then computes per-unit trial observability from `units/obs_intervals`, combines that with `units/is_good_trials`, drops trials where fewer than 90% of good units are usable, and drops any remaining unit that is flagged bad on any retained trial. Sessions with no QC-good units or no units stable over retained trials are rejected.

ii.
```python
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    info['reject'] = 'no QC-good units'
    return None, info
```

```python
oi_index = u['obs_intervals_index'][:]
oi_all = u['obs_intervals'][:]
...
igt_raw = u['is_good_trials'][:][good]
...
usable = observed & igt
trial_stable = usable.mean(axis=0) >= IGT_TRIAL_FRAC
...
unit_ok = usable[:, trials].all(axis=1)
...
good = good[unit_ok]
```

iii. The notes justify `classification == 'good'` as the direct analog of the published QC classifier output. They also justify the extra `obs_intervals`/`is_good_trials` logic as necessary because some sessions otherwise yielded all-zero neural trials or trial/unit instability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset. The script defines a fixed relative window from `-2.5` to `+1.5` s, adds that relative edge grid to each trial’s absolute go-cue timestamp, and bins spikes against those absolute edges.

ii.
```python
ALIGN_EVENT = 'go cue onset'
T_START = -2.5
T_END = 1.5
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
```

```python
go = be['go_start_times/timestamps'][:]
...
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]
```

iii. The notes repeatedly state that all relevant NWB timestamps are on a session-absolute clock, so go-cue alignment only requires expressing each modality on that common time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping 50 ms bins covering 4 s (`-2.5` to `+1.5` s) around the go cue. There is no later temporal rebinning; the spike times are binned directly onto that grid.

ii.
```python
T_START = -2.5
T_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((T_END - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. The AI notes that this is a deliberate deviation from the paper’s original binning because the decoder instructions explicitly require 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times/timestamps` and the per-trial go-cue timestamps. The script uses the last sample/tone onset before each go cue.

ii.
```python
sample_on = be['sample_start_times/timestamps'][:]
...
si = np.searchsorted(sample_on, go) - 1
tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
tone_rel = tone_abs - go
```

iii. The notes say trials can replay the sample epoch after early licks, so the last tone before the go cue is treated as the effective instructive tone for that executed trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each retained trial, the AI computes the last tone onset before the go cue, expresses that tone time relative to the go cue, and then subtracts that offset from each go-aligned bin center. This yields a continuous signed time-from-tone value at every 50 ms bin.

ii.
```python
si = np.searchsorted(sample_on, go) - 1
tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
tone_rel = tone_abs - go
assert np.all(np.isfinite(tone_rel[trials])), 'missing tone onset'
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. In the notes, the AI explicitly says it chose a signed continuous quantity, including negative values before tone onset, because that best matches “Time from tone onset in seconds (continuous, time-varying).”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by evaluating time-from-tone at the exact same 80 go-aligned bin centers used for the neural data.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. The notes say all streams were put on the same go-cue-relative grid, so the input and neural arrays share the same time axis by construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived primarily from the event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`. The script also reads `intervals/trials/photostim_onset`, `photostim_duration`, and `photostim_power` and uses them as a fallback when the event stream is missing.

ii.
```python
ps_onset = np.array([to_float(x) for x in tr['photostim_onset'][:]])
ps_dur = np.array([to_float(x) for x in tr['photostim_duration'][:]])
ps_power = np.array([to_float(x) for x in tr['photostim_power'][:]])
is_stim = np.isfinite(ps_power) & (ps_power > 0)
```

```python
if 'photostim_start_times' in be:
    pstart = be['photostim_start_times/timestamps'][:]
    pstop = be['photostim_stop_times/timestamps'][:]
...
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
if miss.any():
    stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
    stim_rel[miss, 1] = stim_rel[miss, 0] + ps_dur[miss]
```

iii. The notes say the event stream and trials table were cross-checked and found consistent, and that the event stream was preferred because it is already timestamped on the same absolute clock as the neural data.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts photostimulation into a binary time series over the 80 bins. For each retained trial, it computes the stimulation onset and offset relative to the go cue, then marks bins whose centers fall within that interval as `1`, with non-stimulated trials staying all zeros.

ii.
```python
stim_rel = np.full((ntrials_all, 2), np.nan)
...
for k, ti in enumerate(trials):
    if np.isfinite(stim_rel[ti, 0]):
        stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                         (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. The notes describe this as the natural decoder input representation because photostimulation is an on/off manipulation over time rather than a per-trial scalar.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by converting its absolute start and stop times into go-cue-relative times and then comparing them against the same `BIN_CENTERS_REL` used for neural binning.

ii.
```python
stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
...
stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                 (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. In the notes, the AI emphasizes that photostimulation always ended before the go cue and that using go-relative coordinates put it on the same axis as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from the lick event streams `BehavioralEvents/left_lick_times` and `BehavioralEvents/right_lick_times`, using the first lick after the go cue and before trial end. If there is no such lick, the choice is `no lick`. It also computes a sanity check against `outcome` and `trial_instruction`.

ii.
```python
left_lick = be['left_lick_times/timestamps'][:]
right_lick = be['right_lick_times/timestamps'][:]
```

```python
choice = np.full(ntrials_all, 2, dtype=np.int64)
for ti in trials:
    lo, hi = go[ti], stop_time[ti]
    l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
    r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
    ...
```

iii. The notes say the AI preferred direct lick events because they are the most direct source for actual lick direction, and it records that this agreed with the `outcome`/`trial_instruction` inference in 99.8% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first post-go lick is coded as `0` for left, `1` for right, and `2` for no lick. That per-trial value is then broadcast across all 80 bins of the output array.

ii.
```python
choice = np.full(ntrials_all, 2, dtype=np.int64)
...
if np.isinf(l0) and np.isinf(r0):
    choice[ti] = 2
elif l0 <= r0:
    choice[ti] = 0
else:
    choice[ti] = 1
```

```python
out = np.empty((4, NBINS), dtype=np.int64)
out[0] = choice[ti]
```

iii. The notes say per-trial outputs were broadcast across time so all outputs could share a common `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the `intervals/trials/outcome` column.

ii.
```python
outcome = decode_array(tr['outcome'][:])
```

iii. The notes list `outcome` as one of the explicit trial-table behavioral variables available in the NWB files, so no derivation beyond recoding was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers `0 = ignore`, `1 = miss`, `2 = hit`, and the resulting per-trial code is repeated across all 80 bins.

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
...
out[1] = outcome_code[ti]
```

iii. The AI’s notes describe this as a direct categorical recoding of the NWB trial outcome into the decoder’s required class labels.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the `intervals/trials/early_lick` column.

ii.
```python
early_lick = decode_array(tr['early_lick'][:])
```

iii. The notes list `early_lick` as an explicit trials-table field and justify keeping it because the decoder task explicitly requires early-lick as an output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early` to `1` and `no early` to `0`, then repeats that per-trial code across the 80 bins.

ii.
```python
early_code = (early_lick == 'early').astype(np.int64)
...
out[2] = early_code[ti]
```

iii. The notes describe this as a direct binary recoding of the trial label into the required output format.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the frame timestamps, the y coordinate column, and the likelihood column.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
vy = vdata[:, 1]
vlik = vdata[:, 2]
```

iii. The notes describe this series as the side-view DeepLabCut tongue tracking stream and identify `data[:, 1]` as y and `data[:, 2]` as tracking likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI keeps only frames with `likelihood > 0.9` as visible tongue. It computes session-level 40th and 60th percentiles on the visible-frame y values. For each retained trial, it takes frames in the `[-2.5, 1.5]` window around the go cue, bins them into 50 ms bins, averages visible y values within each bin using `np.bincount`, and assigns class `3` to bins with no visible frame. It also filters trials earlier so that every bin has at least one video frame, to distinguish missing video from “tongue not visible.”

ii.
```python
LIKELIHOOD_THRESH = 0.9
...
visible_all = vlik > LIKELIHOOD_THRESH
if visible_all.sum() < 100:
    info['reject'] = 'tongue never tracked in this session'
    return None, info
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
```

```python
for k, ti in enumerate(trials):
    i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
    ...
    b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
    cnt = np.bincount(b[fvis], minlength=NBINS)
    ysum = np.bincount(b[fvis], weights=fy[fvis], minlength=NBINS)
    has = cnt > 0
    ymean = np.zeros(NBINS)
    ymean[has] = ysum[has] / cnt[has]
```

iii. The notes justify the high likelihood threshold by saying tongue likelihood is strongly bimodal and that only clearly visible frames should count. They justify the full-window video requirement as necessary to avoid confusing missing video with a hidden tongue.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After computing the session’s 40th and 60th percentile y thresholds from visible frames, the AI digitizes each bin’s mean visible y into `0`, `1`, or `2` using those percentiles, and assigns `3` when the tongue is not visible in that bin.

ii.
```python
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
...
cls = np.full(NBINS, 3, dtype=np.int64)
cls[has] = np.digitize(ymean[has], [y_p40, y_p60])
tongue_class[k] = cls
```

iii. The notes say this follows the decoder task’s per-session discretization requirement, with the added explicit “not visible” class for bins lacking visible tongue frames.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-cue-relative trial window as the neural data. It locates video frames between `go + T_START` and `go + T_END`, converts frame times to offsets from the go cue, and bins them onto the same 50 ms grid.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]
vidx = np.searchsorted(vts, edges_abs)
frames_per_bin = np.diff(vidx, axis=1)
```

```python
i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
ft = vts[i0:i1] - go[ti]
b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
```

iii. The notes say the camera timestamps share the same session-absolute clock as spikes and behavioral events, so direct binning on the same go-cue-relative grid is sufficient.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. It decodes byte/string fields with helper functions, converts malformed photostim strings to `NaN`, falls back from missing photostim event streams to trials-table photostim timing, expands mismatched `is_good_trials` arrays onto observed trials using `obs_intervals`, rejects sessions with no QC-good units or no usable tongue tracking, drops trials with all-zero neural data, and encodes bins with no visible tongue as class `3`.

ii.
```python
def decode_array(arr):
    return np.array([dec(x) for x in arr])

def to_float(x):
    try:
        return float(dec(x))
    except Exception:
        return np.nan
```

```python
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
if miss.any():
    stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
    stim_rel[miss, 1] = stim_rel[miss, 0] + ps_dur[miss]
```

```python
if igt_raw.shape[1] == ntrials_all:
    igt = igt_raw
else:
    for i in range(len(good)):
        obs_idx = np.where(observed[i])[0]
        if len(obs_idx) == igt_raw.shape[1]:
            igt[i, obs_idx] = igt_raw[i]
```

iii. In the notes, the AI says these checks were added after earlier runs produced crashes or “all neural data is zero” warnings. It frames the strategy as excluding truly unrecorded data while representing genuine tongue invisibility as an explicit class.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session I/O, reading large spike and video arrays, and the per-unit spike binning loop. The script also writes side outputs and optionally plots diagnostics, but the core runtime is dominated by opening NWB files and running `searchsorted` for each retained unit.

ii.
```python
with h5py.File(fname, 'r') as f:
    ...
    st_index = u['spike_times_index'][:]
    st_all = u['spike_times'][:]
```

```python
rates = np.zeros((len(good), nkept, NBINS), dtype=np.float32)
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32)
```

iii. The notes explicitly say the full conversion is dominated by NWB I/O, reading the spike buffer and tongue-tracking arrays, and the per-unit `searchsorted` loop.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes over trials for spike binning, but it still has explicit loops over units (`obs_intervals`, spike binning), over retained trials for photostim, choice, tongue discretization, and final per-trial assembly. Those are the main remaining opportunities for more vectorization.

ii.
```python
for i, ui in enumerate(good):
    o = oi_all[oi_starts[ui]:oi_index[ui], 0]
    ...
```

```python
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
```

```python
for k, ti in enumerate(trials):
    if np.isfinite(stim_rel[ti, 0]):
        ...
for ti in trials:
    ...
for k, ti in enumerate(trials):
    ...
```

iii. The notes say the AI deliberately vectorized the expensive trial dimension of spike binning and used `bincount` within tongue processing, leaving the remaining loops because the data are ragged by unit and because those loops were a smaller share of runtime.

## 10-c. What processing does the code repeat multiple times?

i. The code avoids major multi-pass recomputation, but it does repeat some smaller operations. In the choice loop it calls `np.searchsorted` separately for left and right licks on each trial; it also restacks per-trial arrays during final assembly after earlier session-level computations. There is no second full pass over raw data for the main conversion.

ii.
```python
for ti in trials:
    lo, hi = go[ti], stop_time[ti]
    l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
    r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
```

```python
for k, ti in enumerate(trials):
    neural_list.append(rates[:, k, :])
    input_list.append(np.stack([time_from_tone[k].astype(np.float32),
                                stim_input[k]]).astype(np.float32))
```

iii. The notes emphasize that each file is opened once and processed in one pass, so the AI’s intent was to avoid repeated full-dataset work even though some local computations are repeated inside loops.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes several diagnostics and bookkeeping products that are not required by the decoder matrices themselves: session-selection statistics (`performance`, left/right correct counts), `choice_agreement`, optional processing plots, per-session diagnostic timing and rejection info, and extra metadata such as `anno`, `trial_indices`, `unit_indices`, tongue percentiles, and mean firing rates.

ii.
```python
performance = n_hit / max(n_hit + n_miss, 1)
n_correct_left = int(np.sum(control & (outcome == 'hit') & (instruction == 'left')))
n_correct_right = int(np.sum(control & (outcome == 'hit') & (instruction == 'right')))
info.update(performance=performance, n_correct_left=n_correct_left,
            n_correct_right=n_correct_right, ntrials_all=ntrials_all)
```

```python
info['choice_agreement'] = float(np.mean(ch_str[trials] == expected[trials]))
...
if show_processing:
    make_processing_plot(...)
```

```python
'session_info': [
    {'file': s['file'], 'subject': s['subject'], 'session_id': s['session_id'],
     'n_trials': len(s['neural']), 'n_neurons': int(s['neural'][0].shape[0]),
     'performance': s['performance'], 'n_trials_in_file': s['n_trials_all'],
     'tongue_y_percentiles': s['tongue_pctl'],
     'trial_indices': [int(x) for x in s['trial_indices']],
     'unit_indices': [int(x) for x in s['unit_indices']],
     'mean_firing_rate_hz': s['mean_rate']}
    for s in sessions],
```

iii. The notes frame these computations as sanity checks and documentation rather than decoder inputs or outputs. They were added to validate the conversion and support debugging, not because the downstream decoder directly consumes them.
