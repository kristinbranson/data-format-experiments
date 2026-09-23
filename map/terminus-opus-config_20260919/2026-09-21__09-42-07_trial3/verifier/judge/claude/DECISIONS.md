# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files found via `glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb'))`. Each file is opened with `h5py.File` (not `pynwb`), and trials, units, and behavioral events are read from HDF5 groups (`intervals/trials`, `units`, `acquisition/BehavioralEvents`). Sessions are processed in parallel with `multiprocessing.Pool`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with h5py.File(fname, 'r') as f:
    tr = f['intervals/trials']
    start_time = tr['start_time'][:]
    # ...
    u = f['units']
    classification = decode_array(u['classification'][:])
    # ...
    be = f['acquisition/BehavioralEvents']
    go = be['go_start_times/timestamps'][:]
```

iii. The AI chose h5py over pynwb for direct HDF5 access. The glob pattern finds all 174 NWB files. Multiprocessing (16 workers) is used for efficiency.

## 1-b. How are the data split into subjects?

i. The subject ID is extracted from the NWB filename by splitting on `_` and stripping the `sub-` prefix, rather than reading `nwb.subject.subject_id` from inside the file.

ii.
```python
'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
```

iii. The AI extracts subject IDs from filenames, which gives numeric IDs like `440956`. This produces the same subject identifiers as `nwb.subject.subject_id` since the filenames encode the same IDs.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The session ID is extracted from the filename by splitting on `_` and stripping the `ses-` prefix.

ii.
```python
'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
```

iii. Since each NWB file contains one session, the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. The number of go-cue events is asserted to match the trial count.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
# ...
go = be['go_start_times/timestamps'][:]
assert len(go) == ntrials_all, 'go cue count != trial count'
```

iii. The trials table provides one row per trial, matching the go-cue event count.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filtering criteria:
1. **Session-level**: Performance > 65% and >= 50 correct lick-left and >= 50 correct lick-right trials (from the data paper). This drops 29 sessions.
2. **Auto-water and free-water trials** removed.
3. **`is_good_trials`** stability: Trials where >= 90% of good units are flagged good are kept; units flagged bad on retained trials are also dropped.
4. **Video coverage**: Trials must have at least one video frame in every 50ms bin of the [-2.5, 1.5]s window.
5. **All-zero neural data**: Trials with no spikes from any unit are removed.
6. Minimum 10 usable trials per session (otherwise session dropped).

ii.
```python
# session-level selection
if performance <= MIN_PERFORMANCE:
    info['reject'] = 'performance %.3f <= %.2f' % (performance, MIN_PERFORMANCE)
    return None, info
if n_correct_left < MIN_CORRECT_PER_SIDE or n_correct_right < MIN_CORRECT_PER_SIDE:
    info['reject'] = 'too few correct trials (L %d, R %d)' % (n_correct_left, n_correct_right)
    return None, info

# trial mask
keep = (auto_water == 0) & (free_water == 0) & trial_stable
# ...
keep = keep & video_ok

# units that are still flagged bad on a retained trial are dropped
unit_ok = usable[:, trials].all(axis=1)
```

iii. The AI justified session-level criteria as matching the data paper's criteria. The video coverage requirement ensures tongue output is meaningful. The `is_good_trials` curation is described as an extra step enabled by the NWB release. This results in 138 sessions (vs 173 in the reference).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (with `units/spike_times_index`) for units with `classification == 'good'`, binned relative to `go_start_times`.

ii.
```python
st_index = u['spike_times_index'][:]
st_all = u['spike_times'][:]
starts = np.concatenate([[0], st_index[:-1]])
# ...
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32)
rates /= BIN_SIZE  # Hz
```

iii. `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning [-2.5, 1.5]s relative to the go cue. Counts are divided by bin width (0.05s) to get firing rates in Hz. No smoothing or normalization.

ii.
```python
edges_kept = edges_abs[trials]  # (ntrials_kept, NBINS+1)
flat_edges = edges_kept.ravel()
rates = np.zeros((len(good), nkept, NBINS), dtype=np.float32)
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32)
rates /= BIN_SIZE  # Hz
```

iii. Firing rate in Hz matches the reference code's `sliding_histogram(rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with `classification == 'good'` are kept. Additionally, `is_good_trials` is used: units flagged as bad on any retained trial are dropped. Sessions with no good units are dropped.

ii.
```python
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
# ...
igt_raw = u['is_good_trials'][:][good]
# ...
usable = observed & igt
# ...
unit_ok = usable[:, trials].all(axis=1)
if unit_ok.sum() == 0:
    info['reject'] = 'no units stable over the retained trials'
    return None, info
good = good[unit_ok]
```

iii. The `classification == 'good'` filter follows the white-paper QC classifier. The additional `is_good_trials` filter is an NWB-specific per-probe manual annotation of stable trials, removing units that were not stable during retained trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as absolute times by adding the go-cue time to the relative edges. Spike times are already on the same absolute clock, so `searchsorted` directly gives the counts.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]  # (ntrials, NBINS+1)
# ...
flat_edges = edges_kept.ravel()
counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
```

iii. All NWB times share a session-absolute clock, so alignment to the go cue only requires adding the go-cue time to the relative bin edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning [-2.5, 1.5]s relative to the go cue. No rebinning from an intermediate representation.

ii.
```python
T_START = -2.5
T_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((T_END - T_START) / BIN_SIZE))  # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. The 50ms bin width and [-2.5, 1.5]s window are specified by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onsets) and `go_start_times` (go cue). The last sample onset before each go cue is used as the tone for that trial.

ii.
```python
sample_on = be['sample_start_times/timestamps'][:]
# ...
si = np.searchsorted(sample_on, go) - 1
tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
tone_rel = tone_abs - go  # negative, ~ -1.85 s
```

iii. Early-lick replay causes multiple sample onsets per trial; the last one before the go cue is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The bin centers (relative to go cue) minus the tone onset time (relative to go cue) gives the time from tone onset at each bin center.

ii.
```python
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. Since `tone_rel` is negative (tone before go cue), subtracting it adds the absolute delay, giving positive time-from-tone values. This is algebraically equivalent to the reference's `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin center grid (relative to go cue) is used for both neural data and this input, so they are inherently aligned.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. Using the same go-cue-relative grid for both neural and input ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primarily from `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, with fallback to the trials table columns `photostim_onset` and `photostim_duration` (relative to trial start). Also uses `photostim_power` to identify stimulation trials.

ii.
```python
if 'photostim_start_times' in be:
    pstart = be['photostim_start_times/timestamps'][:]
    pstop = be['photostim_stop_times/timestamps'][:]
# ...
# fall back to the trials table where the event stream is missing
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
if miss.any():
    stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
    stim_rel[miss, 1] = stim_rel[miss, 0] + ps_dur[miss]
```

iii. The AI uses a dual-source approach: event streams first, trials table as fallback. The reference uses only the trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls within [stim_start, stim_stop] (inclusive on both ends), 0 otherwise. Non-stim trials remain 0.

ii.
```python
for k, ti in enumerate(trials):
    if np.isfinite(stim_rel[ti, 0]):
        stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                         (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. Note: the AI uses `<=` for the offset comparison, while the reference uses `<`. This could include one extra bin at the boundary.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are converted to go-cue-relative times, then compared against the same bin centers used for neural data.

ii.
```python
stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
# ...
stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                 (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. Same go-cue-relative coordinate system as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from actual lick times: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first lick after the go cue (within the trial stop time) determines the direction.

ii.
```python
left_lick = be['left_lick_times/timestamps'][:]
right_lick = be['right_lick_times/timestamps'][:]
# ...
choice = np.full(ntrials_all, 2, dtype=np.int64)  # 2 = no lick
for ti in trials:
    lo, hi = go[ti], stop_time[ti]
    l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
    r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
    l0 = l0 if l0 <= hi else np.inf
    r0 = r0 if r0 <= hi else np.inf
    if np.isinf(l0) and np.isinf(r0):
        choice[ti] = 2
    elif l0 <= r0:
        choice[ti] = 0
    else:
        choice[ti] = 1
```

iii. The AI also validates against the instruction x outcome derivation (99.8% agreement). The reference instead derives choice entirely from instruction x outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Per-trial value broadcast across all 80 bins.

ii.
```python
out = np.empty((4, NBINS), dtype=np.int64)
out[0] = choice[ti]
```

iii. Same coding scheme as the reference (left=0, right=1, no lick=2).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, containing 'ignore', 'miss', 'hit'.

ii.
```python
outcome = decode_array(tr['outcome'][:])
# ...
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit. Per-trial value broadcast across all 80 bins.

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
# ...
out[1] = outcome_code[ti]
```

iii. Same coding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing 'early' and 'no early'.

ii.
```python
early_lick = decode_array(tr['early_lick'][:])
# ...
early_code = (early_lick == 'early').astype(np.int64)
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value broadcast across all 80 bins.

ii.
```python
early_code = (early_lick == 'early').astype(np.int64)
# ...
out[2] = early_code[ti]
```

iii. Same coding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 is `tongue_y`, column 2 is `tongue_likelihood`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
vy = vdata[:, 1]
vlik = vdata[:, 2]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood > 0.9 are considered visible. Per-session percentiles (40th/60th) are computed from the raw visible y-values (not from bin means). Per-trial bins: mean y of visible frames in each bin, discretized with `np.digitize` against the session percentiles. Bins with no visible frame get class 3 (not visible).

ii.
```python
LIKELIHOOD_THRESH = 0.9
# ...
visible_all = vlik > LIKELIHOOD_THRESH
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
# ...
for k, ti in enumerate(trials):
    # ...
    cnt = np.bincount(b[fvis], minlength=NBINS)
    ysum = np.bincount(b[fvis], weights=fy[fvis], minlength=NBINS)
    has = cnt > 0
    ymean[has] = ysum[has] / cnt[has]
    cls[has] = np.digitize(ymean[has], [y_p40, y_p60])
```

iii. The AI uses a likelihood threshold of 0.9 (vs 0.5 in the reference) and computes percentiles from raw visible frame values rather than from 50ms bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four categories: 0 (below 40th percentile), 1 (40th-60th), 2 (above 60th), 3 (not visible). Session-level percentiles computed from raw visible frame y-values.

ii.
```python
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
# ...
cls[has] = np.digitize(ymean[has], [y_p40, y_p60])  # 0,1,2
# Default is 3 (not visible)
```

iii. The 40th/60th percentile split and 4 categories match the instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same session-absolute clock. For each trial, frames in the [go+T_START, go+T_END] window are found via `searchsorted`, then assigned to bins based on their offset from trial start.

ii.
```python
i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
ft = vts[i0:i1] - go[ti]
# ...
b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
```

iii. Same go-cue-relative bin assignment as neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple cases handled:
- **Session with NaN classification**: `decode_array` converts non-string entries, `classification == 'good'` returns no matches, session dropped.
- **Trials outside obs_intervals**: Per-unit `obs_intervals` used to identify observed trials; unobserved trials excluded.
- **is_good_trials mismatch**: When the column count doesn't match trial count, the array is expanded onto observed trials.
- **Trials with no video coverage**: Dropped if any bin lacks a video frame.
- **All-zero neural trials**: Removed after spike binning.
- **Tongue not visible**: Assigned class 3.
- **Auto-water and free-water trials**: Removed.

ii.
```python
# obs_intervals per unit
observed = np.zeros((len(good), ntrials_all), dtype=bool)
for i, ui in enumerate(good):
    o = oi_all[oi_starts[ui]:oi_index[ui], 0]
    # ...

# is_good_trials expansion
if igt_raw.shape[1] == ntrials_all:
    igt = igt_raw
else:
    # expand onto observed trials

# no-spike trials
nonempty = rates.sum(axis=(0, 2)) > 0
if not nonempty.all():
    rates = rates[:, nonempty, :]
```

iii. The AI handles more edge cases than the reference, including per-unit obs_intervals, is_good_trials mismatch, video coverage, and all-zero neural trials.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and spike binning dominate. The AI uses multiprocessing (16 workers) to parallelize across sessions. Total conversion took 36.9s for 174 sessions.

ii.
```python
with Pool(min(args.nproc, len(tasks))) as pool:
    for i, res in enumerate(pool.imap(process_session, tasks)):
        results.append(res)
```

iii. Parallelization across sessions provides significant speedup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain:
1. Per-unit loop for spike binning (inherent due to ragged spike times).
2. Per-trial loop for tongue y classification.
3. Per-trial loop for choice derivation from lick times.

ii.
```python
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
# ...
for k, ti in enumerate(trials):
    # tongue processing
# ...
for ti in trials:
    # choice from lick times
```

iii. The per-unit spike binning loop is necessary due to ragged data. The tongue and choice loops iterate over trials. The choice derivation loop is an additional loop not present in the reference (which derives choice algebraically from instruction x outcome).

## 10-c. What processing does the code repeat multiple times?

i. Spike times indexing is read once. The choice derivation from lick times is redundant since the same result can be obtained from instruction x outcome (as the reference does). The AI also reads electrode coordinates for hemisphere/region assignment, which is additional processing.

ii. N/A

iii. The CCF-to-region mapping with electrode coordinates is extra processing not in the reference, which uses a simpler `region_label` function on `anno_name`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. The CCF coordinate lookup and complex region mapping (hemisphere prefix + coarse region) produces region labels like "left Thalamus" that are more elaborate than needed.
2. The choice derivation from lick times is redundant with the instruction x outcome derivation.
3. The `choice_agreement` sanity check is computed but only stored in the info dict.
4. Per-session info JSON with detailed diagnostics is written but not used by the decoder.

ii.
```python
# Complex region mapping
regions = np.array(['%s %s' % (h, ccf_to_region(a, p))
                    for h, a, p in zip(hemi, anno, ap_um)])
```

iii. The elaborate region mapping and lick-time choice derivation add complexity without changing the decoder's behavior.
