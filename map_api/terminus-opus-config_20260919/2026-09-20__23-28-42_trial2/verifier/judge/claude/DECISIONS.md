# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs for all NWB files under `data/sub-*/` sorted alphabetically, then opens each with `pynwb.NWBHDF5IO` and processes it via `convert_session()`. Subjects, trials, and units are read from within each NWB file. Parallel processing with `ProcessPoolExecutor` (16 workers) is used.

ii.
```python
def list_sessions():
    import glob
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```
```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    ...
    units = nwb.units
    ...
    tr = nwb.trials
    ...
    be = nwb.acquisition['BehavioralEvents'].time_series
```

iii. The AI documents that NWB is the published format and `pynwb` is its standard reader. 174 files found across 28 subject directories, matching the `dandiset.yaml` asset summary.

## 1-b. How are the data split into subjects?

i. Each NWB file records the animal in `nwb.subject.subject_id` (numeric string). Unique subject IDs are collected across all sessions, sorted, and indexed.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted({r['subject'] for r in good})
subject_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. The AI notes 28 subjects with 2-10 sessions each. Also stores `nwb.subject.description` (lab mouse name like `SC015`) in metadata.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each is identified by `nwb.identifier`. Session order follows the sorted file list.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
ident = nwb.identifier
```

iii. The AI documents that the dandiset stores one session per file. 173 of 174 files survive (one dropped for having no good units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioural trial. The number of go-cue events is asserted to match the trial count.

ii.
```python
tr = nwb.trials
trial_start = np.asarray(tr['start_time'][:], dtype=float)
...
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
assert len(go) == len(trial_start), 'go cue count != trial count in %s' % ident
```

iii. Documented in CONVERSION_NOTES that go_start_times has exactly one event per trial in all 174 sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) exclude `auto_water == 1` trials, (2) exclude `free_water == 1` trials, (3) post-hoc exclude trials with zero spikes across all good units in the analysis window. Sessions with fewer than 2 surviving trials are dropped. Early-lick, ignore, miss, and photostim trials are deliberately kept.

ii.
```python
keep = (auto_water == 0) & (free_water == 0)
...
# exclude trials with no ephys at all
spikes_per_trial = fr.sum(axis=(0, 2))
rec = spikes_per_trial > 0
```

iii. The AI references `get_regular_trial_mask` for auto/free-water exclusion (reward delivered independent of the animal's action). Zero-spike trials are described as ephys acquisition gaps. The CONVERSION_NOTES report 3,789 auto-/free-water + 1,657 no-ephys trials removed = 5,446 total (5.7%), yielding 89,544 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.spike_times` — the sorted spike times for each unit in session-absolute seconds. Only units with `classification == 'good'` are used. Go-cue times from `BehavioralEvents/go_start_times` define the bin edges.

ii.
```python
spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
...
fr = bin_spikes(spike_lists, go[kidx])
```

iii. The AI notes that `spike_times` is the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 to +1.5 s relative to the go cue, using `np.searchsorted` over flattened bin edges for all trials. Spike counts are divided by the bin width (0.05 s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spikes(spike_times_list, go_times):
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), ntr, NBINS), dtype=np.float32)
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

iii. The AI documents this as the analogue of the reference `sliding_histogram(..., rate=True)` with stride == bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units.classification == 'good'` are kept — the output of the region-specific QC classifiers described in the spike-sorting white paper. No additional metric thresholds are applied. Sessions with no good units are dropped. This retains 69,453 of 272,227 units (25.5%).

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'identifier': ident, 'dropped': 'no good units'}
```

iii. The AI extensively documents in CONVERSION_NOTES that `classification` is the classifier output, matching the reference pipeline's external good-unit lists. The one dropped session has all NaN classifications.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times share the same session-absolute clock. The bin edges relative to the go cue are added to each trial's go-cue time to produce absolute bin edges, and spikes are binned against those edges directly.

ii.
```python
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
...
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
```

iii. No resampling or interpolation is needed since everything is on one global clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins total from -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once as 81 edges relative to the go cue.

ii.
```python
BIN_SIZE = 0.05           # s, 50 ms bins
OFF_START = -2.5
OFF_END = 1.5
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. The AI notes this matches the decoder task specification. The reference code uses 40 ms/3.4 ms sliding bins; the 50 ms bins are prescribed by the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onset timestamps) and `go_start_times` (go cue times). The tone for each trial is the last sample-epoch start at or before its go cue.

ii.
```python
sample_starts = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
...
def tone_onset_times(sample_starts, trial_starts, go_times):
    tidx = np.searchsorted(trial_starts, sample_starts, side='right') - 1
    ok = (tidx >= 0) & (tidx < ntr)
    tidx, ev = tidx[ok], sample_starts[ok]
    ok2 = ev <= go_times[tidx]
    tidx, ev = tidx[ok2], ev[ok2]
    order = np.argsort(ev, kind='stable')
    out[tidx[order]] = ev[order]
    return out
```

iii. The AI documents that early licking triggers a replay of the sample/delay epoch, so a trial can contain several `sample_start_times`; the last one before the go cue is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Compute the tone-onset time relative to the go cue, then for each bin: `bin_center_relative_to_go - tone_relative_to_go` = time from tone onset.

ii.
```python
tone_rel_go = tone - go[kidx]
time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. Straightforward subtraction; no additional processing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both are defined on the same go-cue-aligned bin grid. The bin centers used for computing time-from-tone are the same bin centers used for the neural data.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. Alignment is guaranteed by construction since both use the same temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` timestamps (session-absolute times), together with `trial_start` times and go-cue times for trial assignment and time alignment.

ii.
```python
stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
```

iii. The AI uses the event-based timestamps rather than the per-trial `photostim_onset`/`photostim_duration` columns from the trials table. Cross-checked against the trials table with a mismatch counter.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Stimulation events are assigned to trials by `searchsorted` on trial start times. For each event, the onset and offset are expressed relative to the trial's go cue. A bin is marked 1 if the stimulation interval overlaps the bin (using bin edges), else 0. This produces a binary time-varying input.

ii.
```python
def photostim_binary(stim_on, stim_off, trial_starts, go_times):
    ps = np.zeros((ntr, NBINS), dtype=np.float32)
    tidx = np.searchsorted(trial_starts, stim_on, side='right') - 1
    for i, on, off in zip(tidx, stim_on, stim_off):
        on_rel, off_rel = on - go_times[i], off - go_times[i]
        overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
        ps[i, overlap] = 1.0
    return ps
```

iii. The AI notes this approach detects any overlap between the stimulation interval and each bin, rather than checking if the bin center falls within the interval.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, and compared against the go-cue-relative bin edges, ensuring alignment with the neural data grid.

ii.
```python
on_rel, off_rel = on - go_times[i], off - go_times[i]
overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
```

iii. Same temporal reference frame as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials.outcome` (`hit`/`miss`/`ignore`) and `trials.trial_instruction` (`left`/`right`). A hit means the animal licked the instructed side, a miss means the opposite side, ignore means no lick.

ii.
```python
choice = np.where(out_k == 'ignore', 2,
                  np.where(out_k == 'hit',
                           np.where(instr_k == 'left', 0, 1),
                           np.where(instr_k == 'left', 1, 0))).astype(np.int64)
```

iii. The AI documents that the animal's actual lick direction is not stored but is fully determined by the instructed side and the outcome. Verified against the first post-go lick in >99% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Per-trial constant, broadcast across all 80 bins.

ii.
```python
output_trials = [np.stack([np.full(NBINS, choice[i]),
                           np.full(NBINS, outcome_code[i]),
                           np.full(NBINS, early_code[i]),
                           tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. Straightforward encoding following the task specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table (`ignore`/`miss`/`hit`).

ii.
```python
outcome = np.asarray(tr['outcome'][:]).astype(str)
...
outcome_code = np.where(out_k == 'ignore', 0, np.where(out_k == 'miss', 1, 2)).astype(np.int64)
```

iii. No derivation needed; the trials table stores the outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit. Per-trial constant, broadcast across all 80 bins.

ii.
```python
outcome_code = np.where(out_k == 'ignore', 0, np.where(out_k == 'miss', 1, 2)).astype(np.int64)
```

iii. Follows the instruction ordering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table (`early`/`no early`).

ii.
```python
early = np.asarray(tr['early_lick'][:]).astype(str)
...
early_code = (early[kidx] == 'early').astype(np.int64)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial constant, broadcast across all 80 bins.

ii.
```python
early_code = (early[kidx] == 'early').astype(np.int64)
```

iii. Straightforward binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `data` of shape `(n_frames, 3)` = tongue_x, tongue_y, tongue_likelihood with matching `timestamps`.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries'].time_series
tongue = bts['Camera0_side_TongueTracking']
tt = np.asarray(tongue.timestamps[:], dtype=float)
tdata = np.asarray(tongue.data[:], dtype=float)
ty_binned = bin_tongue(tt, tdata[:, 1], tdata[:, 2], go[kidx])
```

iii. This is the only tongue measurement in the file; present in all 174 sessions at ~294 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Multiple steps: (1) Frames with `likelihood > 0.5` are marked as visible. (2) A 5-sigma velocity outlier rejection is applied — frames where the velocity jump exceeds 5 standard deviations are invalidated. (3) For each trial, valid (visible and not outlier) frames are binned into 50 ms bins and averaged. (4) Per-session percentiles (40th, 60th) of the binned visible values determine class boundaries. (5) Each bin is assigned class 0 (< p40), 1 (p40 to p60), or 2 (> p60). Bins with no valid frame are class 3 (not visible).

ii.
```python
def bin_tongue(ts, y, lik, go_times):
    visible = lik > TONGUE_LIK_THRESH
    valid = visible.copy()
    if visible.sum() > 10:
        iv = np.where(visible)[0]
        dy = np.diff(y[iv]) / np.maximum(np.diff(ts[iv]), 1e-6)
        s = np.std(dy)
        if s > 0:
            bad = np.abs(dy) > VELOCITY_SIGMA * s
            valid[iv[1:][bad]] = False
    ...

def discretize_tongue(y_binned):
    vis = np.isfinite(y_binned)
    cls = np.full(y_binned.shape, 3, dtype=np.int64)
    if vis.sum() > 0:
        p40, p60 = np.percentile(y_binned[vis], [40, 60])
        v = y_binned[vis]
        c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
        cls[vis] = c
```

iii. The AI references the method paper for the 5-sigma velocity outlier rejection. The CONVERSION_NOTES document that the tongue likelihood is strongly bimodal (~89% < 0.1, ~10% > 0.9), making the exact threshold immaterial.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles of the binned visible tongue-y values: class 0 if < 40th percentile, class 1 if >= 40th and <= 60th percentile, class 2 if > 60th percentile, class 3 if not visible.

ii.
```python
p40, p60 = np.percentile(y_binned[vis], [40, 60])
v = y_binned[vis]
c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
cls[vis] = c
```

iii. The AI uses `np.where` with `<` and `<=` boundaries. The instructions specify `< 40th`, `40th to 60th`, and `> 60th`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. Each trial's frame range is found by `searchsorted` on camera timestamps at `go + OFF_START` and `go + OFF_END`, and frames are assigned to bins by their offset from the trial window start, using the same bin width.

ii.
```python
t0 = go_times + OFF_START
lo = np.searchsorted(ts, t0)
hi = np.searchsorted(ts, go_times + OFF_END)
for i in range(ntr):
    bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
    np.clip(bi, 0, NBINS - 1, out=bi)
```

iii. The same go-cue-relative grid is used for both neural and tongue data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Session with all NaN classifications (never QC'd) — dropped because no good units. (2) Trials with no ephys data — detected by zero total spikes across all good units, excluded post-hoc. (3) Frames with low tracking confidence — marked as invalid and excluded from binning; bins with no valid frame become class 3. (4) Velocity outliers in tongue tracking — invalidated by 5-sigma threshold.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)  # NaN -> 'nan'
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'identifier': ident, 'dropped': 'no good units'}
...
spikes_per_trial = fr.sum(axis=(0, 2))
rec = spikes_per_trial > 0
...
visible = lik > TONGUE_LIK_THRESH
```

iii. The AI documents each case in CONVERSION_NOTES with specific numbers (e.g., 1 dropped session, 1,657 no-ephys trials).

## 10-a. What are the most time-consuming steps of the code?

i. Reading spike times (0.40 s/session mean) and binning (0.34 s/session mean) dominate. Pickling the 11.9 GB result takes ~19 s. Total conversion: 22 s with 16-24 parallel workers, plus pickle write.

ii. N/A (timing reported in conversion output)

iii. The AI reports per-step timing and uses parallel processing to reduce wall-clock time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) the per-neuron loop in `bin_spikes` running one `searchsorted` per unit (cannot be collapsed due to ragged spike time arrays), (2) the per-trial loop in `bin_tongue` (runs over trials rather than frames). The per-trial neural binning is already vectorized by flattening all trial edges.

ii.
```python
for k, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
    out[k] = np.diff(idx, axis=1)
...
for i in range(ntr):
    ...
    bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
```

iii. The AI notes the per-unit loop is inherent to ragged spike storage. The tongue loop is not a measurable fraction of runtime.

## 10-c. What processing does the code repeat multiple times?

i. No redundant computation. Each NWB file is opened once, each quantity computed once. Parallel workers each handle independent sessions.

ii. N/A

iii. The bin grid is defined once at module level and reused for all trials and sessions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes extensive diagnostic statistics per session (`info` dict with timing, coverage fractions, photostim mismatch counts, tongue percentile values) that go into metadata but are not used by the decoder. It also computes CCF coordinates and hemisphere-split coarse region labels (28 region labels including hemisphere), which may be more detailed than needed. The 5-sigma velocity outlier rejection is extra processing not present in the reference approach.

ii.
```python
info = {
    'identifier': ident,
    ...
    'frac_trials_observed_at_start': obs_frac_start,
    'frac_trials_observed_at_end': obs_frac_end,
    'photostim_table_mismatch': stim_mismatch,
    'timing': timing,
}
```

iii. This extra information is useful for validation and documentation, though not required for the decoder.
