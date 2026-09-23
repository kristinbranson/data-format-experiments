# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` sorted alphabetically, then processes each file with `pynwb.NWBHDF5IO`. A multiprocessing pool (default 16 workers) processes sessions in parallel. Each file is opened once and trials, units, electrodes, behavioral events, and video tracking are read from within it.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
# ...
with Pool(min(args.nproc, len(files))) as pool:
    for i, r in enumerate(pool.imap(_worker, jobs)):
        results.append(r)
```

Per session:
```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    # ...
    trials = nwb.intervals['trials']
    df = trials.to_dataframe()
    units = nwb.units
    be = nwb.acquisition['BehavioralEvents'].time_series
```

iii. The AI's CONVERSION_NOTES state: "174 NWB files, 28 subjects, 50 GB. DANDI:000363". The AI verified the file count matches `dandiset.yaml`.

## 1-b. How are the data split into subjects?

i. Each NWB file records its subject in `nwb.subject.subject_id` (a numeric string like `'440956'`). At assembly, unique subject IDs are sorted and an index is assigned per session.

ii.
```python
subject_id = str(nwb.subject.subject_id)
# ...
subjects = sorted(set(r['subject_id'] for r in sessions))
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject_id']] for r in sessions], dtype=np.int64),
```

iii. The AI uses the numeric `subject_id` field. CONVERSION_NOTES confirm 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. No grouping or splitting logic is applied. Session order follows the sorted file list.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
# ...
session_id = nwb.identifier
```

iii. CONVERSION_NOTES: "174 NWB files, 28 subjects"; 173 sessions after dropping one with 0 good units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.intervals['trials']`), one row per trial. The AI asserts the number of go-cue events matches the trial count.

ii.
```python
trials = nwb.intervals['trials']
df = trials.to_dataframe()
n_trials_all = len(df)
# ...
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
if len(go_times) != n_trials_all:
    raise RuntimeError(...)
```

iii. CONVERSION_NOTES: "go_start_times has exactly one entry per trial in all 174 sessions (verified)."

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial filters:
1. Exclude `auto_water` and `free_water` trials
2. Require go cue falls within [start_time, stop_time]
3. Require all good units have `obs_intervals` coverage for the trial (intersection over all good units)
4. Post-binning, drop trials with zero spikes across all units
5. Drop sessions with fewer than 2 surviving trials

ii.
```python
keep = (auto_water == 0) & (free_water == 0)
keep &= (go_times >= start_time) & (go_times <= stop_time)
# ...
obs_trial = np.ones(n_trials_all, dtype=bool)
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
    m = np.zeros(n_trials_all, dtype=bool)
    if oi.size:
        rows = np.searchsorted(start_time, oi[:, 0] + 1e-6) - 1
        rows = rows[(rows >= 0) & (rows < n_trials_all)]
        m[rows] = True
    obs_trial &= m
keep &= obs_trial
# ...
has_spikes = spikes_in_trial > 0
if n_trials_nospike:
    trial_idx = trial_idx[has_spikes]
```

iii. CONVERSION_NOTES explain that auto_water and free_water trials are excluded because "reward is delivered independently of the animal's choice, so choice/outcome labels do not reflect a decision" (following the reference `get_regular_trial_mask`). Early-lick, ignore, and photostim trials are deliberately kept as they are required decoder outputs/inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for each unit classified as `'good'`, combined with go-cue times from `BehavioralEvents/go_start_times`.

ii.
```python
spike_index = units['spike_times']
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
```

iii. CONVERSION_NOTES: "spike_times is the only neural representation in the file."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `searchsorted`, then counts are divided by the bin width (0.05s) to produce firing rates in Hz. Bin edges are clipped to trial boundaries. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
flat_edges = edges_clipped.ravel()
rates = np.zeros((n_units, n_trials, NBINS), dtype=np.float32)
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32)
rates /= BIN_SIZE   # spikes/s
```

iii. CONVERSION_NOTES: "spike counts per 50 ms bin / 0.05 s" matching the reference code's `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters on units:
1. `units['classification'] == 'good'` (QC classifier verdict)
2. Valid CCF annotation that maps to one of 14 coarse regions AND finite ML coordinate for hemisphere assignment

Units are mapped to hemisphere-resolved coarse brain regions (e.g., "left ALM", "right Thalamus") using a keyword-based mapper on `anno_name` and `electrodes.x` with midline at 5700 um. A session with no surviving units is dropped.

ii.
```python
classification = np.asarray([str(c) for c in units['classification'][:]])
good = classification == 'good'
if good.sum() == 0:
    return None
# ...
region_idx = np.full(len(classification), -1, dtype=np.int64)
for i in np.where(good)[0]:
    reg = coarse_region(anno[i])
    if reg is None or not np.isfinite(ml[i]):
        continue
    side = 'left' if ml[i] >= ML_MIDLINE else 'right'
    region_idx[i] = BRAIN_REGION_INDEX['%s %s' % (side, reg)]
use_unit = good & (region_idx >= 0)
```

iii. CONVERSION_NOTES: "69,453 good units across 173 sessions" matching the paper's ~69,943. The AI notes all 293 annotations were mapped (0 unknown), so the region filter doesn't drop any units in practice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are constructed relative to each trial's go-cue time. Since spike times and go-cue times share the same session-absolute clock, no resampling is needed.

ii.
```python
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
# ...
edges = go[:, None] + BIN_EDGES_REL[None, :]  # (n_trials, NBINS+1)
```

iii. CONVERSION_NOTES: "everything is timestamped on one global clock."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins from -2.5s to +1.5s relative to the go cue. The bin grid is defined once at module level.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_STOP = 1.5
NBINS = int(round((T_STOP - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. CONVERSION_NOTES note the reference used 40ms/3.4ms stride but the task mandates 50ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onsets) and the go-cue time of each trial. The last sample onset before each trial's go cue is used.

ii.
```python
sample_times = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
# ...
pos = np.searchsorted(sample_times, go, side='left') - 1
tone_onset = np.where(pos >= 0, sample_times[np.clip(pos, 0, len(sample_times) - 1)], np.nan)
```

iii. CONVERSION_NOTES: "Early-lick trials replay the sample epoch, so a trial can carry more than one tone; the last one before the go cue is the one the animal actually used."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial: `time_from_tone = (go - tone_onset) + bin_center`. If no valid tone onset is found (tone before trial start or missing), a fallback of `go - 1.85` is used (the standard sample-delay duration).

ii.
```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

iii. CONVERSION_NOTES confirm "go - last sample onset = 1.85 s exactly for non-early-lick trials."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone values use the same bin centers as the neural data (both defined relative to the go cue), so they are inherently aligned.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

iii. Same bin grid used for both streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (absolute timestamps of laser on/off). The AI uses the behavioral event timestamps rather than the trials table fields.

ii.
```python
if 'photostim_start_times' in be:
    stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
    stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
else:
    stim_on = np.zeros(0)
    stim_off = np.zeros(0)
```

iii. CONVERSION_NOTES verify "table onset == event time - trial start (exact)".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each photostim event, the AI finds the trial it belongs to (via searchsorted on trial start times), then marks bins as 1 where the bin interval overlaps the laser interval. Non-stimulated trials remain 0.

ii.
```python
photostim_b = np.zeros((n_trials, NBINS), dtype=bool)
if len(stim_on):
    which = np.searchsorted(tstart, stim_on, side='right') - 1
    for s_on, s_off, w in zip(stim_on, stim_off, which):
        if w < 0 or w >= n_trials:
            continue
        if s_on < tstart[w] or s_on > tstop[w]:
            continue
        ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
        photostim_b[w] |= ov
```

iii. The AI uses bin-edge overlap rather than bin-center comparison. CONVERSION_NOTES confirm photostim occupies 10-11 bins (~0.5s), matching the methods.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim intervals and bin edges are both in session-absolute time, so they are directly comparable. The bin edges used for the overlap test are the same go-cue-relative edges used for neural binning.

ii.
```python
ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
```

iii. Same time reference system as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `left_lick_times` and `right_lick_times` in the BehavioralEvents, finding the first lick after the go cue within the answer period (go to min(stop_time, go+1.5)). If no lick occurs, choice is 2 (no lick).

ii.
```python
answer_end = np.minimum(tstop, go + 1.5)
choice = np.full(n_trials, 2, dtype=np.int64)
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
    if len(l) == 0 and len(r) == 0:
        continue
    if len(r) == 0:
        choice[i] = 0
    elif len(l) == 0:
        choice[i] = 1
    else:
        choice[i] = 0 if l[0] < r[0] else 1
```

iii. CONVERSION_NOTES explain: "the animal's actual lick direction is not stored, but it is fully determined by the instructed side and the outcome." However, the AI chose to derive from actual lick times rather than from instruction x outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the AI scans left and right lick timestamps in the answer window, picks the earliest lick direction as choice (0=left, 1=right, 2=no lick). The result is a per-trial value repeated across all 80 time bins.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. CONVERSION_NOTES document that lick-derived choice agrees with outcome x instruction for 99.67% of trials after restricting to the 1.5s answer period.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', or 'hit'.

ii.
```python
outcome_str = np.asarray(df['outcome'].values, dtype=object)
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]], dtype=np.int64)
```

iii. No derivation needed; the trials table stores the outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: ignore=0, miss=1, hit=2. Per-trial value repeated across all 80 bins.

ii.
```python
outputs[:, 1, :] = outcome[:, None]
```

iii. Same three categories as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early_str = np.asarray(df['early_lick'].values, dtype=object)
early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]], dtype=np.int64)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String-to-integer mapping: 'no early'=0, 'early'=1. Per-trial value repeated across all 80 bins.

ii.
```python
outputs[:, 2, :] = early[:, None]
```

iii. Binary encoding as specified.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, columns: (x, y, likelihood) with timestamps at ~300 Hz.

ii.
```python
ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
vts = np.asarray(ts_obj.timestamps[:], dtype=float)
vdata = np.asarray(ts_obj.data[:], dtype=float)
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. CONVERSION_NOTES: "This is the only tongue measurement in the file."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Steps:
1. Filter frames by DLC likelihood > 0.9 (LIKELIHOOD_THRESH)
2. Bin visible tongue-y values into 50ms bins using cumulative sums, computing per-bin means
3. Compute 40th and 60th percentiles of all visible bin means across all trial windows in the session
4. Classify each bin: 0 (<p40), 1 (p40 to p60 inclusive), 2 (>p60), 3 (not visible)

ii.
```python
LIKELIHOOD_THRESH = 0.9
visible = vdata[:, 2] > LIKELIHOOD_THRESH
ysum = np.concatenate([[0.0], np.cumsum(np.where(visible, tongue_y, 0.0))])
vcnt = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
vidx = np.searchsorted(vts, edges_clipped)
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
n_y = vcnt[vidx[:, 1:]] - vcnt[vidx[:, :-1]]
ybin = np.where(n_y > 0, s_y / np.maximum(n_y, 1), np.nan)
vis_vals = ybin[np.isfinite(ybin)]
p40, p60 = np.percentile(vis_vals, [40, 60])
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)
if np.isfinite(p40):
    fin = np.isfinite(ybin)
    tongue_cls[fin & (ybin < p40)] = 0
    tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
    tongue_cls[fin & (ybin > p60)] = 2
```

iii. CONVERSION_NOTES explain the bimodal likelihood distribution (threshold 0.5 vs 0.9 gives nearly identical results) and that percentiles on bin means ensures the class split matches the specified 40/20/40 among visible bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using comparison operators against p40 and p60:
- Class 0: y < p40
- Class 1: p40 <= y <= p60
- Class 2: y > p60
- Class 3: no visible frame (default)

ii.
```python
tongue_cls[fin & (ybin < p40)] = 0
tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
tongue_cls[fin & (ybin > p60)] = 2
```

iii. The AI uses explicit comparisons rather than `np.digitize`. At the p60 boundary, values equal to p60 are assigned class 1 (middle) rather than class 2.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events. Per-trial frame ranges are found by searchsorted on camera timestamps at the clipped bin edges (bin edges clipped to trial boundaries). The cumulative sum approach bins all frames efficiently.

ii.
```python
vidx = np.searchsorted(vts, edges_clipped)
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
```

iii. Same time reference and bin grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Session with no good units** (1 session: classification is NaN): dropped entirely.
- **Trials without ephys coverage**: detected via per-unit obs_intervals intersection; 1,060 trials dropped across 8 sessions.
- **Trials with zero spikes**: 2 additional trials dropped post-binning.
- **Missing tone onset**: fallback to `go - 1.85` (standard timing). 0 fallbacks needed in practice.
- **Tongue frames with low likelihood**: excluded from bin means via cumulative sum masking; bins with no visible frame get class 3.
- **Sessions without photostim events**: handled gracefully (empty array).
- **Bins extending beyond trial boundaries**: bin edges clipped to trial interval.

ii.
```python
if good.sum() == 0:
    return None
# ...
has_spikes = spikes_in_trial > 0
if n_trials_nospike:
    trial_idx = trial_idx[has_spikes]
# ...
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
```

iii. Documented in CONVERSION_NOTES Steps 6 and 10.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files dominates (~1.4s per session). Within a session, the spike_times buffer read and per-unit searchsorted loop are the main costs. Full conversion takes ~247s with 24 workers. Pickling the 11.9 GB result takes ~40s additional.

ii. N/A (timing is printed during execution)

iii. CONVERSION_NOTES Step 7: "The largest sessions have ~2.3x more units and ~2x more trials."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain:
1. Per-unit loop for spike binning (one searchsorted per unit, but all trials vectorized)
2. Per-photostim-event loop for marking bins
The per-unit loop cannot be fully vectorized due to ragged spike time arrays.

ii.
```python
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
# ...
for s_on, s_off, w in zip(stim_on, stim_off, which):
    ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
    photostim_b[w] |= ov
```

iii. The per-trial dimension is already vectorized within each unit; the per-unit loop is inherent to the ragged storage.

## 10-c. What processing does the code repeat multiple times?

i. No processing is repeated. Each NWB file is opened once and all quantities computed in a single pass. The bin grid is defined once at module level.

ii. N/A

iii. The tongue percentiles are per-session (computed within the same pass), so no second pass is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes several diagnostic quantities that go into metadata but aren't used by the decoder:
- `edges_clipped` (bin edges clipped to trial boundaries) - extra computation vs raw edges
- `observed` (boolean mask of fully-observed bins)
- `frac_fully_observed`, `frac_bins_observed`
- Various counters and timing info stored in session metadata
- Choice derived from lick times (a per-trial loop) rather than the simpler outcome x instruction derivation

ii.
```python
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
observed = (edges[:, 1:] <= tstop[:, None]) & (edges[:, :-1] >= tstart[:, None])
# ...
'frac_fully_observed': float(np.mean(observed.all(axis=1))),
```

iii. These are diagnostic/metadata fields that don't affect the decoder training data but add some computational overhead.
