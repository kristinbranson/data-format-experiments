# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in `/app/data/sub-*/`. Each `.nwb` file corresponds to one session and is opened using `h5py` (not `pynwb`). All sessions are discovered via a sorted glob pattern. Each session file is processed independently by `convert_session()`, which reads trials, units, behavioral events, and video tracking data directly from the HDF5 structure.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with h5py.File(filepath, 'r') as f:
    subject = _decode(f['general/subject/subject_id'][()])
    identifier = _decode(f['identifier'][()])
    tr = f['intervals/trials']
    start_time = tr['start_time'][:]
    # ...
    be = f['acquisition/BehavioralEvents']
    go_all = be['go_start_times']['timestamps'][:]
```

iii. The AI chose `h5py` instead of `pynwb` for direct HDF5 access to the NWB files. The CONVERSION_NOTES.md documents this as part of the data structure exploration in Step 2. The AI verified 174 NWB files across 28 subjects.

## 1-b. How are the data split into subjects?

i. Each NWB file contains a `general/subject/subject_id` field (a numeric string like `'440956'`). The AI reads this for each session, collects unique subjects, sorts them, and builds a mapping from subject to index.

ii.
```python
subject = _decode(f['general/subject/subject_id'][()])
# ...
subjects = sorted({s['subject'] for s in sessions})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[s['subject']] for s in sessions]),
```

iii. The AI identified 28 unique subjects in the CONVERSION_NOTES.md, consistent with the papers.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by `f['identifier']`. The sorted glob of `.nwb` files defines the session order. Session metadata is recorded in the output.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
identifier = _decode(f['identifier'][()])
```

iii. The AI verified that each NWB file corresponds to a single session and that there are 174 files (173 after dropping the session with no good units).

## 1-d. How are the data split into trials?

i. Trials are read from `intervals/trials` in the NWB file. The AI reads all trial-related columns (`start_time`, `stop_time`, `outcome`, `early_lick`, `trial_instruction`, etc.) as arrays. Go cue times come from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
# ...
go_all = be['go_start_times']['timestamps'][:]
```

iii. The AI verified that there is exactly one go cue event per trial in all sessions. It also includes a fallback for cases where go events might not align 1:1 with trials (using `searchsorted`), though this case does not occur in the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on several criteria: (1) `auto_water == 0` and `free_water == 0` (removes water-delivery trials), (2) `np.isfinite(go)` (go cue must exist), (3) trials must be within `obs_intervals` of the ephys recording (checked across all good units via intersection), and (4) trials with zero spikes across all QC-passing neurons are dropped post-hoc. A session is dropped if fewer than 2 trials survive.

ii.
```python
keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed
trial_idx = np.where(keep)[0]
if len(trial_idx) < 2:
    return None
# ...
# Drop trials with zero spikes
nonempty = fr.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    fr = fr[:, nonempty, :]
    trial_idx = trial_idx[nonempty]
```

The `observed` mask is built by intersecting obs_intervals across all good units:
```python
observed = np.ones(ntrials_all, dtype=bool)
for ui in np.where(good_pre)[0]:
    iv = u_pre['obs_intervals'][oi_starts[ui]:oii[ui]]
    if len(iv) == ntrials_all:
        continue
    m = np.zeros(ntrials_all, dtype=bool)
    k = np.searchsorted(start_time, iv[:, 0] + 1e-6) - 1
    k = k[(k >= 0) & (k < ntrials_all)]
    m[k] = True
    observed &= m
```

iii. The AI documents in CONVERSION_NOTES.md that `auto_water` and `free_water` trials are removed following the reference `get_regular_trial_mask`. Early-lick, no-response, and photostim trials are kept because they are required decoder variables. The obs_intervals check handles 8 sessions where the ephys recording covers only a subset of behavioural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (concatenated spike times) and `units/spike_times_index` (end indices per unit). Only units with `classification == 'good'` and non-empty `anno_name` are used. Go cue times are used for alignment.

ii.
```python
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
# ...
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
```

iii. The AI identified spike times as the only neural representation in the NWB files, consistent with the reference code's use of spike times to compute firing rates.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, +1.5] s relative to the go cue. For each unit, bin edges for all trials are flattened, `np.searchsorted` gives the running count at each edge, and `np.diff` gives counts per bin. Counts are divided by bin width (0.05 s) to get firing rates in Hz.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
out = np.zeros((len(unit_idx), ntrials, N_BINS), dtype=np.float32)
starts = np.concatenate([[0], spike_index[:-1]])
for k, u in enumerate(unit_idx):
    st = spike_times[starts[u]:spike_index[u]]
    if st.size == 0:
        continue
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    out[k] = np.diff(pos, axis=1)
out /= BIN_SIZE       # spike count -> firing rate (Hz)
```

iii. The AI documents this matches the reference code's `sliding_histogram(rate=True)` which returns `binSpikes / bin_width`. The 50 ms bin width is specified by the instructions (rather than the reference's 40 ms sliding window).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` AND non-empty `anno_name` (CCF annotation) are kept. A session with no such units is dropped. No individual quality metric thresholds are applied.

ii.
```python
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
if len(unit_idx) == 0:
    return None
```

iii. The AI notes in CONVERSION_NOTES.md that the reference code requires both the classifier QC label and a histology (CCF) annotation. In practice all `good` units have a CCF annotation, so this additional check doesn't change the results. The AI reports 69,453 good units across 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All times in the NWB file are on the same session-absolute clock. The bin edges relative to the go cue are added to each trial's go-cue time to produce absolute time windows, and spikes are binned against those edges directly. No resampling or interpolation is needed.

ii.
```python
go_keep = go[trial_idx]
# ...
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The AI confirms alignment to the go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins total spanning [-2.5 s, +1.5 s] relative to the go cue. The bin grid is defined once and reused for all trials and sessions.

ii.
```python
BIN_SIZE = 0.05           # s
OFF_START = -2.5          # s
OFF_END = 1.5             # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```

iii. The 50 ms bin width and [-2.5, +1.5] s window are as specified in the instructions. No rebinning is applied since firing rates are computed directly from spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone onset times) and the go cue times. The tone for each trial is the last sample-epoch onset before the trial's go cue.

ii.
```python
sample_all = be['sample_start_times']['timestamps'][:]
# ...
j = np.searchsorted(sample_all, go, side='right') - 1
tone = np.where(j >= 0, sample_all[np.clip(j, 0, len(sample_all) - 1)], np.nan)
# fall back to the nominal 1.85 s if missing
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)
```

iii. The AI documents that early licks trigger a replay of the sample epoch, so the last sample onset before the go cue is the relevant one. A fallback to the nominal 1.85 s gap is used if the tone onset is missing or invalid.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset is computed as: `(go_time - tone_time) + bin_center`. This gives a continuous ramp with 0 at the tone onset, increasing monotonically with 0.05 s steps.

ii.
```python
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
```

iii. The AI verifies that input 0 crosses zero at the tone onset (median -1.85 s relative to the go cue) in the processing plots.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same bin centers as the neural data, since both use `BIN_CENTERS` which are defined relative to the go cue. Each trial's tone-to-go offset shifts the ramp, but the bins are identical.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
```

iii. Same bin grid used for neural data and inputs ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `intervals/trials/photostim_onset`, `photostim_duration`, `photostim_power`, and `start_time`. The onset is relative to trial start (stored as string, converted to float). Power is used to determine whether stimulation occurred.

ii.
```python
ps_onset = np.array([_tofloat(x) for x in tr['photostim_onset'][:]])
ps_dur = np.array([_tofloat(x, 0.0) for x in tr['photostim_duration'][:]])
ps_power = np.array([_tofloat(x, 0.0) for x in tr['photostim_power'][:]])
```

iii. The AI notes photostim values are stored as strings (`N/A` when no stimulation), converted to floats with a fallback to NaN.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary per-bin signal: 1 if any part of the bin overlaps with the stimulation window, 0 otherwise. Stimulation is detected when `ps_power > 0` and both onset and duration are finite. The stim onset is converted to go-cue-relative time, and bins are marked as overlapping if `BIN_EDGES[1:] > s0` and `BIN_EDGES[:-1] < s1`.

ii.
```python
has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)
for k, i in enumerate(trial_idx):
    if not has_stim[i]:
        continue
    s0 = start_time[i] + ps_onset[i] - go[i]     # relative to the go cue
    s1 = s0 + ps_dur[i]
    overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
    photostim[k, overlap] = 1.0
```

iii. The AI uses bin-edge overlap (any overlap marks the bin as 1) rather than bin-center containment. The AI also checks `ps_power > 0` as the condition for stimulation being present, rather than checking `photostim_onset != 'N/A'`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stim onset and offset are expressed relative to the go cue (same reference point as the bin edges), so the overlap check is in the same coordinate system as the neural bins.

ii.
```python
s0 = start_time[i] + ps_onset[i] - go[i]     # relative to the go cue
s1 = s0 + ps_dur[i]
overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
```

iii. Alignment is guaranteed by using the go-cue-relative coordinate system for both neural bins and stimulation windows.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/outcome` and `intervals/trials/trial_instruction`. Choice is not stored directly. Hit means the animal licked the instructed side, miss means the opposite, and ignore means no lick.

ii.
```python
oc = outcome[trial_idx]
ins = instruction[trial_idx]
licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')
choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
```

iii. The AI verified this derivation against the actual first lick after the go cue from `left/right_lick_times` and reports 100%/99.8% agreement in test sessions.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left), 1 (right), 2 (no lick). It is a per-trial value repeated across all 80 time bins.

ii.
```python
choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
# ...
outputs = [np.stack([np.full(N_BINS, choice[i]), ...]).astype(np.int64) for i in range(ntr)]
```

iii. The coding follows the instruction specification (left, right, no lick).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which contains strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
# ...
outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2)).astype(np.int64)
```

iii. The three outcome categories match the instruction specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. This is a per-trial value repeated across all 80 time bins.

ii.
```python
outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2)).astype(np.int64)
# ...
outputs = [np.stack([..., np.full(N_BINS, outcome_code[i]), ...]) for i in range(ntr)]
```

iii. Straightforward mapping, consistent with the instruction specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, which contains strings like `'no early'` and `'early'`.

ii.
```python
early = np.array([_decode(x) for x in tr['early_lick'][:]])
# ...
early_code = np.array([0 if e == 'no early' else 1 for e in el_tr], dtype=np.int64)
```

iii. The AI notes that the early lick flag marks licking during the sample or delay epoch, before the go cue.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) and 1 (early). Any value that is not `'no early'` is treated as early. Per-trial value repeated across 80 bins.

ii.
```python
early_code = np.array([0 if e == 'no early' else 1 for e in el_tr], dtype=np.int64)
```

iii. The reference uses a strict dictionary `{'no early': 0, 'early': 1}`, while the AI uses `0 if e == 'no early' else 1`, which would map any non-standard value to 1. In practice the data only contains these two values.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which provides `(n_frames, 3)` data: tongue_x, tongue_y, tongue_likelihood, with timestamps. Column 1 is tongue_y, column 2 is the DeepLabCut likelihood.

ii.
```python
tt_ds = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vdata = tt_ds['data'][:]
vts = tt_ds['timestamps'][:]
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH   # TONGUE_LIKELIHOOD_THRESH = 0.9
yv = vdata[:, 1]
```

iii. The AI identifies this as the only tongue measurement in the files, present in all 174 sessions at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a likelihood threshold of 0.9 to determine tongue visibility. The 40th and 60th percentiles are computed over all visible raw frames in the session (not binned means). For each bin in each trial, the LAST visible video frame within the bin determines the tongue y value. Bins with no visible frame get class 3 (not visible). Visible values are discretized into 3 classes using the percentile thresholds: 0 if y < 40th pct, 1 if between 40th and 60th, 2 if y > 60th.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
# ...
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
yv = vdata[:, 1]
if np.any(vis):
    thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
# ...
# In tongue_class_per_bin:
ybin = np.full(N_BINS, np.nan)
ybin[idx] = yy     # frames are ordered, so the last one in a bin wins
good = ~np.isnan(ybin)
c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
cls[i, good] = c
```

iii. The AI justifies the 0.9 likelihood threshold based on DeepLabCut conventions and notes the bimodal distribution (most frames either very low or very high likelihood). The "last frame" approach is documented as matching the reference code's `align_markers_between_lims`. The percentile computation is done on raw visible frames rather than on session bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three visible classes plus a "not visible" class:
- 0: y < 40th percentile of session visible frames
- 1: 40th to 60th percentile
- 2: y > 60th percentile
- 3: not visible (no frame with likelihood > 0.9 in the bin)

ii.
```python
c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
cls[i, good] = c
```

iii. Percentiles are computed per-session as specified in the instructions. The boundary conditions use strict inequalities (`<` and `>`), meaning values exactly at the 40th or 60th percentile fall into class 1.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. Each trial's frame range is found by `searchsorted` on the camera timestamps at `go + OFF_START` and `go + OFF_END`. Frames are assigned to bins by their offset from the trial start, using the same bin grid as the neural data.

ii.
```python
lo = np.searchsorted(ts, t0, side='left')
hi = np.searchsorted(ts, t1, side='left')
# ...
idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)
```

iii. Same bin grid as neural data ensures alignment. The AI verifies this with processing plots showing the tongue class changing from "not visible" to visible categories right after the go cue.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **`anno_name` as NaN**: The `_decode()` helper returns empty string for non-string values, so units without annotations are excluded by the `anno != ''` filter.
- **Trials outside ephys recording**: `obs_intervals` intersection across all good units identifies which trials have spike data; others are excluded.
- **Zero-spike trials**: Trials with no spikes across all neurons are dropped after binning (catches recording stopped mid-trial).
- **Missing go cue**: `np.isfinite(go)` check (no instances in data, but handled).
- **Missing tone onset**: Fallback to nominal 1.85 s gap (`tone = go - 1.85`).
- **Sessions with < 2 usable trials**: Dropped entirely.

ii.
```python
def _decode(x):
    if isinstance(x, bytes): return x.decode()
    if isinstance(x, str): return x
    return ''

# tone fallback
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)

# zero-spike trials
nonempty = fr.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    fr = fr[:, nonempty, :]
```

iii. The AI documents each edge case in CONVERSION_NOTES.md Step 10 Check 5, and describes specific instances encountered in the data.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports the main bottlenecks as: reading spike times from HDF5 (~20 s total), binning spikes (~40 s total), and tongue discretisation (~15 s total). With 12-process multiprocessing, the full conversion takes ~43 s plus pickling time.

ii.
```python
timings['read_spikes'] = time.time() - t0
# ...
timings['bin_spikes'] = time.time() - t0
# ...
timings['tongue'] = time.time() - t0
```

iii. The AI includes timing instrumentation and documents performance in CONVERSION_NOTES.md Step 7.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) the per-unit loop in `bin_spikes` that runs one `searchsorted` per unit (cannot be collapsed because each unit has different spike counts), and (2) the per-trial loop in `tongue_class_per_bin` that processes one trial's frames at a time.

ii.
```python
for k, u in enumerate(unit_idx):
    st = spike_times[starts[u]:spike_index[u]]
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    out[k] = np.diff(pos, axis=1)
# ...
for i in range(ntrials):
    # per-trial tongue processing
```

iii. The AI notes that the per-unit loop is inherent to the ragged spike time storage, and that the per-trial tongue loop is not a performance bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The classification and anno_name arrays are read twice: once before trial filtering (`classification_pre`, `anno_pre`, `good_pre`) for the obs_intervals check, and once after (`classification`, `anno`, `good`) for the actual unit selection.

ii.
```python
# First read (for obs_intervals)
classification_pre = np.array([_decode(x) for x in u_pre['classification'][:]])
anno_pre = np.array([_decode(x) for x in u_pre['anno_name'][:]])
good_pre = (classification_pre == 'good') & (anno_pre != '')
# ...
# Second read (for unit selection)
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
```

iii. Both reads reference the same `f['units']` HDF5 group, so the second read is redundant but harmless.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores several metadata fields that are not used by the decoder: `hemisphere` per unit, `n_units_total` and `n_trials_total` per session, `tongue_thresholds`, and `trial_idx` (the original trial indices). The `classify_region` function performs elaborate keyword-based region classification with AP coordinate calculations, producing 15 fixed region categories. The electrode coordinates (`x`, `y`, `z`) are read and transformed but only used for ALM classification. The `photostim_power` array is read but only used as a boolean check.

ii.
```python
eidx = u['electrodes'][:]
el = f['general/extracellular_ephys/electrodes']
ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
ux, uy, uz = ex[eidx], ey[eidx], ez[eidx]
ap = CCF_AP_BREGMA - uz
hemi = np.where(ux >= CCF_ML_MIDLINE, 'left', 'right')
```

iii. The extra metadata could be useful for debugging but adds processing time and output size.
