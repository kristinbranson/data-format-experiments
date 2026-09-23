# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files with a glob over `sub-*/*.nwb` under `/app/data`, sorts them, and processes each with `h5py.File`. Each session's trials, units, and behavioral events are read from the HDF5 groups. Multiprocessing (16 workers) is used to process sessions in parallel.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with Pool(min(args.nproc, len(files))) as pool:
    results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
```

```python
with h5py.File(path, 'r') as f:
    sess_id = f['identifier'][()].decode()
    subject = f['general/subject/subject_id'][()].decode()
    # ...
    t = f['intervals/trials']
    trial_start = t['start_time'][:]
```

iii. The AI chose `h5py` instead of `pynwb` for performance. The CONVERSION_NOTES document that 174 NWB files were found across 28 subject folders, matching the dandiset metadata.

## 1-b. How are the data split into subjects?

i. Each NWB file's `general/subject/subject_id` provides the numeric subject ID. At assembly, subjects are the sorted unique set of these IDs, and `subject_idx` maps each session to its index.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
# ...
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The AI's CONVERSION_NOTES confirm 28 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Session identity is `f['identifier']`, e.g. `SC015_20190207_120657_s1`. Sorted file paths give deterministic ordering.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
sess_id = f['identifier'][()].decode()
```

iii. The AI documented 174 NWB files, 173 with at least 1 good unit, matching the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. The number of go cues is checked against the number of trials.

ii.
```python
t = f['intervals/trials']
trial_start = t['start_time'][:]
trial_stop = t['stop_time'][:]
# ...
go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go_all) != ntrials_file:
    print('%s: %d go cues for %d trials, skipping' % (...), flush=True)
    return None
```

iii. The AI treats a mismatch between go cues and trials as grounds for skipping the session, though no such mismatch occurs in the data.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) `auto_water` or `free_water` trials are removed, (2) trials outside the units' `obs_intervals` are removed (checked across ALL good units), (3) trials with zero total spikes in the extracted window are removed. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
keep = ~(auto_water | free_water)
keep &= np.isfinite(go_all)
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
# ...
has_spikes = fr.sum(axis=(0, 2)) > 0
if not has_spikes.all():
    fr = fr[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
```

iii. The AI justifies removing auto/free water trials because reward delivery is not driven by the animal's choice, citing the reference code's `get_regular_trial_mask`. Early-lick, ignore, and photostim trials are kept because they are decoder targets or inputs. The zero-spike check handles edge cases at the end of truncated recordings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times for each unit), filtered by `units/classification` and `units/anno_name`. Go cue times from `BehavioralEvents/go_start_times` provide bin edge anchors.

ii.
```python
spikes = u['spike_times'][:]
s0, s1 = _spike_slices(u['spike_times_index'][:])
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)
```

iii. The AI notes that `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, +1.5] s relative to the go cue. Spike counts per bin are computed via `np.searchsorted` on flattened bin edges, then divided by bin width (0.05 s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spikes(spike_times, starts, stops, unit_ids, edges_abs):
    ntrials = edges_abs.shape[0]
    flat = edges_abs.ravel()
    out = np.empty((len(unit_ids), ntrials, NBINS), dtype=np.float32)
    for i, k in enumerate(unit_ids):
        sp = spike_times[starts[k]:stops[k]]
        idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
        out[i] = np.diff(idx, axis=1) / BIN_SIZE
    return out
```

iii. The AI documents this matches the reference code's `sliding_histogram(..., rate=True)` formula of count/bin_width, with the bin size changed to 50 ms as required by the decoder specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` AND a non-empty `anno_name` (CCF annotation) are kept. Sessions with no such units are dropped. No firing rate threshold is applied.

ii.
```python
classification = u['classification'][:]
anno = _decode(u['anno_name'][:])
good = (classification == b'good')
good &= np.array([a.strip() != '' for a in anno])
unit_ids = np.where(good)[0]
if len(unit_ids) == 0:
    return None
```

iii. The AI justifies the CCF annotation requirement by noting the reference code (`helper_get_neuron_id_area`) also requires histology. The AI explicitly does NOT apply the 2 Hz firing rate filter from the method paper, arguing it was specific to encoding analyses and would remove 33% of good units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are constructed as absolute times by adding the relative bin edges to each trial's go cue time. Spike times are already on the same session-absolute clock, so no additional alignment is needed.

ii.
```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
# ...
idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
```

iii. The AI notes all NWB timestamps share the same session-absolute clock, so alignment only requires knowing the go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins total spanning [-2.5, +1.5] s relative to the go cue. Bin edges are defined once as offsets.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
```

iii. The AI documents this as a deviation from the reference code's 40 ms window / 3.4 ms stride, required by the decoder specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` timestamps in BehavioralEvents, along with `trial_start` times and go cue times. The last sample start before the go cue within each trial is selected as the tone onset.

ii.
```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:] \
    if 'sample_start_times' in f['acquisition/BehavioralEvents'] else np.array([])
tone = tone_onset_times(trial_start, go_all, sample_start)[trial_idx]
```

iii. The AI explains that early licks replay the sample epoch, so there can be multiple `sample_start_times` per trial. The last one before the go cue is the instruction tone the animal responded to.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute bin centers (go cue + relative bin centers) minus the tone onset time gives time from tone onset in seconds. A fallback of 1.85 s (0.65 s sample + 1.2 s delay) before the go cue is used if no sample_start is found.

ii.
```python
def tone_onset_times(trial_start, go, sample_start):
    ntrials = len(go)
    tone = np.full(ntrials, np.nan)
    if len(sample_start):
        idx = np.searchsorted(trial_start, sample_start, side='right') - 1
        for i, s in zip(idx, sample_start):
            if 0 <= i < ntrials and s < go[i]:
                if np.isnan(tone[i]) or s > tone[i]:
                    tone[i] = s
    missing = np.isnan(tone)
    if missing.any():
        tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
    return tone
```

```python
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES state that the fallback was never triggered.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and this input share the same bin grid (80 bins of 50 ms centered on the go cue). The bin centers are used for the time-from-tone computation, the bin edges for the spike counts.

ii.
```python
centers_abs = go[:, None] + BIN_CENTERS[None, :]
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. Using the same go-cue-relative grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Preferentially from `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents` (absolute timestamps). Falls back to `photostim_onset` and `photostim_duration` in the trials table (relative to trial start, stored as strings).

ii.
```python
def photostim_intervals(f, trial_start, go):
    be = f['acquisition/BehavioralEvents']
    if 'photostim_start_times' in be and len(be['photostim_start_times/timestamps']) > 0:
        on = be['photostim_start_times/timestamps'][:]
        off = be['photostim_stop_times/timestamps'][:]
        n = min(len(on), len(off))
        return np.stack([on[:n], off[:n]], axis=1)
    # fallback: reconstruct from the trials table
    t = f['intervals/trials']
    onset = _decode(t['photostim_onset'][:])
    dur = _decode(t['photostim_duration'][:])
    iv = []
    for i in range(len(trial_start)):
        if onset[i] != 'N/A' and dur[i] != 'N/A':
            a = trial_start[i] + float(onset[i])
            iv.append([a, a + float(dur[i])])
    return np.array(iv).reshape(-1, 2)
```

iii. The AI prefers the event timestamps because they are absolute and avoid string parsing. The fallback handles sessions missing the event time series.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is set to 1.0 if the stimulation interval overlaps the bin (i.e., stim start < bin right edge AND stim end > bin left edge), and 0.0 otherwise. This is an overlap test on the bin edges, not a center-in-interval test.

ii.
```python
stim_iv = photostim_intervals(f, trial_start, go_all)
photostim = np.zeros((len(trial_idx), NBINS), dtype=np.float32)
if len(stim_iv):
    for a, b in stim_iv:
        photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. The overlap test means that a bin is "on" if any part of the stimulation falls within the bin, not just its center.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The stim intervals are in absolute session time, and the bin edges (`edges_abs`) are also in absolute session time (go cue + relative edges), so the overlap comparison directly aligns photostim with the neural bins.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
# ...
photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. Same absolute time base ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials/outcome` (hit/miss/ignore) and `trials/trial_instruction` (left/right). A hit means the animal licked the instructed side; a miss means it licked the opposite side; ignore means no lick.

ii.
```python
oc = outcome[trial_idx]
instr = instruction[trial_idx]
choice = np.zeros(len(trial_idx), dtype=np.int64)
licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
licked_right = ((oc == 'hit') & (instr == 'right')) | ((oc == 'miss') & (instr == 'left'))
choice[licked_left] = 1
choice[licked_right] = 2
```

iii. The AI notes this is the same definition as the reference code's `behavior_report` x `task_trial_type`, verified to agree with lick-time-based choice on 99.4% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=no lick, 1=left, 2=right (per the AI's `OUTPUT_VALUES[0]`). It is a per-trial value broadcast across all 80 time bins.

ii.
```python
OUTPUT_VALUES = [
    ['no lick', 'left', 'right'],
    ...
]
# ...
o = np.empty((4, NBINS), dtype=np.int64)
o[0] = choice[i]
```

iii. The AI chose no_lick=0, left=1, right=2 ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', 'hit'.

ii.
```python
outcome = _decode(t['outcome'][:])
oc = outcome[trial_idx]
outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                         default=0).astype(np.int64)
```

iii. The trials table provides outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Per-trial value broadcast across all 80 bins.

ii.
```python
outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                         default=0).astype(np.int64)
# ...
o[1] = outcome_code[i]
```

iii. The ordering matches the instructions (ignore, miss, hit).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early = _decode(t['early_lick'][:])
el_tr = early[trial_idx]
early_code = (el_tr == 'early').astype(np.int64)
```

iii. Directly available in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value broadcast across all 80 bins.

ii.
```python
early_code = (el_tr == 'early').astype(np.int64)
o[2] = early_code[i]
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking` (or `Camera3_side_TongueTracking` as fallback). The data array is (n_frames, 3) = (x, y, likelihood) with matching timestamps.

ii.
```python
bt = f['acquisition/BehavioralTimeSeries']
key = None
for k in ('Camera0_side_TongueTracking', 'Camera3_side_TongueTracking'):
    if k in bt:
        key = k
        break
data = bt[key]['data'][:]
ts = bt[key]['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. The AI handles sessions with a different camera name via the fallback.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.9 are marked not visible. For each 50 ms bin, visible frames are averaged using cumulative sums. Per-session 40th/60th percentiles are computed over all visible bin means across the trial windows, giving class boundaries.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9

visible = (lik > TONGUE_LIKELIHOOD_THRESHOLD) & np.isfinite(y)
yv = np.where(visible, y, 0.0)
cs_y = np.concatenate([[0.0], np.cumsum(yv)])
cs_n = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
idx = np.searchsorted(ts, edges_abs)
lo, hi = idx[:, :-1], idx[:, 1:]
nvis = cs_n[hi] - cs_n[lo]
sumy = cs_y[hi] - cs_y[lo]
mean_y = np.where(nvis > 0, sumy / np.maximum(nvis, 1), np.nan)
```

iii. The AI chose 0.9 as the DLC likelihood threshold, noting the distribution is strongly bimodal (99.8% of frames below 0.1 or above 0.9), so the exact value is irrelevant.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th, 60th) are computed over all visible bin means across the extracted trial windows. Bins are classified as: 0 (< 40th), 1 (40th-60th, inclusive on both bounds), 2 (> 60th), 3 (not visible).

ii.
```python
def discretize_tongue(mean_y, visible):
    cls = np.full(mean_y.shape, 3, dtype=np.int64)
    vals = mean_y[visible]
    if vals.size == 0:
        return cls
    p40, p60 = np.percentile(vals, [40, 60])
    cls[visible & (mean_y < p40)] = 0
    cls[visible & (mean_y >= p40) & (mean_y <= p60)] = 1
    cls[visible & (mean_y > p60)] = 2
    return cls
```

iii. The percentiles are computed over the visible bins of only the trial windows (not the entire session). The class balance among visible bins is reported as 40/20/40, matching the expected 40th/60th split.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same absolute session clock. `np.searchsorted` on the camera timestamps at the bin edges gives the frames within each 50 ms bin. Cumulative sums enable vectorized averaging.

ii.
```python
idx = np.searchsorted(ts, edges_abs)  # (ntrials, NBINS+1)
lo, hi = idx[:, :-1], idx[:, 1:]
```

iii. Same absolute time base as spikes and go cues ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three main cases: (1) Sessions with no good units (classification == 'good' and non-empty anno_name) are skipped (1 session). (2) Trials outside obs_intervals or with all-zero spikes are removed. (3) Tongue bins with no visible frames get class 3 ('not visible'). A fallback tone onset time (1.85 s before go cue) handles the theoretical case of missing sample_start_times. Sessions missing the tongue tracking time series get NaN/not-visible throughout.

ii.
```python
if len(unit_ids) == 0:
    return None
# ...
has_spikes = fr.sum(axis=(0, 2)) > 0
if not has_spikes.all():
    fr = fr[:, has_spikes, :]
# ...
missing = np.isnan(tone)
if missing.any():
    tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
# ...
if key is None:
    return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))
```

iii. The AI's CONVERSION_NOTES document each edge case and its resolution, noting the tone onset fallback was never triggered.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 spike times array and computing firing rates dominates per-session time. The full conversion took ~24 seconds with 16-worker multiprocessing (plus ~21 seconds for pickle writing). Average ~0.5 s per session.

ii. N/A (timing from CONVERSION_NOTES)

iii. The AI implemented multiprocessing to parallelize session processing, reducing wall-clock time by ~10x.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `bin_spikes` iterates over each unit to run `searchsorted` on its spike times. The per-trial loop in `tongue_y_per_bin` is replaced by a vectorized cumulative-sum approach. The per-stim-interval loop in `photostim` construction could potentially be vectorized but handles few intervals per session.

ii.
```python
for i, k in enumerate(unit_ids):
    sp = spike_times[starts[k]:stops[k]]
    idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
    out[i] = np.diff(idx, axis=1) / BIN_SIZE
```

iii. The per-unit loop cannot be fully vectorized because each unit has a different number of spikes (ragged arrays). The trial dimension is already vectorized within each unit.

## 10-c. What processing does the code repeat multiple times?

i. The AI's code reads and processes each NWB file exactly once. However, when `make_plots=True`, the `plot_processing` function reopens the NWB file to read raw spike times again for plotting, duplicating some I/O.

ii.
```python
def plot_processing(res, path, go, ...):
    with h5py.File(path, 'r') as f:
        u = f['units']
        s0, s1 = _spike_slices(u['spike_times_index'][:])
        # re-reads spike data for plotting
```

iii. This duplication only occurs for up to 2 sessions in plotting mode, not during the main conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `session_start_time`, `n_units_total` (total units in file), `n_trials_file`, `trial_idx`, and `timing` info per session in the results dict. Some of this metadata (like `n_units_in_file`, `n_trials_in_file`) is stored in the final pickle's `session_info` but is not used by the decoder. The electrode coordinates (`ex`, `ml`) are computed for all units but only used for the good units' hemisphere assignment.

ii.
```python
session_start = f['session_start_time'][()].decode()
el = f['general/extracellular_ephys/electrodes']
ex = el['x'][:]
e_of_unit = u['electrodes'][:][np.concatenate([[0], u['electrodes_index'][:][:-1]])]
ml = ex[e_of_unit]  # computed for all units, used only for good ones
```

iii. The extra metadata is small overhead. The electrode coordinate computation for all units rather than just good units is a minor inefficiency.
