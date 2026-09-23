# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all NWB files under `/app/data/sub-*/` by walking the directory tree, opens each with `pynwb.NWBHDF5IO`, and processes them one at a time (parallelised across sessions with `multiprocessing.Pool`). Trials, units, and behavioral events are read from within each NWB file.

ii.
```python
def list_session_files(data_dir=DATA_DIR):
    files = []
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.nwb'):
                files.append(os.path.join(d, f))
    return files
```

```python
io = NWBHDF5IO(path, 'r', load_namespaces=True)
nwb = io.read()
```

iii. The AI chose `pynwb` (as required by the instructions) and enumerated files by walking the directory tree rather than a glob. Multiprocessing (16 workers by default) was used to speed up the conversion.

## 1-b. How are the data split into subjects?

i. Each NWB file records its subject in `nwb.subject.subject_id`. The AI reads this for every session and builds a sorted unique list of subjects with an index mapping for each session.

ii.
```python
subject = str(nwb.subject.subject_id)
```

```python
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. `subject_id` is the canonical identifier in the NWB file. The AI uses the numeric ID (e.g. `440956`) rather than the mouse name (e.g. `SC015`).

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. No further grouping or splitting is needed. Sessions are identified by `nwb.identifier`.

ii.
```python
session_id = nwb.identifier
```

iii. The DANDI dataset stores one session per NWB file, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`), one row per behavioral trial. The number of go-cue events is checked against the trial count as validation.

ii.
```python
trials = nwb.trials
n_trials_raw = len(trials)
...
go_times = _events(nwb, 'go_start_times')
if len(go_times) != n_trials_raw:
    raise RuntimeError(...)
```

iii. The trials table provides a well-defined trial structure with one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies four trial filters: (1) restrict to trials covered by `obs_intervals` (ephys coverage), (2) remove `auto_water` and `free_water` trials (following the reference code's `get_regular_trial_mask`), (3) remove trials not annotated as "good" in `is_good_trials` for every retained unit (drift QC), and (4) remove trials where no single good unit fired any spike. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
# (a) reference get_regular_trial_mask: no auto-water, no free-water trials
keep = (~auto_water[obs_trial]) & (~free_water[obs_trial])
# (b) recording-stability annotation
is_good_trials = np.asarray(units['is_good_trials'][:])[good]
keep &= is_good_trials.all(axis=0)
...
# trials with no spikes at all
nonempty = fr.sum(axis=(0, 2)) > 0
if not nonempty.all():
    fr = fr[:, nonempty, :]
```

iii. The AI justifies `auto_water` and `free_water` removal from the reference code's `get_regular_trial_mask`. The `is_good_trials` filter is from the QC white paper's drift annotation. The no-spike filter catches trials where the recording stopped mid-trial. Early-lick, ignore, and photostim trials are deliberately kept because they are required decoder outputs/inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times in absolute session seconds) for units with `classification == 'good'`, aligned to go-cue times from `BehavioralEvents/go_start_times`.

ii.
```python
sv = units['spike_times']
ends = np.asarray(sv.data[:])
starts = np.concatenate([[0], ends[:-1]])
flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)
```

iii. `spike_times` is the only neural representation in the NWB file.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins and converted to firing rates in Hz (spike count / bin width). The `bin_spikes_rate` function uses vectorized `np.searchsorted` across all trials simultaneously.

ii.
```python
def bin_spikes_rate(spike_times, go_times):
    edges = go_times[:, None] + BIN_EDGES[None, :]
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
    idx = idx.reshape(edges.shape)
    counts = np.diff(idx, axis=1)
    return (counts / BIN_SIZE).astype(np.float32)
```

iii. The AI matches the reference code's `sliding_histogram(rate=True)` approach of computing firing rates in Hz, adapted to the task-specified 50ms bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No firing-rate thresholds are applied. Sessions with no good units are dropped (1 session).

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
if len(good) == 0:
    io.close()
    return None
```

iii. `classification` is the QC classifier verdict from the white paper. The method paper's 2 Hz firing-rate filter was specific to R-squared analyses and is not applied here, which the AI explicitly justifies.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are defined relative to the go cue and added to each trial's absolute go-cue time. Spikes are then binned against these absolute edges using `np.searchsorted`.

ii.
```python
edges = go_times[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(spike_times, edges.ravel(), side='left')
```

iii. All NWB timestamps share a single session-absolute clock, so alignment requires only adding the relative bin edges to the go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to the go cue, giving 80 time bins per trial. No temporal rebinning or smoothing is applied.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The window and 50ms bin width are specified by the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onset timestamps) and the go-cue time. The tone used for a trial is the last `sample_start_time` before the go cue.

ii.
```python
sample_start = _events(nwb, 'sample_start_times')
tone_pos = np.searchsorted(sample_start, go, side='right') - 1
tone_time = sample_start[tone_pos]
```

iii. Early-lick trials replay the sample epoch, producing multiple tone onsets per trial. The last one before the go cue is the one that actually precedes the delay/go sequence.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the value is the bin center (relative to go cue) plus the go-to-tone offset: `bin_center + (go - tone)`. This gives a continuous ramp in seconds.

ii.
```python
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. The time-from-tone is a continuous, time-varying input as required by the task specification.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin grid defined relative to the go cue. The bin centers used for the time-from-tone calculation are the same centers that define the neural firing-rate bins.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. Sharing the same bin grid guarantees alignment between neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` — the actual laser event timestamps.

ii.
```python
try:
    ps_start = _events(nwb, 'photostim_start_times')
    ps_stop = _events(nwb, 'photostim_stop_times')
except KeyError:
    ps_start = ps_stop = np.zeros(0)
```

iii. The AI uses the event-based photostim timestamps rather than the trials-table columns (`photostim_onset`, `photostim_duration`). Both encode the same information but from different NWB containers. The AI also cross-validated these against the trials-table columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is marked 1 if the overlap between the photostim interval and the bin exceeds 1ms (`OVERLAP_TOL`), 0 otherwise. This produces a binary time-varying input.

ii.
```python
OVERLAP_TOL = 1e-3
...
def interval_overlap_bins(starts, stops, go_time):
    ...
    ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
    on[ov > OVERLAP_TOL] = 1.0
    return on
```

iii. The 1ms overlap tolerance was introduced to handle the fact that photoinhibition ends at the go cue to within +/-0.5ms. Without the tolerance, the first post-go bin would be incorrectly marked as stimulated on about half of the stimulated trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim start/stop times are converted to be relative to the go cue, then compared against the same bin edges used for the neural data.

ii.
```python
for j, g in enumerate(go):
    m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
    if m.any():
        photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```

iii. Using the same go-cue-relative bin grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trials.trial_instruction` (left/right) and `trials.outcome` (hit/miss/ignore), since the actual lick direction is not stored directly.

ii.
```python
instr_k = instruction[keep_idx]
out_k = outcome[keep_idx]
choice = np.full(n_trials, CHOICE_NOLICK, dtype=np.int64)
hit = out_k == 'hit'
miss = out_k == 'miss'
choice[hit & (instr_k == 'left')] = CHOICE_LEFT
choice[hit & (instr_k == 'right')] = CHOICE_RIGHT
choice[miss & (instr_k == 'left')] = CHOICE_RIGHT
choice[miss & (instr_k == 'right')] = CHOICE_LEFT
```

iii. Hit means the animal licked the instructed side; miss means it licked the opposite side; ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Per-trial values are broadcast across all 80 time bins.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
...
outputs = [np.stack([np.full(NBINS, choice[j]),
                     np.full(NBINS, outcome_code[j]),
                     np.full(NBINS, early_code[j]),
                     tongue_class[j]]).astype(np.int64)
           for j in range(n_trials)]
```

iii. Broadcasting per-trial values across time bins allows all four outputs to share a single `(n_output, n_timepoints)` array shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = np.asarray(trials['outcome'][:])
...
outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
```

iii. The trials table stores the outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Broadcast across all 80 time bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
```

iii. Straightforward categorical encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'early'` and `'no early'`.

ii.
```python
early_lick = np.asarray(trials['early_lick'][:])
...
early_code = (early_k == 'early').astype(np.int64)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no=0, yes=1. Broadcast across all 80 time bins.

ii.
```python
early_code = (early_k == 'early').astype(np.int64)
```

iii. Simple binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data (x, y, likelihood) with matching timestamps. Column index 1 (y) is the value; the likelihood column determines visibility.

ii.
```python
ts_obj = bts.time_series['Camera0_side_TongueTracking']
frame_t = np.asarray(ts_obj.timestamps[:], dtype=np.float64)
data = np.asarray(ts_obj.data[:, 1:3], dtype=np.float64)   # y, likelihood
y = data[:, 0]
visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
```

iii. This is the only tongue measurement in the NWB file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood <= 0.9` are marked invisible. Visible frames are assigned to their parent trial (using `searchsorted` on trial start times), then only frames from retained trials within the [-2.5, 1.5] window are kept. These are averaged into 50ms bins via `np.bincount`. Bins with no visible frames get NaN.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
...
visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1
...
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
cnts = np.bincount(flat, minlength=n_cells)
mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
```

iii. The AI uses a likelihood threshold of 0.9 (vs. the reference's 0.5), but notes the distribution is strongly bimodal so any threshold in [0.1, 0.99] gives effectively the same result. The AI assigns frames to trials before binning, preventing cross-trial contamination.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed over all visible bin-mean values across the retained trials of the session. Then: values below the 40th percentile get class 0, values between 40th and 60th (inclusive) get class 1, values above the 60th percentile get class 2, and bins with no visible frame get class 3.

ii.
```python
def discretise_tongue(tongue_y):
    cls = np.full(tongue_y.shape, 3, dtype=np.int64)
    vis = ~np.isnan(tongue_y)
    vals = tongue_y[vis]
    p_lo = np.percentile(vals, TONGUE_LOW_PCT)
    p_hi = np.percentile(vals, TONGUE_HIGH_PCT)
    c = np.ones(v.shape, dtype=np.int64)
    c[v < p_lo] = 0
    c[v > p_hi] = 2
    cls[vis] = c
    return cls, (float(p_lo), float(p_hi))
```

iii. The 40/60 percentile split and per-session scope follow the task instructions. The "not visible" class is the fourth category required by the spec.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes. Frame-to-bin assignment uses the same go-cue-relative bin grid as the neural data. Frames are mapped to bins by computing `floor((frame_time - (go + OFF_START)) / BIN_SIZE)`.

ii.
```python
rel = frame_t[vis] - go[pos]
inwin = (rel >= OFF_START) & (rel < OFF_END)
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. Using the same bin grid guarantees bin k of the tongue output covers the same interval as bin k of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) One session with NaN `classification` for all units (never quality-controlled) is dropped because it has 0 good units. (2) Trials without ephys coverage are excluded via `obs_intervals`. (3) `auto_water` and `free_water` trials are excluded. (4) Trials flagged as not good in `is_good_trials` are excluded. (5) Trials with zero spikes across all units are dropped (2 total). (6) Tongue frames with low likelihood are set to NaN; bins with no visible frames become class 3. (7) Missing CCF x-coordinates for hemisphere determination fall back to the insertion target.

ii.
```python
if len(good) == 0:
    io.close()
    return None
...
nonempty = fr.sum(axis=(0, 2)) > 0
if not nonempty.all():
    fr = fr[:, nonempty, :]
...
missing = np.isnan(elec_x)
if missing.any():
    hemisphere[missing] = np.array([t.split(' ')[0] for t in probe_target])[missing]
```

iii. The AI handles each type of missing data with an appropriate strategy: drop where nothing was recorded, use explicit categories where the measurement is legitimately absent.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and the per-unit searchsorted loop for spike binning dominate. The full conversion takes about 40s with 16-way multiprocessing (26s for conversion + 13s for pickling the 11.8 GB result).

ii. N/A

iii. The AI used multiprocessing to parallelize across sessions, achieving a ~10x speedup over serial processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in spike binning runs one `np.searchsorted` per unit (but all trials are vectorized). This cannot be fully vectorized because each unit has a different number of spikes (ragged storage).

ii.
```python
for i, u in enumerate(good):
    st = flat_spikes[starts[u]:ends[u]]
    fr[i] = bin_spikes_rate(st, go)
```

iii. The per-trial dimension is already vectorized within `bin_spikes_rate`. The tongue binning is also vectorized using `np.bincount`.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed. Each NWB file is opened once and every quantity is computed once. The bin grid is defined once at module level.

ii. N/A

iii. The conversion is a single pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes hemisphere information (`ccf_x >= 5700` rule) and stores it in metadata but not in the main `brain_region_idx`. The reference code's 14 coarse region mapping (via `ccf_regions.py`) is more complex than needed. Debug information is collected when `--show-processing` is used. Detailed per-session timing and trial-count metadata is computed and stored.

ii.
```python
elec_x = np.asarray(nwb.electrodes['x'][:], dtype=np.float64)[elec_idx]
hemisphere = np.where(elec_x >= ML_MIDLINE_UM, 'left', 'right')
...
'neuron_hemisphere': [[int(v) for v in r['hemisphere']] for r in results],
```

iii. The hemisphere computation adds metadata richness but is not used by the decoder. The complex CCF annotation mapping via `ccf_regions.py` replicates the reference code's region grouping but is more work than the simple approach.
