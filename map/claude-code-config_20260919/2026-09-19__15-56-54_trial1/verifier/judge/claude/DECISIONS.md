# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files organized as one file per session under `data/sub-<subject_id>/`. It uses `h5py` (not `pynwb`) to open each file directly, iterating over all files found via `glob.glob`. Multiprocessing with up to 12 worker processes is used to parallelize session processing.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with h5py.File(fname, 'r') as f:
    ident = f['identifier'][()].decode()
    mouse = f['general/subject/description'][()].decode().strip()
    subject_id = f['general/subject/subject_id'][()].decode().strip()
    tr = f['intervals/trials']
    # ... reads trial columns, units, behavioral events
```

iii. The AI chose `h5py` over `pynwb` for direct low-level access to the HDF5 structure, which it justified as enabling efficient bulk reads of spike times and avoiding per-unit small reads. The multiprocessing approach was chosen for speed.

## 1-b. How are the data split into subjects?

i. Each NWB file's subject is identified via `general/subject/description` (e.g., "SC015"), which is the mouse name. This is used to build the `subjects` list and `subject_idx` array.

ii.
```python
mouse = f['general/subject/description'][()].decode().strip()
# ...
subjects = sorted({r['info']['mouse'] for r in kept})
sub_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub_index[r['info']['mouse']] for r in kept], dtype=np.int64),
```

iii. The AI uses the mouse name (e.g., "SC015") from `general/subject/description` rather than the numeric `subject_id` (e.g., "440956"). Both uniquely identify subjects; the mouse name is more human-readable and matches what appears in session identifiers.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Sessions are identified by `f['identifier']` (e.g., "SC015_20190207_120657_s1"). The sorted glob of files determines session order.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
ident = f['identifier'][()].decode()
```

iii. Since each NWB file is a single session, no splitting logic is needed. Sorting the file paths gives chronological order within each subject.

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. The AI reads trial columns directly via h5py and verifies that go-cue event count matches the trial count.

ii.
```python
tr = f['intervals/trials']
n_trials_table = len(tr['id'])
start_time = tr['start_time'][:]
# ...
go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go) == n_trials_table, f'{ident}: go cue count != trial count'
```

iii. The trials table provides the canonical trial boundaries, with one go-cue event per trial confirmed by assertion.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered on three criteria: (1) covered by the ephys recording (`obs_intervals`), (2) not `auto_water` and not `free_water`, and (3) at least one neuron fires a spike in the 4s window. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
valid = observed & (auto_water == 0) & (free_water == 0)
trials = np.where(valid)[0]
if len(trials) < 2:
    return None
# ...
has_data = fr.sum(axis=(0, 2)) > 0
n_zero_trials = int((~has_data).sum())
if n_zero_trials:
    fr = fr[:, has_data, :]
    trials = trials[has_data]
```

iii. The AI justifies excluding auto_water and free_water trials as matching the reference code's `get_regular_trial_mask` (excluding the early-lick/no-response/photostim parts which are decoder targets). The zero-spike-trial filter catches cases where `obs_intervals` over-reports coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (with `spike_times_index` for ragged indexing). Only units with `classification == 'good'` and non-empty `anno_name` (CCF annotation) contribute. Go-cue times from `BehavioralEvents/go_start_times` define the bin edges.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
# ...
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

iii. The AI identified `classification` as the QC classifier verdict from the spike-sorting white paper, and required a non-empty CCF annotation (`anno_name`) to match the reference code's intersection of ephys units with histology.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning [-2.5, +1.5) s relative to the go cue, then divided by the bin width (0.05s) to get firing rates in spikes/s. No smoothing or normalization is applied.

ii.
```python
def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    edges = go_times[:, None] + BIN_EDGES[None, :]
    flat_edges = np.ascontiguousarray(edges.ravel())
    fr = np.empty((len(unit_idx), n_trials, N_BINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        lo = spike_index[u - 1] if u > 0 else 0
        st = spike_times[lo:spike_index[u]]
        pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
        fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
    fr /= BIN_SIZE
    return fr
```

iii. The conversion from spike counts to firing rates matches the reference code's `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` AND non-empty `anno_name` (CCF annotation) are kept. Sessions with no such units are dropped. This yields 69,453 units across 173 sessions.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
if len(unit_idx) == 0:
    return None
```

iii. The AI documents that `classification` is the spike-sorting QC classifier described in the white paper (`qc_mode='classifier'` in the reference code). The additional requirement for a CCF annotation matches the reference code's `helper_get_neuron_id_area`, which intersects ephys unit IDs with histology unit IDs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB timestamps share the same session-absolute clock, so alignment to the go cue is done by adding the relative bin edges to each trial's go-cue time to get absolute bin edges, then using `searchsorted` to count spikes per bin.

ii.
```python
go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# ...
edges = go_times[:, None] + BIN_EDGES[None, :]
flat_edges = np.ascontiguousarray(edges.ravel())
# ...
pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```

iii. The go-cue-aligned approach matches both the instructions and the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins per trial spanning [-2.5, +1.5) s. The bin grid is defined once as 81 edges and reused for all trials and sessions. No rebinning from an intermediate resolution is performed — spikes are binned directly at the target resolution.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)  # (81,)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])      # (80,)
```

iii. The 50ms bin width is mandated by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times/timestamps` (tone onsets) and the go-cue times. The last sample-epoch start before each trial's go cue is used as the tone onset.

ii.
```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
# ...
j = np.searchsorted(sample_start, go_v - 1e-9) - 1
assert np.all(j >= 0), f'{ident}: trial without a preceding tone onset'
tone_time = sample_start[j]
```

iii. Early licks can replay the sample epoch, producing multiple tone onsets per trial; using the last one before the go cue captures the final instruction tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as: `bin_center_relative_to_go + (go_time - tone_time)`. This gives a continuous ramp starting negative (before the tone) and increasing.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)
```

iii. Straightforward arithmetic combining the bin center offsets with the per-trial go-to-tone delay.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same go-cue-relative bin grid (`BIN_CENTERS`), so the k-th element of the input array corresponds to the k-th column of the neural firing rate matrix for each trial.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)
```

iii. Alignment is inherent in the shared bin grid definition.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from the trials table columns `photostim_onset` and `photostim_duration` (strings, `'N/A'` when no stimulation), plus `start_time` to convert to absolute time, and go-cue times for alignment.

ii.
```python
ps_onset = _decode(tr['photostim_onset'][:])
ps_dur = _decode(tr['photostim_duration'][:])
# ...
has_ps = ps_onset[trials] != 'N/A'
on = np.array([float(x) for x in ps_onset[trials][has_ps]])
dur = np.array([float(x) for x in ps_dur[trials][has_ps]])
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
ps_t1 = ps_t0 + dur
```

iii. Onset is stored relative to trial start, so it is converted to go-cue-relative time by adding `start_time` and subtracting the go-cue time.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is marked as 1 if it **overlaps** the stimulation interval `[ps_t0, ps_t1)`, using bin-edge overlap detection. Bins are 0 for non-stimulated trials (where `photostim_onset == 'N/A'`).

ii.
```python
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
photostim[has_ps] = ov.astype(np.float32)
```

iii. The overlap-based approach marks a bin as "on" if any part of the bin overlaps the stimulation interval, rather than checking only the bin center.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are expressed relative to the go cue, matching the neural bin grid. The bin edges used for overlap detection are the same edges used for spike binning.

ii.
```python
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]   # rel. to go cue
ps_t1 = ps_t0 + dur
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
```

iii. Using the same go-cue-relative coordinate system ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from two trials-table columns: `outcome` ('hit'/'miss'/'ignore') and `trial_instruction` ('left'/'right'). Choice is not stored directly.

ii.
```python
oc = outcome[trials]
ins = instruction[trials]
choice = np.full(len(trials), 2, dtype=np.int64)              # 2 = no lick
licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
choice[licked_left] = 0
choice[licked_right] = 1
```

iii. A hit means the animal licked the instructed side, a miss means the opposite, and ignore means no lick. This logic derives the actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick. It is a per-trial value broadcast across all 80 time bins.

ii.
```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. Per-trial outputs are repeated across bins so all outputs share a consistent `(n_output, n_timepoints)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains 'ignore', 'miss', or 'hit'.

ii.
```python
outcome = _decode(tr['outcome'][:])
# ...
oc = outcome[trials]
outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                      default=-1).astype(np.int64)
```

iii. The trials table stores outcome explicitly in the three categories the instructions require.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers: ignore=0, miss=1, hit=2. An assertion checks for unexpected values. The result is broadcast across all 80 bins.

ii.
```python
outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                      default=-1).astype(np.int64)
assert np.all(outcome_c >= 0), f'{ident}: unexpected outcome value'
```

iii. Straightforward categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early = _decode(tr['early_lick'][:])
# ...
early_c = (early[trials] == 'early').astype(np.int64)
```

iii. The trials table provides this flag directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early lick) or 1 (early lick) via a boolean comparison. Broadcast across all 80 bins.

ii.
```python
early_c = (early[trials] == 'early').astype(np.int64)
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` data = (x, y, likelihood) with corresponding timestamps.

ii.
```python
tongue_key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
tdata = f[tongue_key + '/data'][:]
tts = f[tongue_key + '/timestamps'][:]
```

iii. This is the DeepLabCut tongue tracking from the side camera, present in all sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.5 are marked as not visible. Visible frames are averaged into 50ms bins per trial. Per-session percentiles (40th, 60th) of the visible per-bin mean y values are computed, and bins are classified as 0 (<40th), 1 (40th-60th), 2 (>60th), or 3 (not visible).

ii.
```python
def tongue_per_bin(track_data, track_ts, go_times):
    ycol = track_data[:, 1]
    vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH
    # ... per-trial binning of visible frames
    cnt = np.bincount(b_idx[v], minlength=N_BINS)
    tot = np.bincount(b_idx[v], weights=ycol[a:b][v], minlength=N_BINS)
    y[t, m] = tot[m] / cnt[m]

def discretize_tongue(y, visible):
    vals = y[visible]
    p40, p60 = np.percentile(vals, TONGUE_PCTL)
    cls[visible & (y < p40)] = 0
    cls[visible & (y >= p40) & (y <= p60)] = 1
    cls[visible & (y > p60)] = 2
```

iii. The AI computes percentiles over visible per-bin mean y values from the trial windows. The likelihood threshold of 0.5 was chosen because the distribution is bimodal (89% < 0.01, 10.5% > 0.99).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using explicit comparisons against the 40th and 60th percentiles of visible per-bin y values:
- Class 0: y < 40th percentile
- Class 1: 40th percentile <= y <= 60th percentile
- Class 2: y > 60th percentile
- Class 3: tongue not visible

ii.
```python
cls[visible & (y < p40)] = 0
cls[visible & (y >= p40) & (y <= p60)] = 1
cls[visible & (y > p60)] = 2
```

iii. The explicit inequality approach handles boundaries with `<=` for the middle class upper bound.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same session-absolute clock as spikes and go cues. Per-trial frames are found via `searchsorted` on camera timestamps at `go + OFF_START` and `go + OFF_END`, then assigned to the same 50ms bins used for neural data.

ii.
```python
lo = np.searchsorted(track_ts, go_times + OFF_START, side='left')
hi = np.searchsorted(track_ts, go_times + OFF_END, side='left')
for t in range(n_trials):
    rel = track_ts[a:b] - go_times[t]
    b_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b_idx, 0, N_BINS - 1, out=b_idx)
```

iii. The same go-cue-relative bin grid ensures bin k of the tongue output covers the same time interval as bin k of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled:
- **Sessions with no QC-passing units** (1 session with NaN classification): dropped entirely.
- **Trials outside ephys coverage**: excluded via `obs_intervals`. Additionally, trials where no neuron fires at all are dropped (catches obs_intervals over-reporting).
- **Tongue not visible**: bins with no visible frame get class 3 ("not visible").
- **Auto-water and free-water trials**: excluded as non-genuine behavioral reports.

ii.
```python
# Session with no good units
if len(unit_idx) == 0:
    return None

# Zero-spike trials
has_data = fr.sum(axis=(0, 2)) > 0
if n_zero_trials:
    fr = fr[:, has_data, :]

# Tongue not visible
cls = np.full(y.shape, 3, dtype=np.int64)
```

iii. The AI documents investigating specific edge cases like the SC015_20190208_133600 session where obs_intervals lists one more trial than the spike record covers.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports that reading each NWB file dominates runtime. With multiprocessing (16 workers), the full conversion takes ~41s for 174 sessions. Within a session, bulk reading of `units/spike_times` and the per-unit `searchsorted` loop are the main costs. Pickling the 11.89 GB result takes additional time.

ii. N/A (runtime characteristics, not specific code)

iii. The AI profiled the conversion and found that with parallelization, no further optimization was needed to stay within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) the per-unit loop in spike binning (one `searchsorted` per unit, but trials are already vectorized via flattened edges), and (2) the per-trial loop in tongue processing.

ii.
```python
for i, u in enumerate(unit_idx):
    lo = spike_index[u - 1] if u > 0 else 0
    st = spike_times[lo:spike_index[u]]
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```

iii. The per-unit loop cannot be vectorized because each unit has a different number of spikes (ragged storage). The tongue loop iterates over trials rather than frames and is not a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed. Each NWB file is opened once and every quantity derived from it is computed once. The bin grid is defined once at module level.

ii. N/A

iii. The tongue percentiles are per-session and computed within the single-pass processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes extensive per-session metadata (hemisphere fractions, mean firing rates, detailed trial accounting) stored in `session_info`. The explicit CCF-coordinate-based brain region assignment with the 14-group mapping is more elaborate than needed, since the downstream decoder doesn't differentiate between brain regions. The `raw` data retained for `--show-processing` plots is discarded for non-diagnostic runs.

ii.
```python
info = dict(
    session_id=ident, mouse=mouse, subject_id=subject_id, file=os.path.basename(fname),
    n_units_total=len(classification), n_units_good=int((classification == 'good').sum()),
    n_neurons=len(unit_idx), n_trials_table=n_trials_table, n_trials=n_tr,
    n_trials_observed=int(observed.sum()),
    n_excluded_water=int((observed & ((auto_water != 0) | (free_water != 0))).sum()),
    n_excluded_no_spikes=n_zero_trials,
    # ... more metadata
)
```

iii. The extra metadata is useful for documentation and debugging but is not used by the decoder.
