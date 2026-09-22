# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes one NWB file per session with `h5py`; by default it uses a spawned multiprocessing pool (12 workers, 16 in the documented full run). It reads trial-table, event, unit, electrode, and tongue-tracking datasets directly from their HDF5 paths.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with h5py.File(fname, 'r') as f:
    tr = f['intervals/trials']
    go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    u = f['units']
```

iii. The notes say the release contains 174 NWB files and that each is one session. Sorting makes ordering deterministic; reading each file once and parallelizing sessions made the full conversion take about 41 s.

## 1-b. How are the data split into subjects?

i. The agent reads both the numeric `subject_id` and the mouse name from `general/subject/description`, but uses the mouse name (for example `SC015`) as the output subject identifier. It builds a sorted unique mouse list and maps each retained session into it.

ii.
```python
mouse = f['general/subject/description'][()].decode().strip()
subject_id = f['general/subject/subject_id'][()].decode().strip()
subjects = sorted({r['info']['mouse'] for r in kept})
'subject_idx': np.array([sub_index[r['info']['mouse']] for r in kept], dtype=np.int64),
```

iii. The notes map `description` to the paper's mouse name and report 28 unique mice, matching the dataset/paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; retained sessions remain in sorted-file order and are identified by the NWB `identifier`.

ii.
```python
ident = f['identifier'][()].decode()
results = [None] * len(files)
'neural': [r['neural'] for r in kept],
```

iii. The agent states that this is the dataset's native organization. It retained 173 of 174 files, dropping the file with no good units.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The same row indices subset go cues and all trial variables; the code asserts one go cue per row and finally creates one matrix per retained trial.

ii.
```python
n_trials_table = len(tr['id'])
assert len(go) == n_trials_table
trials = np.where(valid)[0]
neural = [np.ascontiguousarray(fr[:, t, :]) for t in range(n_tr)]
```

iii. The trials table is authoritative; unlike sample events, go cues have exactly one entry per trial. The original row indices are saved in session metadata for traceability.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials covered by the union of good units' `obs_intervals`, excludes both `auto_water` and `free_water`, and then drops any retained trial with zero spikes from every selected neuron throughout the 4-s window. Sessions with fewer than two remaining trials are dropped. Early-lick, ignore, and photostimulation trials are deliberately retained.

ii.
```python
valid = observed & (auto_water == 0) & (free_water == 0)
trials = np.where(valid)[0]
has_data = fr.sum(axis=(0, 2)) > 0
trials = trials[has_data]
if len(trials) < 2: return None
```

iii. The agent says required decoder targets/inputs would become constant if early-lick, ignore, or stimulation trials were removed. Water trials are considered nonstandard, and two all-zero trials were interpreted as `obs_intervals` coverage errors. This yielded 89,544 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `spike_times_index`, selected using `units/classification` and `anno_name`, and placed relative to `go_start_times`.

ii.
```python
classification = _decode(u['classification'][:])
keep = (classification == 'good') & (anno != '')
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

iii. The notes identify classifier-`good` units as the paper/reference QC and require a CCF annotation for downstream region assignment.

## 2-b. How is the `neural` data processed?

i. For each selected unit, the code uses `searchsorted` at all trial/bin edges, differences cumulative positions to obtain spike counts, then divides by 0.05 s to produce unsmoothed firing rates in spikes/s (`float32`).

ii.
```python
pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
fr /= BIN_SIZE
```

iii. The agent says this mirrors the reference `sliding_histogram(..., rate=True)` with stride equal to bin width and validated random bins against direct spike counting.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose classifier label is `good` and whose annotation string is nonempty are retained. A session with none is discarded; no individual metric thresholds or `is_good_trials` filtering are used.

ii.
```python
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
if len(unit_idx) == 0:
    return None
```

iii. The classifier is described as the region-specific spike-sorting QC used by the reference. The agent found `is_good_trials` did not correspond to absent or unusually low-rate data and did not use it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are each trial's go-cue timestamp plus fixed offsets from -2.5 to +1.5 s; spikes use the same session clock.

ii.
```python
edges = go_times[:, None] + BIN_EDGES[None, :]
flat_edges = np.ascontiguousarray(edges.ravel())
```

iii. NWB event and spike timestamps share a session-absolute clock, so no clock transform or interpolation is needed. Time-resolved decoder curves were used as an alignment sanity check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over `[-2.5, 1.5)` s. Raw spikes are newly histogrammed at this resolution; there is no further rebinning or smoothing.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

iii. This is exactly the decoder task's requested window and resolution.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, each trial's go cue, and the common bin centers. The last sample start strictly before the go cue is selected.

ii.
```python
j = np.searchsorted(sample_start, go_v - 1e-9) - 1
tone_time = sample_start[j]
```

iii. Early licks can replay the sample epoch, so a trial can contain multiple tone starts; the agent chose the last tone used before that go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The go-relative bin-center time is shifted by `go_time - tone_time`, producing elapsed seconds since tone onset as `float32`.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] +
                  (go_v - tone_time)[:, None]).astype(np.float32)
```

iii. The notes report the expected 1.85-s tone-to-go interval on most trials and validate the ramp visually.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same 80 go-relative bin centers as the neural histograms.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
time_from_tone = BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]
```

iii. Both streams use the same go cue and bin grid, so index `k` denotes the same time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trial-table `photostim_onset`, `photostim_duration`, and `start_time`, then converted to go-relative times with the trial go cue.

ii.
```python
on = np.array([float(x) for x in ps_onset[trials][has_ps]])
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
ps_t1 = ps_t0 + dur
```

iii. The agent notes that onset is stored relative to trial start and cross-checked these intervals against BehavioralEvents to within 10 ms.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Nonstimulated trials are all zero. For stimulated trials, a bin is one if any part of its half-open interval overlaps the stimulation interval, otherwise zero.

ii.
```python
photostim = np.zeros((len(trials), N_BINS), dtype=np.float32)
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & \
     (BIN_EDGES[None, 1:] > ps_t0[:, None])
photostim[has_ps] = ov.astype(np.float32)
```

iii. The agent wanted a binary time-varying signal and explicitly chose overlap. It documented that a nominal 0.5-s stimulus often marks 11 bins because it straddles boundaries.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation start/end are expressed relative to the go cue and compared with the exact neural bin edges.

ii.
```python
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
```

iii. The shared go-relative grid aligns the binary input to each neural column.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from the trials-table `outcome` and `trial_instruction`: a hit selects the instructed side, a miss the opposite side, and ignore remains no lick.

ii.
```python
licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
```

iii. No direct choice column exists. The agent treats trial-table outcome as authoritative and reports cross-validation against response-window lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are 0 left, 1 right, 2 no lick; the per-trial code is repeated across all 80 time bins.

ii.
```python
choice = np.full(len(trials), 2, dtype=np.int64)
choice[licked_left] = 0
choice[licked_right] = 1
np.full(N_BINS, choice[t])
```

iii. Repetition lets all four outputs share one time-varying `(4, 80)` array while preserving a per-trial label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome = _decode(tr['outcome'][:])
oc = outcome[trials]
```

iii. The column already contains exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` are mapped to 0, 1, and 2, validated against unexpected values, and broadcast over 80 bins.

ii.
```python
outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2], default=-1)
assert np.all(outcome_c >= 0)
```

iii. This directly follows the requested category order and common output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from `intervals/trials/early_lick` (`early` or `no early`).

ii.
```python
early = _decode(tr['early_lick'][:])
early_c = (early[trials] == 'early').astype(np.int64)
```

iii. The trials table supplies the required flag directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Equality to `early` produces 1; all `no early` values produce 0. The label is repeated over 80 bins.

ii.
```python
early_c = (early[trials] == 'early').astype(np.int64)
np.full(N_BINS, early_c[t])
```

iii. This matches the requested no/yes ordering and the shared output-array layout.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses columns 1 (y) and 2 (DeepLabCut likelihood) plus timestamps from `Camera0_side_TongueTracking`, aligned with trial go cues.

ii.
```python
tdata = f[tongue_key + '/data'][:]
tts = f[tongue_key + '/timestamps'][:]
ycol = track_data[:, 1]
vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. The notes identify this side-camera stream as the reference's tongue-y stream and use likelihood to distinguish a visible tongue from tracker noise.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial and bin, the code averages y over frames with likelihood above 0.5. Bins with no qualifying frame remain NaN/not visible. Percentiles are then computed over all visible per-bin means from retained trial windows in that session.

ii.
```python
cnt = np.bincount(b_idx[v], minlength=N_BINS)
tot = np.bincount(b_idx[v], weights=ycol[a:b][v], minlength=N_BINS)
y[t, m] = tot[m] / cnt[m]
vals = y[visible]
p40, p60 = np.percentile(vals, TONGUE_PCTL)
```

iii. The agent says the likelihood distribution is strongly bimodal, per-bin means match the quantity classified, and per-session percentiles satisfy the instruction.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Using per-session 40th/60th percentiles: class 0 is below p40, class 1 is p40 through p60 inclusive, class 2 is above p60, and class 3 is not visible.

ii.
```python
cls = np.full(y.shape, 3, dtype=np.int64)
cls[visible & (y < p40)] = 0
cls[visible & (y >= p40) & (y <= p60)] = 1
cls[visible & (y > p60)] = 2
```

iii. This implements the requested 40/20/40 visible-bin proportions plus an explicit missing-visibility category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frame slices are selected from `go-2.5` to `go+1.5`; each frame is assigned by floor division to the same 50-ms go-relative bin index used for neural activity.

ii.
```python
lo = np.searchsorted(track_ts, go_times + OFF_START, side='left')
rel = track_ts[a:b] - go_times[t]
b_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. Camera, spike, and event timestamps share the NWB session clock, so matching absolute intervals gives direct bin alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no usable units is dropped; partial ephys coverage is inferred from `obs_intervals`; two all-zero spike windows are dropped; missing photostimulation becomes zeros; absent visible tongue becomes class 3; a session with no visible tongue receives all class 3 and NaN thresholds. Truncated within-trial spike coverage is otherwise left as zero firing rate rather than imputed or causing trial removal.

ii.
```python
if len(unit_idx) == 0: return None
has_data = fr.sum(axis=(0, 2)) > 0
photostim = np.zeros((len(trials), N_BINS), dtype=np.float32)
cls = np.full(y.shape, 3, dtype=np.int64)
```

iii. The notes distinguish true missing measurements from whole-trial coverage failures. They retain truncated error trials because requiring full coverage would remove most misses, and quantify the resulting decoder artifact.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike/tongue arrays, per-unit spike binning, conversion/assembly of an 11.89-GB object, and final pickling are the main costs. Multiprocessing reduced the documented conversion to 41 s; decoder training (not conversion) took about nine minutes.

ii.
```python
spike_times = u['spike_times'][:]
for i, u in enumerate(unit_idx):
    pos = np.searchsorted(st, flat_edges)
with ctx.Pool(min(args.workers, len(files))) as pool:
```

iii. The agent explicitly optimized away many small HDF5 reads and parallelized the independent session workload.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit loop in `bin_spikes`, trial loop in `tongue_per_bin`, annotation loop, observation-interval union loop, and output/list assembly loops remain. Trial/bin spike-edge work is already vectorized; ragged unit spike arrays make complete unit vectorization awkward. Tongue frames and output assembly could be globally indexed/vectorized further.

ii.
```python
for i, u in enumerate(unit_idx): ...
for t in range(n_trials): ...
for i, name in enumerate(anno): ...
for k in unit_idx: ...
```

iii. The notes focus on the important optimization: replacing a nested unit-by-trial spike loop with one `searchsorted` per unit over flattened edges.

## 10-c. What processing does the code repeat multiple times?

i. Per-session setup, HDF5 reads, observation mapping, region mapping, and array/list assembly repeat for every file. Within each session, `searchsorted` repeats per unit and tongue aggregation repeats per trial. In optional plotting, raw spikes are reopened/read again. Core derived quantities are otherwise computed once.

ii.
```python
for i, res in enumerate(pool.imap(_worker, list(zip(files, want_raw)))):
for i, u in enumerate(unit_idx):
for t in range(n_trials):
with h5py.File(raw['fname'], 'r') as f:
```

iii. The agent justifies repetition as natural for independent files and ragged streams; global bin constants are created only once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `stop_time`, numeric `subject_id`, ML coordinates, and several counts mainly for diagnostics/metadata; it performs extensive 14-region mapping not required by the decoder predictions but required by the target bookkeeping. Optional `--show-processing` retains raw arrays, reopens spikes, and creates plots that are discarded from the pickle. It also computes detailed summaries after conversion.

ii.
```python
stop_time = tr['stop_time'][:]
subject_id = f['general/subject/subject_id'][()].decode().strip()
ccf_ml = etab['x'][:][el[unit_idx]]
if want_raw: raw = dict(...)
make_processing_plot(...)
_summary(data)
```

iii. The agent presents these as validation, traceability, required region bookkeeping, and optional diagnostics rather than decoder features; `raw` is `None` in the normal full run.
