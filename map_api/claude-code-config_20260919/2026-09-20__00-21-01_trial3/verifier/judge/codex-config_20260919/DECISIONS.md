# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes each file (normally in a 24-process pool). Each file is opened with `pynwb.NWBHDF5IO`; trials, events, tracking, units, electrodes, and subject metadata are read through the NWB API.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    raw = read_session(nwb)
```

iii. The agent states that NWB is the release format, that one file represents one session, and that `pynwb` was required. It found 174 files in 28 subject directories and used parallel session conversion for speed.

## 1-b. How are the data split into subjects?

i. For each session it reads both `nwb.subject.description` (mouse name such as `SC015`) and `subject_id`; output subjects are the sorted unique mouse descriptions, and `subject_idx` maps each retained session to that list.

ii.
```python
'mouse': nwb.subject.description,
'subject_id': str(nwb.subject.subject_id),
...
mice = sorted({r['mouse'] for r in sessions})
'subject_idx': np.array([mice.index(r['mouse']) for r in sessions], dtype=np.int64),
```

iii. The agent preferred the paper-style mouse name while retaining the numeric NWB subject id in session metadata. It regarded both as stable identifiers and reported 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session, identified by `nwb.identifier`. Retained sessions are sorted by identifier before assembly; behavioral criteria additionally reject some whole sessions.

ii.
```python
'identifier': nwb.identifier,
...
sessions = sorted(sessions, key=lambda r: r['identifier'])
```

iii. The file boundary is the published session boundary. The agent also applied the data-paper criteria of performance above 65% and at least 50 correct control trials per direction, producing 142 retained sessions.

## 1-d. How are the data split into trials?

i. Rows of `nwb.intervals['trials']` define trials. The row count must equal the count of `go_start_times`; retained row indices are used consistently for all trial-level arrays.

ii.
```python
trials = nwb.intervals['trials']
'start_time': np.asarray(trials['start_time'][:], dtype=np.float64),
...
assert len(go) == n_trials_all
trial_idx = np.flatnonzero(keep)
```

iii. The trials table is the authoritative trial segmentation, while the assertion checks the one-go-cue-per-trial mapping observed in all files.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials covered by unit `obs_intervals`, removes `auto_water` and `free_water` trials, and removes trials with no video frame in the four-second window. Early-lick, ignore, and photostimulation trials remain. Sessions must then have at least two usable trials and pass behavioral session curation.

ii.
```python
keep = raw['recorded'] & (raw['auto_water'] == 0) & (raw['free_water'] == 0)
n_frames_all = (np.searchsorted(raw['video_t'], edges_all[:, -1])
                - np.searchsorted(raw['video_t'], edges_all[:, 0]))
keep &= n_frames_all > 0
```

iii. Unrecorded trials would create false all-zero neural data; water trials are not genuine behavioral reports; video-absent trials would conflate unmeasured tongue position with invisibility. Required decoder classes are deliberately retained. Behavioral session thresholds were taken from the data paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each QC-good unit's `units['spike_times']`; `go_start_times` supplies absolute trial alignment.

ii.
```python
spike_index = units['spike_times']
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
```

iii. Spike times are the available raw neural representation, and all NWB timestamps share the session clock.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` obtains cumulative spike positions at every trial/bin edge, `diff` produces counts, and division by 0.05 converts counts to Hz. No smoothing or normalization is applied.

ii.
```python
pos = np.searchsorted(spike_times, edges_abs.ravel()).reshape(edges_abs.shape)
return np.diff(pos, axis=1).astype(np.int32)
...
neural = np.ascontiguousarray(
    counts.transpose(1, 0, 2).astype(np.float32) / BIN_WIDTH)
```

iii. This matches the reference analysis's firing-rate histogram behavior while vectorizing all trials for each ragged unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `good` are retained. Sessions with no good unit are rejected; no second set of metric thresholds is applied.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
...
if n_good == 0:
    reject = 'no good units'
```

iii. The agent identifies this classification as the classifier verdict described by the QC paper and argues that thresholding its input metrics again would double-filter units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's absolute bin edges are the go-cue timestamp plus the fixed relative edges from -2.5 to +1.5 seconds; spikes are counted between those edges.

ii.
```python
def window_edges(go_times):
    return go_times[:, None] + BIN_EDGES[None, :]
edges = window_edges(go)
```

iii. Spikes, go cues, and behavior use the same absolute NWB clock, so no offset correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins spanning `[-2.5, 1.5)` seconds around the go cue. Raw spike times are histogrammed directly to that resolution; there is no later rebinning.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_WIDTH))
BIN_EDGES = OFF_START + BIN_WIDTH * np.arange(N_BINS + 1)
```

iii. These values are explicitly required by the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, the retained trials' go-cue times, and the common bin centers. The selected tone is the last sample onset at or before each go cue.

ii.
```python
tone_idx = np.searchsorted(raw['sample_start'], go, side='right') - 1
tone = raw['sample_start'][tone_idx]
```

iii. Early licking can replay sample/delay epochs, creating multiple sample starts; the last one is the instruction episode immediately preceding the eventual go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The go-relative bin center is shifted by `go - tone`, yielding elapsed seconds from tone onset at each bin center and cast to float32.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go - tone)[:, None]).astype(np.float32)
```

iii. This is the direct clock difference; assertions ensure every selected tone exists and lies inside its trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the centers of exactly the same go-relative 50-ms bins used for neural spike counts.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
time_from_tone = BIN_CENTERS[None, :] + (go - tone)[:, None]
```

iii. Shared go cues and bin centers make input column `b` contemporaneous with neural bin `b`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It reads absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, using trial start times to assign each event to a retained trial.

ii.
```python
'photostim_start': np.asarray(events['photostim_start_times'].timestamps[:]),
'photostim_stop': np.asarray(events['photostim_stop_times'].timestamps[:]),
stim_trial = np.searchsorted(raw['start_time'], raw['photostim_start'], side='right') - 1
```

iii. The agent chose timestamped event streams as the direct record of illumination, and cross-checked them against trial photostimulation flags.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It initializes zeros and sets a bin to one if the half-open neural bin has any overlap with a stimulation interval. Multiple intervals would be combined.

ii.
```python
overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
photostim[row, overlap] = 1.0
```

iii. This creates the requested time-varying binary input and preserves partially illuminated bins rather than reducing stimulation to a trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are intersected with the same absolute edge matrix used to count spikes.

ii.
```python
edges = window_edges(go)
overlap = (edges[row, :-1] < off) & (edges[row, 1:] > on)
```

iii. All involved timestamps use the same session clock, so identical bin boundaries ensure alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `trial_instruction` and `outcome` because no direct choice column is present.

ii.
```python
instruction = raw['instruction'][trial_idx]
outcome = raw['outcome'][trial_idx]
```

iii. A hit implies the instructed port, a miss the opposite port, and ignore no response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Instruction maps left/right to 0/1, misses invert that code, and ignores become 2. The per-trial value is repeated across all 80 bins.

ii.
```python
choice = np.where(instruction == 'left', CHOICE_LEFT, CHOICE_RIGHT)
choice = np.where(outcome == 'miss', 1 - choice, choice)
choice = np.where(outcome == 'ignore', CHOICE_NOLICK, choice)
outputs[:, 0, :] = choice[:, None]
```

iii. This reconstructs actual reported direction from task semantics and gives no-response trials their required category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the retained rows of the trials-table `outcome` column.

ii.
```python
outcome = raw['outcome'][trial_idx]
```

iii. The raw categories already exactly match ignore, miss, and hit.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to integer codes 0/1/2 and the trial-level code is repeated over time.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[o] for o in outcome], dtype=np.int8)
outputs[:, 1, :] = outcome_code[:, None]
```

iii. Integer categorical labels satisfy the decoder format; repetition permits one uniform `(4, 80)` output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from retained rows of `trials['early_lick']`.

ii.
```python
early = raw['early_lick'][trial_idx]
```

iii. The NWB table explicitly stores `early` versus `no early`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Equality with `early` produces 1; all `no early` rows produce 0. The value is repeated across bins.

ii.
```python
early_code = (early == 'early').astype(np.int8)
outputs[:, 2, :] = early_code[:, None]
```

iii. This implements the required no/yes categorical output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 and 2 (y and likelihood) from `Camera0_side_TongueTracking`.

ii.
```python
tongue = tracking['Camera0_side_TongueTracking']
out['video_t'] = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
out['tongue_y'] = tongue_data[:, 1]
out['tongue_likelihood'] = tongue_data[:, 2]
```

iii. This is the release's side-camera tongue measurement; likelihood distinguishes visible tracking from retracted/occluded frames.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood above 0.9 count as visible. Cumulative sums reduce their y values to means on the neural 50-ms grid. No-visible-frame bins become class 3. The 40th/60th percentiles are computed from visible bin means across retained trial windows in that session.

ii.
```python
visible = raw['tongue_likelihood'] > TONGUE_LIKELIHOOD_THRESHOLD
_, n_visible_bin, y_sum_bin = segment_sums(raw['tongue_y'], visible, edges, raw['video_t'])
y_mean = np.where(bin_visible, y_sum_bin / np.maximum(n_visible_bin, 1), np.nan)
p_low, p_high = np.percentile(y_mean[bin_visible], [40.0, 60.0])
```

iii. The agent reports strongly bimodal likelihoods, making 0.9 nearly insensitive to threshold choice. It uses bin means because those are the classified observations and uses only retained trials because that is the population actually decoded.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible means below the 40th percentile are 0, values from the 40th through 60th percentile are 1, and values above the 60th are 2; bins with no visible frame are 3.

ii.
```python
cls = np.where(y_mean < p_low, 0, np.where(y_mean > p_high, 2, 1))
tongue_class[bin_visible] = cls[bin_visible].astype(np.int8)
```

iii. This directly implements the requested four category definitions, with equality assigned to the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples are located between the exact same absolute go-cue-relative bin edges used for spikes; cumulative counts/sums yield one tongue mean per neural bin.

ii.
```python
pos = np.searchsorted(sample_times, edges_abs.ravel()).reshape(edges_abs.shape)
n_masked_bin = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
masked_sum = cum_v[pos[:, 1:]] - cum_v[pos[:, :-1]]
```

iii. Camera and electrophysiology timestamps share the NWB session clock, so no interpolation or independent offset is required.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Assertions guard event/trial correspondence, tone placement, and observation intervals. No-good-unit sessions, insufficient/bad-performance sessions, unrecorded/water/no-video trials are rejected. Low-confidence or absent tongue samples map to `not visible`. Worker exceptions reject the affected file while allowing the run to continue.

ii.
```python
assert len(go) == n_trials_all
if reject is not None:
    return {'identifier': raw['identifier'], 'rejected': reject, ...}
...
except Exception as exc:
    return {'identifier': os.path.basename(path), 'rejected': f'exception: {exc!r}'}
```

iii. The stated principle is to exclude genuinely unrecorded observations rather than fabricate zeros, while representing legitimate tongue invisibility as an explicit class. Assertions and recorded rejection reasons make inconsistencies auditable.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike/video arrays, per-unit spike binning, holding/assembling roughly 10 GB of neural output, and final pickling are the dominant work. Sessions are therefore processed in parallel.

ii.
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for res in pool.map(_worker, jobs):
        results.append(res)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Notes identify NWB I/O and ragged spike searches as data-scaled costs and report about 26 seconds conversion plus output writing on the available many-core machine.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial and bin loops for neural and video processing were already vectorized. A loop remains over ragged units, and a small loop remains over stimulation events; session work is parallelized. The unit loop is difficult to collapse because every unit has a different spike array.

ii.
```python
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
...
for row, on, off in zip(...):
    photostim[row, overlap] = 1.0
```

iii. The agent explicitly optimized the costly trial/bin dimensions using flattened edge searches and prefix sums; further vectorization would chiefly affect ragged data or a small number of events.

## 10-c. What processing does the code repeat multiple times?

i. Little expensive work is repeated: each session is opened once and shared grids are module constants. It does separately retrieve every good unit's ragged spike array, and assembly makes contiguous per-trial copies required by the target representation.

ii.
```python
for i, unit in enumerate(raw['good_units']):
    counts[i] = bin_spike_times(np.asarray(spike_index[int(unit)]), edges)
...
data['neural'].append([np.ascontiguousarray(x) for x in r['neural']])
```

iii. The agent claims no substantive recomputation; per-session percentiles allow a single pass and fixed edge arrays are reused.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Core conversion intermediates are necessary, but some computed diagnostics do not affect decoder arrays: `segment_sums` returns total frame counts that are ignored, and the script calculates extensive QC summaries/metadata. With `--show-processing`, raw debug traces and diagnostic plots are also generated solely for validation.

ii.
```python
_, n_visible_bin, y_sum_bin = segment_sums(...)
...
'frac_bins_no_spikes_in_trial': float(np.mean(...)),
if collect_debug:
    result['debug'] = {...}
```

iii. The agent describes these diagnostics as sanity checks and provenance rather than downstream decoder features; debug work is optional, while metadata is intentionally preserved in the pickle.
