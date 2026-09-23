# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `.nwb` file under each subject directory, sorts the paths, and processes one file per session with `pynwb.NWBHDF5IO`. Full runs use a spawn-based multiprocessing pool (16 workers by default).

ii.
```python
for sub in sorted(os.listdir(data_dir)):
    ...
    for f in sorted(os.listdir(d)):
        if f.endswith('.nwb'):
            files.append(os.path.join(d, f))
```
```python
io = NWBHDF5IO(path, 'r', load_namespaces=True)
nwb = io.read()
```

iii. The notes say there are 174 NWB files and that each is one session. The agent chose `pynwb` as required, deterministic sorting, and multiprocessing because NWB reads and per-session processing dominate runtime.

## 1-b. How are the data split into subjects?

i. The subject for a session is read from `nwb.subject.subject_id`. At assembly, unique IDs are sorted and each session gets an integer index into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The agent identifies the NWB subject field as the canonical animal identifier and reports 28 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; retained session order follows sorted file order. Sessions with no good units or fewer than two curated trials are dropped.

ii.
```python
session_id = nwb.identifier
...
if len(good) == 0:
    return None
...
if n_trials < 2:
    return None
```

iii. The notes state that the dataset layout is one NWB file per behavioral session and that dropping the sole file with no QC-good units recovers the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Trials are rows of `nwb.trials`. The agent requires one `go_start_times` event per raw trial, maps observed intervals back to trial rows by start time, and retains the corresponding row indices.

ii.
```python
n_trials_raw = len(trials)
go_times = _events(nwb, 'go_start_times')
if len(go_times) != n_trials_raw:
    raise RuntimeError(...)
obs_trial = np.searchsorted(trial_start, obs[:, 0] - 1e-6)
```

iii. The trials table gives the explicit trial boundaries. The notes say `obs_intervals` exactly match trial start/stop boundaries and are needed because some recordings cover only a subset of behavioral trials.

## 1-e. How are trials filtered based on quality controls?

i. Starting from trials represented in `units.obs_intervals`, the agent removes `auto_water` and `free_water` trials, requires `is_good_trials` to be true for every retained good unit, and later removes trials with zero spikes across every good unit and time bin. Early-lick, ignore, and photostim trials are deliberately retained. Sessions with fewer than two survivors are removed.

ii.
```python
keep = (~auto_water[obs_trial]) & (~free_water[obs_trial])
is_good_trials = np.asarray(units['is_good_trials'][:])[good]
keep &= is_good_trials.all(axis=0)
keep_idx = obs_trial[keep]
```
```python
nonempty = fr.sum(axis=(0, 2)) > 0
fr = fr[:, nonempty, :]
keep_idx = keep_idx[nonempty]
```

iii. The agent cites the repository's `get_regular_trial_mask` for water-trial removal and interprets `is_good_trials` as recording-stability/drift QC. It keeps early/ignore/photostim trials because those categories are required decoder targets or inputs, and removes all-zero trials as periods when amplifiers were not running.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `nwb.units['spike_times']`, selected by `units['classification']`, and is positioned using `BehavioralEvents/go_start_times`.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
sv = units['spike_times']
flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)
```

iii. The notes identify spike times as the NWB neural representation and `classification == 'good'` as the released region-specific QC classifier verdict.

## 2-b. How is the `neural` data processed?

i. For every good unit, absolute bin edges are formed around all retained go cues. `searchsorted` followed by `diff` produces spike counts, which are divided by 0.05 s to obtain Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
edges = go_times[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(spike_times, edges.ravel(), side='left').reshape(edges.shape)
counts = np.diff(idx, axis=1)
return (counts / BIN_SIZE).astype(np.float32)
```

iii. The agent states this matches the reference `sliding_histogram(..., rate=True)` while replacing its paper-specific sliding grid with the task-required nonoverlapping 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `'good'` are retained. There is no additional firing-rate or individual-metric threshold. A session with no good units is dropped.

ii.
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
if len(good) == 0:
    return None
```

iii. The classifier is the QC approach described in the data/QC papers. The notes reject a 2 Hz filter because it was specific to a firing-rate prediction analysis and could discard decodable information.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All timestamps use the NWB session clock. For each trial, relative edges from -2.5 to +1.5 s are added to that trial's go-cue timestamp before binning spikes.

ii.
```python
go = go_times[keep_idx]
edges = go_times[:, None] + BIN_EDGES[None, :]
```

iii. The agent notes that spikes, video, and events share one absolute clock, so alignment requires no interpolation or clock correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 nonoverlapping 50 ms bins over `[-2.5, 1.5)` s relative to the go cue. Raw spikes are histogrammed directly into these bins; no later rebinning is done.

ii.
```python
OFF_START, OFF_END = -2.5, 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. These values are explicitly required by the decoder task. The agent uses bin centers for continuous aligned covariates.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and the retained trial's go-cue time. The selected tone is the last sample start preceding the go cue.

ii.
```python
sample_start = _events(nwb, 'sample_start_times')
tone_pos = np.searchsorted(sample_start, go, side='right') - 1
tone_time = sample_start[tone_pos]
```

iii. Early licks replay the sample epoch, so the notes argue that the last preceding sample start is the instruction actually preceding the final delay/go sequence.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The go-relative center of each neural bin is shifted by `go - tone_time`, producing a continuous seconds-since-tone ramp.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. The agent follows the explicit requirement that this input be continuous and time-varying; its median tone-to-go interval sanity check is 1.85 s.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact centers of the same go-cue-relative 50 ms bins used for neural firing rates.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. Shared go times and the shared bin grid make input column `k` correspond to neural column `k`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, plus each retained go time. If the event streams are absent, both are treated as empty.

ii.
```python
ps_start = _events(nwb, 'photostim_start_times')
ps_stop = _events(nwb, 'photostim_stop_times')
```

iii. The agent prefers the timestamped event intervals because they directly state when stimulation was active on the common session clock and correspond to the reference stimulation onset/offset representation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The output is binary per bin. For each trial, stimulation intervals overlapping the extraction window are found; a bin is 1 when overlap exceeds 1 ms. Missing streams or no interval yield zeros, and mismatched start/stop counts cause a warning and all-zero input.

ii.
```python
ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
on[ov > OVERLAP_TOL] = 1.0
```

iii. The 1 ms tolerance avoids marking a post-go bin due only to sub-millisecond timing jitter when stimulation ends at the go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute start/stop times are shifted by the trial's go time and tested against the same relative neural-bin edges.

ii.
```python
for a, b in zip(starts - go_time, stops - go_time):
    ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
```

iii. Since both use the common session clock and identical edges, no resampling offset is needed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `trial_instruction` and `outcome`: a hit takes the instructed side, a miss takes the opposite side, and ignore becomes no lick.

ii.
```python
instr_k = instruction[keep_idx]
out_k = outcome[keep_idx]
choice[hit & (instr_k == 'left')] = CHOICE_LEFT
choice[miss & (instr_k == 'left')] = CHOICE_RIGHT
```

iii. The notes report validating these semantics against first post-go lick direction and finding that ignore trials have no response-window lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are encoded 0 left, 1 right, 2 no lick, then broadcast over all 80 time bins in output row 0.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
np.full(NBINS, choice[j])
```

iii. Choice is a per-trial categorical target; broadcasting lets it share one rectangular output matrix with time-varying tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` column for retained trial indices.

ii.
```python
outcome = np.asarray(trials['outcome'][:])
out_k = outcome[keep_idx]
```

iii. The raw column already contains exactly the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to 0 ignore, 1 miss, 2 hit and are repeated across all bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
```

iii. This is a direct categorical encoding in the order specified by `output_values`; repetition represents its per-trial nature.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` strings for retained trials.

ii.
```python
early_lick = np.asarray(trials['early_lick'][:])
early_k = early_lick[keep_idx]
```

iii. The NWB field explicitly records whether licking occurred during sample/delay, so no event reconstruction is needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` is encoded as 1 and everything else (`'no early'`) as 0, then repeated across 80 bins.

ii.
```python
early_code = (early_k == 'early').astype(np.int64)
np.full(NBINS, early_code[j])
```

iii. This implements the requested no/yes categorical target while retaining early-lick trials that reference analyses normally exclude.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1–2 (y and likelihood) of `BehavioralTimeSeries/Camera0_side_TongueTracking`, together with trial starts and go times.

ii.
```python
ts_obj = bts.time_series['Camera0_side_TongueTracking']
frame_t = np.asarray(ts_obj.timestamps[:], dtype=np.float64)
data = np.asarray(ts_obj.data[:, 1:3], dtype=np.float64)
y = data[:, 0]
visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
```

iii. The notes identify this as the released side-camera DeepLabCut tongue measurement and use likelihood to distinguish protruded/visible tongue from tracker output while occluded.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are assigned to raw trials, restricted to retained trials and the extraction window, filtered at likelihood greater than 0.9, and averaged within each trial's 50 ms bins using `bincount`. Empty bins remain NaN before categorization.

ii.
```python
frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1
rel = frame_t[vis] - go[pos]
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
```

iii. Trial assignment prevents frames from a neighboring trial entering a window. The agent says likelihood is strongly bimodal, making thresholds from 0.1–0.99 nearly equivalent, and uses bin means because that is the quantity being classified.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over all non-NaN 50 ms means from retained trials. Values below the 40th percentile are 0, between inclusive boundaries are 1, above the 60th are 2, and bins without a visible frame are 3.

ii.
```python
p_lo = np.percentile(vals, TONGUE_LOW_PCT)
p_hi = np.percentile(vals, TONGUE_HIGH_PCT)
c = np.ones(v.shape, dtype=np.int64)
c[v < p_lo] = 0
c[v > p_hi] = 2
```

iii. The 40/60 split and session scope follow the task. The agent computes thresholds on binned visible values to match the classified quantity and explicitly represents occlusion rather than imputing it.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frame timestamps are shifted by each retained trial's go cue and assigned to the same `[-2.5, 1.5)` 50 ms bins as neural activity.

ii.
```python
rel = frame_t[vis] - go[pos]
inwin = (rel >= OFF_START) & (rel < OFF_END)
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. Camera and spikes share the NWB absolute clock; the trial gate additionally avoids cross-trial contamination.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/unmapped brain annotation is warned about and mapped to `OtherCortex`; missing tongue series yields all class 3; missing photostim streams yield zeros; mismatched photostim event counts warn and yield zeros. Sessions/trials without usable neural data are dropped. Any exception in a worker is printed and that session is returned as `None` and dropped.

ii.
```python
region_name[unknown] = 'OtherCortex'
...
if bts is None or 'Camera0_side_TongueTracking' not in bts.time_series:
    return out, 0.0
```
```python
except Exception as exc:
    traceback.print_exc()
    return None
```

iii. The agent distinguishes absent measurements (explicit not-visible/zero states) from absent neural recordings (exclusion), uses warnings for recoverable inconsistencies, and documents zero padding outside observed recording intervals as a caveat.

## 10-a. What are the most time-consuming steps of the code?

i. NWB I/O (large spike and video arrays), per-good-unit spike binning, and writing the roughly 12 GB pickle dominate. Sessions are parallelized across processes and timing is collected by setup/neural/input/output stage.

ii.
```python
with ctx.Pool(min(args.nproc, len(files))) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
        ...
```

iii. The notes report 12.6 minutes for conversion and about 2.2 minutes for pickle writing, with 16 workers, and attribute most compute to searches over large spike trains and data access.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. A Python loop remains over good units because each unit has a ragged spike train. Trials within each unit are already vectorized. Photostim still loops over trials and matching intervals; tongue binning is vectorized globally with `bincount`. Assembly and summary also contain small session/trial loops.

ii.
```python
for i, u in enumerate(good):
    st = flat_spikes[starts[u]:ends[u]]
    fr[i] = bin_spikes_rate(st, go)
```
```python
for j, g in enumerate(go):
    ...
    photostim_on[j] = interval_overlap_bins(...)
```

iii. The agent explains that ragged per-unit spike arrays prevent a straightforward single search. It already removed the more expensive per-trial neural and tongue loops through flattened indices/vectorized operations.

## 10-c. What processing does the code repeat multiple times?

i. The fixed grid is computed once, but some repeated work remains: photostim overlap is evaluated trial by trial; output arrays are constructed trial by trial; subject indices use repeated linear `list.index`; and the complete outputs are concatenated again solely to print class summaries after conversion.

ii.
```python
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```
```python
allout = np.concatenate([np.concatenate([o[:, None, :] for o in r['output']], axis=1)
                         .reshape(len(OUTPUT_NAMES), -1) for r in results], axis=1)
```

iii. The notes primarily claim a single pass per session and reuse of module-level bin definitions. The repeated assembly/summary work is small relative to NWB reads and spike binning, though the final full-output concatenation temporarily duplicates output data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads trial stop times and computes hemisphere, detailed region mapping, timing, tongue visibility fractions/percentiles, and extensive metadata not consumed by decoder training. The optional debug path creates plots. The final class-summary concatenation is discarded after printing. Core neural/input/output processing is retained in the pickle.

ii.
```python
trial_stop = np.asarray(trials['stop_time'][:], dtype=np.float64)
hemisphere = np.where(elec_x >= ML_MIDLINE_UM, 'left', 'right')
...
allout = np.concatenate(...)
```

iii. The agent includes these for brain-region fidelity, auditability, sanity checks, and optional processing visualization. They are not required by the decoder itself, but most are useful metadata rather than accidental waste.
