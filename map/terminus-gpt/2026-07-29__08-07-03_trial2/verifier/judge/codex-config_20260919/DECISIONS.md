# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file below the relative `data` directory, sorts the paths, and opens each session once with `h5py`. It reads trials, units, subject metadata, electrode locations, and behavioral time series directly from HDF5 groups. `--sample` truncates the list to two files; otherwise all found files are processed.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    paths = paths[:2]
data = build_dataset(paths, show_processing=args.show_processing)
```
```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    units = f['units']
```

iii. The notes say the NWB files are the source of truth because the paper code mainly uses preprocessed arrays and unavailable external paths. The agent documented 174 files across 28 subjects and used the raw NWB structure it had inspected.

## 1-b. How are the data split into subjects?

i. Each retained session gets a subject ID from `general/subject/subject_id`, falling back to the parent directory name. Subjects are accumulated in first-seen (sorted-file) order, and each session receives the corresponding integer index.

ii.
```python
sid = f['general/subject/subject_id'][()]
return _decode(sid)
```
```python
if sid not in subject_to_idx:
    subject_to_idx[sid] = len(subjects)
    subjects.append(sid)
subject_idx.append(subject_to_idx[sid])
```

iii. The agent regarded the NWB subject field as canonical and documented that the dataset contains 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. A session is appended only if at least two trials remain and at least one selected unit exists. The session ID is the file stem.

ii.
```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
    if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
        continue
    neural.append(ntr)
```

iii. The notes identify the NWB layout as one file per behavioral/ephys session. The agent expected 174 sessions, while also noting that the paper reports 173 and leaving that mismatch unresolved.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The code loops over every row and creates one neural, input, and output array for each row that passes its validity and post-binning checks.

ii.
```python
n_trials = len(trials['id'])
for i in range(n_trials):
    if not valid_trials[i]:
        continue
    ...
    neural_trials.append(fr)
```

iii. The agent identified the NWB trials table as the trial source. It did not validate trial rows against recorded go-cue events because it believed those event timestamps were absent.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have recognized instruction, outcome, and early-lick labels; their inferred four-second window must lie between the earliest and latest spike among selected units; and the final binned matrix must contain at least one spike. Sessions with fewer than two surviving trials are discarded. The code does not use `obs_intervals` or exclude `free_water` explicitly.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok
```
```python
if np.all(fr == 0):
    continue
```

iii. The notes say this was added after thousands of all-zero trials appeared because behavior continued beyond recorded ephys. The agent viewed global spike coverage and explicit all-zero removal as fixes, while acknowledging clock/alignment concerns. It retained early-lick and ignore trials because they are decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `spike_times_index`, restricted by `units/unit_quality`. Inferred go-cue times from `trials/start_time` determine the bin edges.

ii.
```python
spikes = units_group['spike_times'][:]
index = units_group['spike_times_index'][:]
```
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

iii. The agent correctly identified spike times as the neural source, but believed `unit_quality == good` implemented paper QC. Its notes explicitly flag that this produces 154,948 units rather than the paper/reference count near 69,943.

## 2-b. How is the `neural` data processed?

i. For every retained trial and good-labeled unit, spikes within the inferred window are histogrammed into non-overlapping bins and divided by 0.05 seconds to produce float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The agent intended to match the required 50-ms firing-rate representation and the reference convention of binned firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps units whose older `unit_quality` field equals `good`; it does not use the classifier verdict in `units/classification`. Sessions with no such units are ultimately skipped.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The notes cite the paper's good-unit rule but concede that the implementation retains 154,948 rather than about 69,943 units and that the mismatch remained unresolved.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not read `BehavioralEvents/go_start_times`. It assumes the go cue is always 1.85 seconds after trial start, adds the requested relative bin edges, and bins spikes against those inferred absolute edges.

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The agent interpreted the methods' 0.65-second sample plus 1.2-second delay as a fixed go offset and stated that explicit go timestamps were absent. The notes later recognized that use of trial-start-derived timing caused serious alignment/coverage concerns but did not replace it with the available event stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins from -2.5 to +1.5 seconds. Raw spike timestamps are rebinned into firing rates on this grid.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
```

iii. This directly follows the decoder instructions, overriding a 40-ms resolution seen in some paper analysis code.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `trials/start_time`, a constant assumed tone offset of zero, the assumed 1.85-second go offset, and bin centers. It does not use `BehavioralEvents/sample_start_times`.

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The agent said explicit tone timestamps were absent and therefore derived tone onset from the documented task timing relative to trial start.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the inferred absolute tone onset is subtracted from the inferred absolute bin center. This produces a continuous float32 ramp (equivalently, relative go-centered bin centers plus 1.85 seconds).

ii.
```python
centers_abs = align + BIN_CENTERS.copy()
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The agent considered simple subtraction sufficient after adopting fixed task timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same inferred go-aligned bin centers as the neural histogram, so array indices correspond, although both are anchored to the wrong inferred event time when actual events differ.

ii.
```python
edges_abs = align + BIN_EDGES
centers_abs = align + centers_rel
time_from_tone = centers_abs - tone_onset_times[i]
```

iii. The agent's plan was to put every stream on the common 50-ms go-centered grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses `trials/photostim_onset`, `trials/photostim_duration`, and `trials/start_time`; inferred go-aligned bin centers are used for rasterization.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
p_start = trial_start[i] + p_on
p_stop = p_start + p_dur
```

iii. The agent identified these trial fields during NWB exploration and treated `N/A` as no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Optional string times are parsed to floats. A length-80 zero vector is set to one where a bin center falls in the half-open stimulation interval `[start, stop)`.

ii.
```python
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. This creates the required binary, time-varying input and naturally leaves non-stimulation trials all zero.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds and neural bin centers are compared on the trial-time clock. Thus the photostim and neural arrays share indices, but the centers inherit the fixed, inferred go-cue alignment.

ii.
```python
centers_abs = align + centers_rel
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The agent intended all streams to share the same go-centered 50-ms grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is taken directly from `trials/trial_instruction` only. Outcome is not used to infer the animal's actual response, and no no-lick category is produced.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The agent described this field as choice in its mapping and documentation. It did not justify the crucial assumption that instructed direction equals chosen lick direction on miss trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left/right strings are mapped to 0/1 and repeated over all 80 time bins. `output_values` contains only left and right.

ii.
```python
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```
```python
['left', 'right'],
```

iii. The agent repeated per-trial categorical outputs so all outputs could share a `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` column.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
```

iii. The raw categories exactly match the requested ignore, miss, and hit output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, and 2 hit, then repeated across 80 bins.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The mapping follows the requested category order and uses the common time-shaped output representation.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `trials/early_lick`.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. The NWB trial column explicitly contains the requested early-lick label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` and `early` are mapped to 0 and 1 and repeated across all bins.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The agent used the requested no/yes coding and a common time-shaped output representation.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It selects a tongue-tracking series under `acquisition/BehavioralTimeSeries`, preferring Camera0, reads timestamps, assumes columns `[x, y, likelihood]`, uses column 1 as y, and masks likelihood below 0.5.

ii.
```python
candidates = [k for k in bts.keys() if 'TongueTracking' in k]
...
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
tongue_y = tongue_data[:, 1].astype(float)
tongue_y[lik < 0.5] = np.nan
```

iii. The agent documented that the tracking convention was assumed and used confidence masking to exclude unreliable positions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Percentiles are computed from all finite raw y frames in the session. For each neural bin center, the code selects the first later valid tracked frame (`searchsorted(..., side='left')`); samples over 100 ms away become NaN. It initializes every bin as middle class 1, so missing/unavailable measurements remain middle rather than becoming `not visible`.

ii.
```python
q40, q60 = np.nanpercentile(tongue_y, [40, 60])
...
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
idx = np.searchsorted(vt, centers_abs, side='left')
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y[far] = np.nan
```

iii. The notes say nearest-valid sampling and the 0.5 likelihood threshold were attempts to improve a strongly imbalanced tongue output. They acknowledge the imbalance but do not recognize that missing values require a fourth class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Raw finite session frames supply 40th/60th percentile thresholds. Values below q40 become 0, values above q60 become 2, and all remaining bins—including missing values—stay 1. Only three category names are declared.

ii.
```python
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```
```python
['lt_40pct', '40_to_60pct', 'gt_60pct'],
```

iii. The agent intended to implement per-session 40/60 percentile discretization, but omitted the explicitly required `not visible` category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Each inferred go-aligned neural bin center is matched to the next valid camera timestamp, provided it is within 100 ms. It does not average camera frames over the same 50-ms intervals used for spikes.

ii.
```python
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
```

iii. The agent aimed to resample behavioral video to the common decoder grid. Its notes describe this as nearest-valid sampling, although the implementation is one-sided rather than truly nearest.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Optional photostim parse failures become no stimulation; missing subject IDs fall back to directory names; missing brain locations become `unknown`; unavailable tongue tracking or thresholds silently yield middle tongue class; invalid trial labels and all-zero neural trials are dropped. Broad `except Exception` blocks hide several metadata errors.

ii.
```python
except Exception:
    return path.parent.name
```
```python
except Exception:
    brain_region_labels = ['unknown' for _ in good_spike_times]
```
```python
else:
    tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

iii. The agent prioritized producing a verifier-valid dataset and documented fixes for missing neural coverage and brain regions. It did not distinguish missing tongue observations as their own required category, and it knew the unit-QC discrepancy remained unresolved.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant computation is the nested trial-by-unit spike filtering and histogramming. Reading full spike arrays, tongue arrays, building the 25-GB output, and pickling it are also costly. The notes measured about 17.7 seconds per session in an early implementation and projected more than 50 minutes before later changes.

ii.
```python
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. The notes explicitly call the per-neuron histogram loop a bottleneck. Full conversion and repeated regeneration were time-consuming, and the final pickle is documented as roughly 25 GB.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-unit neural loop could at least vectorize all trials per unit using one flattened edge grid and `searchsorted`, as the reference does. The unit spike slicing loop and electrode-region loop process ragged arrays and are less important, while tongue alignment is already vectorized across centers within a trial.

ii.
```python
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. The agent itself identified the per-neuron histogram loop as the bottleneck and said optimization was needed, but the final code still repeats it separately for every trial.

## 10-c. What processing does the code repeat multiple times?

i. It computes `go_cue_times` twice, copies the same bin-center vector per trial, scans every unit's full spike array once per trial to form a mask, and invokes a new histogram for each nonempty trial-unit pair. Dataset regeneration and verification were also repeated during debugging, though that is outside the conversion's runtime behavior.

ii.
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```
```python
centers_rel = BIN_CENTERS.copy()
mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
```

iii. The notes record repeated full conversions to eliminate all-zero trials and restore brain-region labels. They noticed performance problems but did not remove these repeated calculations from the final script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It materializes spike-time arrays for every raw unit before discarding non-`good` units, bins complete neural trials and constructs inputs/outputs before discarding all-zero trials, copies bin centers per trial, and calculates `spike_t_min` even though it is normally zero and not a reliable observation boundary. Repeating trial-level labels across 80 bins greatly enlarges the 25-GB pickle, though the target format permits time-shaped outputs.

ii.
```python
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```
```python
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)
if np.all(fr == 0):
    continue
```

iii. The agent focused on verifier compatibility and did not document these as discarded work. Its notes instead emphasize that the large output verified and trained successfully.
