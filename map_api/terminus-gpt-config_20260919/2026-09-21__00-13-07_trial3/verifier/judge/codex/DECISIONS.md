# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file under `/app/data`, sorts the paths, and processes each file as one session. Each file is opened once with `pynwb.NWBHDF5IO`; trials, units, behavioral events, and tongue tracking are read from the NWB object. Full mode processes all files; sample mode stops after two usable sessions.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
for p in files:
    res=process_session(p, make_plot)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials
    events = nwb.acquisition['BehavioralEvents'].time_series
```

iii. The agent justified this from the observed release layout: 174 NWB files, one per session, organized below subject directories. It explicitly used `pynwb`, as required, and reported 173 usable sessions after excluding the file with no selected units.

## 1-b. How are the data split into subjects?

i. The subject is read from `nwb.subject.subject_id` for each session. After conversion, unique IDs are sorted and each session receives an integer `subject_idx` into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
unique_subjects=sorted(set(subjects))
subject_lookup={x:i for i,x in enumerate(unique_subjects)}
'subject_idx':np.asarray([subject_lookup[r['subject']] for r in results],dtype=np.int64)
```

iii. The notes identify the NWB subject field as the canonical identifier and validate that it produces the expected 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is the sorted path order, and `nwb.identifier` plus source path and summary information are stored in metadata.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
identifier = str(nwb.identifier)
info = dict(identifier=identifier, source_file=str(path), subject=subject, ...)
```

iii. The agent found that the release contains one behavior/ephys NWB file per session, so no inferred session boundary is needed.

## 1-d. How are the data split into trials?

i. Rows in `nwb.trials` define trials. The agent asserts a one-to-one correspondence with `go_start_times`; trial-indexed neural, input, and output arrays are then made from that ordering.

ii.
```python
trials = nwb.trials
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == len(trials)
original_trial_indices = np.arange(len(go_all), dtype=np.int64)
```

iii. The notes report that every session has one go event per trial, whereas sample/delay events may repeat after early licks, making the trials table plus go-event ordering the unambiguous mapping.

## 1-e. How are trials filtered based on quality controls?

i. Initially all trial rows are retained. After spike binning, a trial is removed if every selected unit has zero spikes throughout the full four-second window. A session is removed if fewer than two trials remain. Early-lick, ignore, miss, photostimulation, auto-water, and free-water classes are otherwise retained.

ii.
```python
trial_keep = np.ones(len(go_all), dtype=bool)
rates = bin_spikes(nwb.units, unit_inds, go)
activity_keep = np.any(rates != 0, axis=(1, 2))
if activity_keep.sum() < 2:
    return None
rates = rates[activity_keep]
```

iii. The agent argued that retaining behavioral classes is necessary because they are requested inputs/outputs. It interpreted simultaneous all-unit silence as an acquisition gap and preferred this over `obs_intervals`, which it believed encoded an analysis subset. Its final notes report 3,511 excluded trials and no all-zero matrices.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each selected unit's `units['spike_times']`, aligned with `BehavioralEvents/go_start_times`. Unit selection also uses `classification` and `is_good_trials`.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. The agent identified raw spike times as the appropriate electrophysiology representation and the classifier label as the paper-matched QC verdict.

## 2-b. How is the `neural` data processed?

i. For each unit, spikes are assigned to the latest nonoverlapping trial window, converted to one of 80 bins, counted with `np.bincount`, and divided by 0.05 s to produce float32 firing rates in Hz. The final per-trial matrix is neuron by time; there is no smoothing or normalization.

ii.
```python
trial_i = np.searchsorted(starts, spikes, side='right') - 1
rel = sp - starts[ti]
bi = np.floor(rel[valid2] / BIN).astype(np.int64)
flat = ti * N_TIME + bi
counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
rates[:, k, :] = counts.astype(np.float32) / BIN
```

iii. The notes say this is equivalent to per-trial histograms and matches the reference's count/bin-width definition of firing rate while applying the task-required 50-ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit must have `classification == 'good'` and its entire `is_good_trials` vector must be true. Sessions with no such units are dropped. This removes 565 classifier-good units and leaves 68,888 units.

ii.
```python
candidates = np.flatnonzero(classification == 'good')
keep = []
for j in candidates:
    if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
        keep.append(int(j))
```

iii. The agent considered `classification` the direct paper QC flag and chose unit-wise exclusion to preserve fixed neuron dimensions without dropping every trial invalid for any unit. It rejected the analysis-specific 2-Hz cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows start at each absolute go time minus 2.5 s. Spikes are expressed relative to that start and binned through go plus 1.5 s.

ii.
```python
starts = go + OFF_START
trial_i = np.searchsorted(starts, spikes, side='right') - 1
rel = sp - starts[ti]
```

iii. The agent states that spikes and events share the same session clock, so adding go-relative offsets is sufficient and no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins spanning -2.5 to +1.5 s around the go cue. Raw point-process spikes are histogrammed directly; no further rebinning is applied.

ii.
```python
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
N_TIME = 80
```

iii. The agent explicitly treated the requested 50-ms resolution as overriding the method paper's 40-ms windows/17-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` as tone onsets and `go_start_times` to associate the last tone at or before each go cue with a trial.

ii.
```python
sample = np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64)
sample_i = np.searchsorted(sample, go, side='right') - 1
tone = sample[sample_i]
```

iii. Because early licking can replay the sample, the agent intentionally selected the last sample onset before go rather than assuming one sample event per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Absolute neural-bin centers are formed from go time plus relative centers, then the chosen tone timestamp is subtracted. Values are stored as float32 seconds.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```

iii. The notes describe the result as a slope-one trace of elapsed time from the last valid tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same 80 go-relative bin centers used by the neural histogram.

ii.
```python
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
absolute_centers = go[:, None] + CENTERS_REL[None, :]
```

iii. The shared absolute-center grid was used as the alignment guarantee and was independently sanity-checked by the agent.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from paired timestamp arrays `BehavioralEvents/photostim_start_times` and `photostim_stop_times`.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
assert len(ps) == len(pe)
```

iii. The agent preferred timestamped event intervals over the string-valued trial columns and reports verifying all usable-session events as paired 0.5-s intervals.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For every bin center, the latest stimulation onset is located. The value is 1 if an onset exists and the center precedes its paired stop; otherwise it is 0, stored as float32.

ii.
```python
event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
valid_event = event_i >= 0
safe_i = np.maximum(event_i, 0)
photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. The notes justify a binary time series at bin centers as capturing the late-delay intervention and naturally producing all-zero nonstimulated trials.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are evaluated at `absolute_centers`, the same go-relative bin centers used for all trial streams.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. The agent independently compared selected bins against raw interval timestamps and reported exact agreement.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
instruction = np.asarray(trials['trial_instruction'][:]).astype(str)[trial_keep]
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. The agent notes there is no direct choice column but that instruction plus outcome uniquely determines the requested actual choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are coded 0 left, 1 right, and 2 no lick, then repeated across all 80 timepoints in the output matrix.

ii.
```python
choice = np.full(len(go), 2, dtype=np.int64)
choice[(outcome_s == 'hit') & (instruction == 'left')] = 0
choice[(outcome_s == 'hit') & (instruction == 'right')] = 1
choice[(outcome_s == 'miss') & (instruction == 'left')] = 1
choice[(outcome_s == 'miss') & (instruction == 'right')] = 0
```

iii. The coding follows the requested categories; repetition creates a uniform time-shaped output accepted by the decoder.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `nwb.trials['outcome']`.

ii.
```python
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. The raw column already contains exactly `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, 2 hit and the per-trial value is repeated across 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
```

iii. The map follows the requested category order and repetition permits all outputs to share one matrix shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `nwb.trials['early_lick']`.

ii.
```python
early_s = np.asarray(trials['early_lick'][:]).astype(str)[trial_keep]
```

iii. The agent found that the trial table explicitly stores `early` and `no early`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Equality with `'early'` is converted to integer 1; all `'no early'` entries become 0. The value is repeated across 80 bins.

ii.
```python
early = (early_s == 'early').astype(np.int64)
```

iii. This directly implements the requested no/yes coding and uniform output layout.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: column 1 for tongue y, column 2 for tracking likelihood, and the series timestamps for temporal matching.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_t = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
y_all, like_all = tongue_data[:, 1], tongue_data[:, 2]
```

iii. The agent selected Camera0 because it is the uniform side-camera tongue source in all sessions and includes an explicit likelihood channel.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session thresholds are computed from raw finite y samples with likelihood at least 0.9. At every neural bin center, the nearest video sample is selected; it is visible only if it is within 10 ms, finite, and has likelihood at least 0.9. Unavailable or low-confidence samples become category 3.

ii.
```python
visible_all = np.isfinite(y_all) & np.isfinite(like_all) & (like_all >= LIKELIHOOD_THRESHOLD)
p40, p60 = np.percentile(y_all[visible_all], [40, 60])
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
```

iii. No paper threshold was found, so the agent chose a conservative 0.9 likelihood threshold and a 10-ms maximum gap to detect missing/truncated video without dropping otherwise valid neural trials.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of visible raw y samples define category 0 for y below p40, 1 for p40 through p60 inclusive, 2 above p60, and 3 when not visible.

ii.
```python
tongue_class = np.full(query.shape, 3, dtype=np.int64)
tongue_class[visible & (y < p40)] = 0
tongue_class[visible & (y >= p40) & (y <= p60)] = 1
tongue_class[visible & (y > p60)] = 2
```

iii. The per-session percentile scope follows the instruction. The agent chose raw visible samples (rather than binned means) and explicitly handles a session with no visible samples by leaving all values in class 3.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The nearest camera frame to each absolute neural-bin center is used, subject to the 10-ms gap check.

ii.
```python
query = absolute_centers
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
```

iii. The agent relied on shared NWB timestamps and viewed center sampling as direct alignment. Truncated or absent temporal coverage is represented as not visible rather than interpolated.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no selected units is skipped; units with any false `is_good_trials` flag are removed; all-zero neural trials are removed; fewer-than-two-trial sessions are skipped; missing/low-confidence/out-of-range tongue frames become class 3; and no-visible-frame sessions get NaN thresholds and therefore all class 3. Assertions check event counts, tone placement, nonoverlapping windows, shapes, and finite nonnegative rates.

ii.
```python
if len(unit_inds) == 0:
    return None
if activity_keep.sum() < 2:
    return None
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & ...
tongue_class = np.full(query.shape, 3, dtype=np.int64)
assert np.isfinite(rates).all() and (rates >= 0).all()
```

iii. The agent's policy was to avoid fabricating measurements: omit objectively unusable neural data, but retain valid neural trials when only video is missing and represent that absence with the requested not-visible class.

## 10-a. What are the most time-consuming steps of the code?

i. The main costs are opening/reading 174 large NWB files, reading and binning ragged spike trains for tens of thousands of units, reading tongue arrays, retaining the roughly 12-GB neural payload, and serializing the final pickle. The full run took about 265 s and produced an 11.760-GB pickle.

ii.
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes anticipated I/O, per-unit histogramming, and final storage as dominant and therefore processed/closed one session at a time and used compact dtypes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-selection loop, spike-binning loop, region lookup loop, file/session loop, and trial-to-list construction loops remain. Trials are already vectorized within each unit; fully vectorizing ragged spike trains or NWB objects is difficult. The output/list loops could be replaced by bulk arrays followed by views, and unit selection could use a vectorized reduction if the ragged flags were materialized.

ii.
```python
for j in candidates:
    if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
for k, unit_i in enumerate(unit_inds):
for j in range(len(go))
```

iii. The agent explicitly says spike assignment was vectorized per unit across trials with `searchsorted`/`bincount`; it kept unavoidable ragged-unit and session loops while favoring clarity and bounded intermediates.

## 10-c. What processing does the code repeat multiple times?

i. Per trial, it repeatedly allocates constant arrays for choice, outcome, and early lick, stacks inputs/outputs, and extracts neural trial views. Per unit it repeatedly converts ragged spike and validity data; region/electrode lookup is another unit-wise pass. Optional plotting rereads already-computed arrays but only when requested.

ii.
```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
neural = [rates[j] for j in range(len(go))]
```

iii. The agent's notes emphasize that each NWB file and each derived scientific quantity is computed only once; the remaining repetition is format construction necessitated by the nested-list target.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `fully_observed_trial_mask` is defined but never called, and `Counter` is imported but unused. `EDGES_REL` is only needed to obtain centers. The code also computes/stores fine region labels, unit IDs, raw nearest tongue y/likelihood, original trial indices, and detailed session diagnostics that the decoder does not consume; raw y/likelihood are discarded unless plotting is enabled, while the rest remain metadata. Optional plots are diagnostic rather than downstream decoder inputs.

ii.
```python
from collections import Counter
def fully_observed_trial_mask(units, unit_inds, go):
    ...
fine_regions = np.asarray(nwb.units['anno_name'][:]).astype(str)[unit_inds]
unit_ids = np.asarray(nwb.units.id[:])[unit_inds].tolist()
```

iii. The agent justified the diagnostics as traceability, validation, and sanity-check support. It did not identify the unused helper/import as necessary conversion work.
