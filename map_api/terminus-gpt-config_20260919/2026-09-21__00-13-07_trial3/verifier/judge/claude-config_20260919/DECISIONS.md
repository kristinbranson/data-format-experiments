# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session, stored under `/app/data/sub-<subject_id>/`. The AI discovers every file with a single recursive glob of `/app/data`, sorts it for determinism, and processes each file exactly once in a sequential loop. Each file is opened once with `pynwb.NWBHDF5IO` inside a `with` block (`h5py` is never imported), and every stream needed — `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, `nwb.acquisition['BehavioralTimeSeries']`, `nwb.subject`, `nwb.identifier`, and the unit→electrode links — is read from that one open handle before it is closed. 174 files are found; 173 produce output (see 2-c).

ii.
```python
DATA_ROOT = Path('/app/data')
...
files = sorted(DATA_ROOT.rglob('*.nwb'))
target = 2 if args.sample else None
print(f'Found {len(files)} NWB files; mode={"sample" if args.sample else "full"}', flush=True)

results=[]; subjects=[]; all_regions=set(); t0=time.perf_counter(); skipped=[]
for p in files:
    make_plot=args.show_processing and len(results)<2
    res=process_session(p, make_plot)
```
```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read()
    unit_inds, classifier_good = selected_unit_indices(nwb.units)
    ...
    trials = nwb.trials
    events = nwb.acquisition['BehavioralEvents'].time_series
```

iii. From CONVERSION_NOTES Step 2/Step 6: "`/app/data` contains 174 NWB files (about 50 GB), organized as one directory per subject and one `behavior+ecephys+ogen.nwb` file per session. All inspection used `pynwb.NWBHDF5IO`; `h5py` was not used." and "Each session is opened, processed, and closed sequentially... Files are read once per conversion session; only compact converted arrays and metadata remain in memory." The one-file-per-session layout makes the directory listing the complete session set, so a glob is sufficient; `pynwb` is mandated by the instructions.

## 1-b. How are the data split into subjects?

i. Each NWB file carries its animal in `nwb.subject.subject_id` (a numeric string such as `'440956'`). That string is recorded per session; at assembly the sorted set of unique ids becomes `subjects`, and `subject_idx` holds each session's index into that list, in the same order as `neural`/`input`/`output`. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject = str(nwb.subject.subject_id)
...
results.append(res); subjects.append(res['subject'])
...
unique_subjects=sorted(set(subjects)); subject_lookup={x:i for i,x in enumerate(unique_subjects)}
...
'subjects':unique_subjects,
'subject_idx':np.asarray([subject_lookup[r['subject']] for r in results],dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 maps "`subject.subject_id` → `subjects`, `subject_idx`; unique sorted subject IDs and per-session index; 28 subjects." Step 2 independently enumerated the per-subject session counts from the files and confirmed 28 animals, matching the data paper's "173 behavioral sessions and 28 mice". The subject directory name is derived from the same id, so no separate grouping step is required.

## 1-c. How are the data split into sessions?

i. No splitting is performed: one NWB file *is* one session. Each session is labelled with `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, encoding mouse name, date, time and session number) and stored in `metadata['session_info']` together with subject, trial count, unit count, unit ids, and the original trial row indices. Session order in the output follows the sorted file list (chronological within subject, because the filename embeds the acquisition timestamp).

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
```
```python
identifier = str(nwb.identifier)
...
info = dict(identifier=identifier, source_file=str(path), subject=subject,
            n_trials=len(go), source_trials=len(trials), original_trial_indices=...,
            ...)
```
```python
'session_info':[r['info'] for r in results],
'skipped_files':skipped,
```

iii. CONVERSION_NOTES Step 2: "one directory per subject and one `behavior+ecephys+ogen.nwb` file per session. All sessions share one trial-table schema and one unit-table schema." The file boundary is therefore the session boundary. Step 4 records the decision to keep 173 of 174 sessions, "Exclude the zero-good-unit session because it cannot form a neural decoder session. This yields 173 usable sessions, matching the paper."

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table, one row per behavioural trial, with no re-derivation of trial boundaries from event streams. The mapping from trials to go cues is asserted to be one-to-one (`len(go_start_times) == len(nwb.trials)`), and the code additionally asserts that consecutive go cues are separated by more than the 4 s extraction window, so per-trial windows never overlap. Per-trial column values (`trial_instruction`, `outcome`, `early_lick`) are read as whole-column arrays and indexed positionally. Row indices of retained trials are stored in `info['original_trial_indices']` so every converted trial can be traced back to its source row.

ii.
```python
trials = nwb.trials
events = nwb.acquisition['BehavioralEvents'].time_series
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == len(trials)
...
original_trial_indices = np.arange(len(go_all), dtype=np.int64)
go = go_all
assert np.all(np.diff(go) > (OFF_END-OFF_START)), 'overlapping requested windows'
```

iii. CONVERSION_NOTES Step 2: "Trial table (one row/trial): `start_time`, `stop_time`, one-based `trial`, `photostim_onset`, ... `outcome`, `auto_water`, and `free_water`" and "Go-start event count equals trial count in every session." Step 10 Check 5 adds: "Minimum go spacing is 4.5827 s, so requested windows do not overlap" and "Trial numbering is one-based in NWB metadata, but conversion uses row indices explicitly and stores original row indices; no one-based subtraction ambiguity remains."

## 1-e. How are trials filtered based on quality controls?

i. No behavioural quality filter is applied — early-lick, `miss`, `ignore`, photostimulated, auto-water and free-water trials are all retained, because three of them are required decoder outputs and one is a required decoder input. The only trial exclusion is a neural-data-availability filter applied *after* binning: a trial is dropped if every selected unit has zero spikes anywhere in its full 4 s window (`rates` all zero). A session is dropped if fewer than 2 trials survive. `obs_intervals` is deliberately **not** used as a trial filter; the function that would have done so (`fully_observed_trial_mask`) remains in the file but is never called, and `trial_keep` is hard-coded to all-True. This removes 3,511 of 94,370 trials in retained sessions (3.7%), leaving 90,859.

ii.
```python
# obs_intervals encode a paper-analysis subset that strongly excludes error
# trials, not general acquisition validity. Preserve all released trial rows;
# true simultaneous acquisition gaps are detected from zero total raw spikes.
trial_keep = np.ones(len(go_all), dtype=bool)
original_trial_indices = np.arange(len(go_all), dtype=np.int64)
go = go_all
```
```python
rates = bin_spikes(nwb.units, unit_inds, go)
# Simultaneous zero spikes across all selected units for the full 4-s window
# indicates an acquisition gap not represented reliably by obs_intervals.
activity_keep = np.any(rates != 0, axis=(1, 2))
excluded_zero_activity_trials = int((~activity_keep).sum())
if activity_keep.sum() < 2:
    return None
rates = rates[activity_keep]
go = go[activity_keep]
... (tone, time_from_tone, photostim, choice, outcome, early, tongue_class,
     y, nearest_like, original_trial_indices all subset identically)
```

iii. Two justifications, both documented. On retaining behavioural classes (Step 4/Step 5 Decision 3): "Published primary analyses excluded early-lick and no-response trials... This task explicitly requires early lick, outcome (including ignore/miss), and photostimulation as decoder variables. Consequently those trial classes must be retained unless data validity fails; this is a justified task-required departure from paper-specific trial exclusions."

On the zero-activity rule, Step 10 Issue 2/3 records an explicit iteration: the AI first filtered by `obs_intervals` (requiring the entire go−2.5…go+1.5 window to lie inside the observation intervals of *every* selected unit), then found that "filtering by these intervals reduced miss outcomes from 16.5% to 0.93%, revealing they encode an analysis subset rather than general recording validity. Rejected that policy, restored all behavioral classes." It instead verified against raw pynwb reads that the residual all-zero trials truly contain zero spikes for all 90–923 classifier-good units over 4 s, concluded that "such simultaneous complete silence across hundreds of units is physiologically implausible and indicates source acquisition gaps", and excluded those. Step 10 Check 4 then confirms the filter is not class-selective: outcome fractions move only from 14.84/16.47/68.69% (raw) to 14.91/16.64/68.45% (converted).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units['spike_times']` (session-absolute seconds, ragged, one entry per unit) for the selected units only, combined with `BehavioralEvents/go_start_times` timestamps, which set the window start of each trial. No other neural representation is used.

ii.
```python
def bin_spikes(units, unit_inds, go):
    starts = go + OFF_START
    ntr, nneu = len(go), len(unit_inds)
    rates = np.zeros((ntr, nneu, N_TIME), dtype=np.float32)
    for k, unit_i in enumerate(unit_inds):
        spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
```
```python
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 1: "Modality is electrophysiology, not calcium imaging; delta-F/F is not applicable. Neural data are spike times converted to firing rates." Step 5 maps "`units.spike_times` for `classification == "good"` and all-true `is_good_trials` → `neural`".

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation, or baseline subtraction. For each selected unit, every spike is assigned to the trial whose window start is the latest one at or before it (`searchsorted(..., 'right') - 1`), spikes falling beyond the 4 s window are discarded, the offset within the window is floored into one of 80 bins, and a single `np.bincount` over the flattened `trial*80 + bin` index accumulates all counts at once. Counts are divided by the 0.05 s bin width to give Hz, stored as `float32`, and the array is transposed per trial to (n_neurons, n_timepoints). Assertions check that all rates are finite and non-negative. Values are exact multiples of 20 Hz, as expected of counts over a 50 ms bin.

ii.
```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    trial_i = np.searchsorted(starts, spikes, side='right') - 1
    valid = trial_i >= 0
    if not valid.any():
        continue
    sp = spikes[valid]
    ti = trial_i[valid]
    rel = sp - starts[ti]
    valid2 = (rel >= 0) & (rel < (OFF_END - OFF_START))
    if not valid2.any():
        continue
    ti = ti[valid2]
    bi = np.floor(rel[valid2] / BIN).astype(np.int64)
    flat = ti * N_TIME + bi
    counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
    rates[:, k, :] = counts.astype(np.float32) / BIN
```
```python
neural = [rates[j] for j in range(len(go))]
assert all(x.shape == (len(unit_inds), N_TIME) for x in neural)
assert np.isfinite(rates).all() and (rates >= 0).all()
```

iii. CONVERSION_NOTES Step 1 identifies the reference `sliding_histogram` which "histograms spike times over a specified interval using bin width/stride and optionally divides counts by bin width to produce Hz"; Step 5 applies the same count/bin-width rate definition but with the task-mandated grid: "Histogram absolute spikes in 80 non-overlapping 50-ms bins spanning go−2.5 s to go+1.5 s; divide counts by 0.05 s; float32 neuron × time. Task-mandated window/bin replaces reference 40-ms/17-ms sliding analysis bins." Step 6 justifies the vectorisation: "Verified the minimum go-cue spacing is 4.5827 s, greater than the 4.0-s requested window, so windows never overlap. For each unit, spikes are assigned vectorially to the latest trial-window start and accumulated with one `np.bincount`, exactly matching independent non-overlapping histograms."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two stacked per-unit filters. First, `units['classification'] == 'good'` — the verdict of the region-specific spike-sorting QC classifier (69,453 units across the release). Second, an additional requirement that the unit's per-trial validity flag `units['is_good_trials']` be `True` for **every** trial of the session; this removes a further 565 units, leaving 68,888. No thresholds are applied to any individual QC metric (ISI violation, presence ratio, etc.), and the method paper's >2 Hz mean-rate filter is deliberately not applied. A session with zero surviving units is dropped entirely — this removes exactly one file, `SC017_20190216_162508_s4`, whose 1,852 units have no classifier/anatomy labels, reconciling 174 files with the paper's 173 sessions.

ii.
```python
def selected_unit_indices(units):
    """Paper-matched classifier-good units valid on every trial."""
    classification = np.asarray(units['classification'][:]).astype(str)
    candidates = np.flatnonzero(classification == 'good')
    keep = []
    for j in candidates:
        if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
            keep.append(int(j))
    return np.asarray(keep, dtype=np.int64), len(candidates)
```
```python
unit_inds, classifier_good = selected_unit_indices(nwb.units)
if len(unit_inds) == 0:
    return None
```

iii. CONVERSION_NOTES Step 3: "Published dataset uses region-specific logistic-regression QC classifiers trained from blinded manual Phy labels... Therefore the NWB `classification == "good"` field is the direct paper-matched unit inclusion flag. Manual `unit_quality` (`good`/`multi`) is not a substitute for this classifier output." and "The method paper's additional >2-Hz filter was specific to video-to-neural prediction and should not automatically be imposed."

For the extra `is_good_trials` requirement, Step 5 Decision 2: "Start with `classification == 'good'`, then exclude the 565 good units whose `is_good_trials` is false on any trial. This leaves 68,888 units and preserves all 94,370 trials in usable sessions. The alternative of removing every trial invalid for any selected unit would remove 1,569 trials and unnecessarily alter output distributions. Session neuron identity/dimension remains fixed." Step 4 notes the paper/release count discrepancy is documented rather than papered over: "69,453 classifier-good units across all 174 files vs 69,943 good units [in the paper]... The 490-unit difference is attributed to release/version/export differences and cannot be repaired by substituting manual labels."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so no resampling, interpolation, or per-stream offset correction is required. The alignment event is the go cue, read as `BehavioralEvents/go_start_times.timestamps`, one per trial. For each trial the window start is `go + (-2.5)`; spike times are converted to offsets from that start and floored into bins, which places bin *k* at `[go - 2.5 + 0.05k, go - 2.5 + 0.05(k+1))` for every trial and session. The inputs and the tongue output use the identical go-anchored grid (`absolute_centers = go[:,None] + CENTERS_REL[None,:]`), so bin *k* covers the same interval in all streams.

ii.
```python
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```
```python
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
```
```python
starts = go + OFF_START
...
rel = sp - starts[ti]
valid2 = (rel >= 0) & (rel < (OFF_END - OFF_START))
bi = np.floor(rel[valid2] / BIN).astype(np.int64)
```
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
```
```python
'temporal_alignment_event':'auditory go cue onset (BehavioralEvents/go_start_times)',
'off_start':OFF_START,'off_end':OFF_END,
```

iii. CONVERSION_NOTES Step 4: "Use NWB timestamps and one-to-one go events for alignment"; Step 2: "Native spike times and behavioral/event timestamps are on the same session time base. Trial rows and go events provide the mapping for go-aligned extraction." Step 10 Check 2 independently re-counted raw spikes per unit/trial/bin for first/middle/last sessions straight from pynwb and matched the converted Hz values with `np.allclose()`; the `--show-processing` plots put mean rate, time-from-tone, photostim and tongue on the same go-relative axis with a marker at t=0 to make any offset visible.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 per trial, spanning −2.5 s to +1.5 s relative to the go cue. The grid is built once at module level as 81 edges (`np.linspace(-2.5, 1.5, 81)`) with centres at −2.475 … +1.475 s, and reused for every trial, session, and data stream, so every trial has exactly 80 timepoints. Bins are non-overlapping and left-closed/right-open. No rebinning or resampling of an intermediate representation is performed — spikes are histogrammed directly onto the final grid, and `metadata['time_bin_size']` is set to 50.0 ms.

ii.
```python
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```
```python
'time_bin_size':50.0,
'time_bin_centers_seconds':CENTERS_REL.astype(np.float32),
'neural_measure':'firing rate (Hz) from non-overlapping spike-count bins',
```

iii. CONVERSION_NOTES Step 4/Step 5 Decision 4: "Neural binning — Reference: 40-ms sliding window, 17-ms stride for method-paper video prediction. Resolution: Use mandated non-overlapping 50-ms bins over −2.5 to +1.5 s. This is an explicit task override." and "Edges are `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges); centers are −2.475 through +1.475 s (80 points). Left-closed/right-open histogram semantics avoid off-by-one overlap." Step 10 Check 5 re-verified "81 edges produce exactly 80 left-closed 50-ms bins". The verifier reports mean/min/max T = 80 for all 173 sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times.timestamps` (the instruction-tone onsets) together with each trial's go cue. Because an early lick replays the sample epoch, a trial can contain several sample onsets; the AI takes the **last** sample onset at or before the go cue as that trial's tone. `trials['start_time']` is used only to assert the chosen tone falls inside its trial.

ii.
```python
sample = np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64)
sample_i = np.searchsorted(sample, go, side='right') - 1
assert np.all(sample_i >= 0)
tone = sample[sample_i]
trial_start_all = np.asarray(trials['start_time'][:], dtype=np.float64)
trial_start = trial_start_all[trial_keep]
assert np.all((tone >= trial_start) & (tone <= go))
```

iii. CONVERSION_NOTES Step 5 Decision 5: "Early-lick replay creates 10,921 extra sample events; selecting the latest sample onset before go maps the final epoch correctly. All mapped onsets fall within trial boundaries." Step 4 flags the same hazard: "sample/delay events can repeat after early licks... Tone onset mapping must account for replay events."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying ramp: for every bin the value is the absolute time of the bin centre minus the trial's tone onset, in seconds. Equivalently, bin centre (relative to go) plus the tone→go interval. It is stored as `float32` in row 0 of the (2, 80) input array. No binarisation, clipping, or normalisation is applied. Across the full dataset the values span −1.5 to 11.9 s (the long tail coming from trials whose sample epoch was replayed several times, delaying the go cue).

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```
```python
inputs = [np.stack((time_from_tone[j], photostim[j]), axis=0).astype(np.float32)
          for j in range(len(go))]
assert all(x.shape == (2, N_TIME) for x in inputs)
```
```python
'input_names':['time from tone onset','photostimulation on'],
```

iii. CONVERSION_NOTES Step 5 Decision 5: "The user explicitly requests 'time from tone onset in seconds,' so this is a continuous time-varying ramp, not a binary onset pulse." (The task spec's "if an input is a time such as onset of some stimulus, represent it as a binary time series" is read as applying to onset *events*, whereas this input is explicitly specified as a continuous elapsed time.) Step 9 checks the resulting range: "[−1.525, 11.8943] s — Plausible; includes pre-tone bins and replay-delayed go trials." Step 10 Check 2 independently re-derived the tone and the ramp from raw pynwb timestamps and matched with `np.allclose()`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the centres of the identical go-anchored 50 ms grid used to bin the spikes. `absolute_centers` is literally `go[:, None] + CENTERS_REL[None, :]`, and `CENTERS_REL` is derived from the same `EDGES_REL` that defines the spike bins, so input column *k* and neural column *k* describe the same 50 ms interval of the same trial by construction. No interpolation or offset is involved.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
...
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```

iii. Alignment needs nothing extra because, as CONVERSION_NOTES Step 2 records, "Native spike times and behavioral/event timestamps are on the same session time base." Step 7's plot review confirms it empirically: "Time-from-tone traces increase linearly at slope one" on the same −2.5…+1.5 s go-relative axis as the firing rates, with "No temporal discontinuity or array-length mismatch."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The paired event streams `BehavioralEvents/photostim_start_times.timestamps` and `BehavioralEvents/photostim_stop_times.timestamps`, which give absolute on/off times of each stimulation epoch. The trials-table columns `photostim_onset` / `photostim_duration` / `photostim_power` are *not* used as the source (they are string-valued with `'N/A'` on unstimulated trials); they were only used during exploration to cross-check the event streams. Sessions with no photostimulation simply have empty event arrays.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
assert len(ps) == len(pe)
```

iii. CONVERSION_NOTES Step 5 Decision 6: "Use paired timestamp intervals and evaluate state at bin centers. Nonstimulated trials are all-zero. This captures the late-delay 0.5-s intervention described in the paper without relying on string `N/A` parsing." The mapping table adds: "Event timestamps are preferred to string-valued trial columns after equivalence checks." Step 2 records the cross-check: "Six sessions have zero photostimulation events, which is valid and agrees with the corresponding trial fields." Step 10 Check 5: "All 18,441 photostimulation events in usable sessions pair correctly and last 0.5 s."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary, time-varying indicator rather than a per-trial flag: a bin is 1 if its centre falls inside a stimulation interval `[start, stop)` and 0 otherwise. For each bin centre the code finds the latest stimulation onset at or before it and tests whether the centre is still before that epoch's stop time. Centres preceding the first onset are 0 by construction; if the session has no photostim events at all, the whole array stays 0. Stored as `float32` in row 1 of the input array, range [0, 1].

ii.
```python
photostim = np.zeros((len(go), N_TIME), dtype=np.float32)
ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
assert len(ps) == len(pe)
if len(ps):
    event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
    valid_event = event_i >= 0
    safe_i = np.maximum(event_i, 0)
    photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping row: "Binary 1 where a 50-ms bin center lies in the trial's photostimulation interval, else 0", which follows the task requirement that photostimulation be "whether photostimulation is on at every time point (discrete, time-varying)". Step 10 Check 2 verified it independently: "Input 1: independently tested bin centers against paired photostimulation event intervals, including an actually stimulated bin; matched with `np.allclose()`."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Through the same `absolute_centers` array used for the tone input — the 80 go-anchored 50 ms bin centres per trial. Stimulation times are absolute session times on the shared clock, so they are compared directly against the absolute bin centres with no conversion, and the resulting column *k* matches neural column *k*.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
...
event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. Same rationale as 3-c: one global clock, one shared bin grid. CONVERSION_NOTES Step 7's plot review confirms "stimulation is binary and confined to the expected late-delay interval" on the go-relative axis, consistent with the paper's late-delay 0.5 s intervention.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The actual lick direction is not stored in the file, so it is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). The raw left/right lick event streams were examined during validation but were not used as the source.

ii.
```python
instruction = np.asarray(trials['trial_instruction'][:]).astype(str)[trial_keep]
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. CONVERSION_NOTES Step 5 Decision 7: "Derive actual lick choice from instruction and outcome. A hit means instructed side, a miss means the opposite side, and ignore means no lick. This matches task semantics more robustly than choosing the first among many raw lick events." The mapping table records the cross-validation against the raw lick streams: "Agrees with first response-period lick in 99.65% of raw trials; outcome-derived choice avoids incidental lick-event ambiguity."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0 = left, 1 = right, 2 = no lick. The array is initialised to 2 (no lick), then the four hit/miss × left/right combinations are assigned by boolean mask: hit → instructed side, miss → the opposite side. `ignore` trials fall through and keep code 2. Choice is a single value per trial, which is then tiled across all 80 bins so that all four outputs live in one `(4, 80)` `int8` array. `output_values[0]` is `['left','right','no lick']`.

ii.
```python
choice = np.full(len(go), 2, dtype=np.int64)  # no lick
choice[(outcome_s == 'hit') & (instruction == 'left')] = 0
choice[(outcome_s == 'hit') & (instruction == 'right')] = 1
choice[(outcome_s == 'miss') & (instruction == 'left')] = 1
choice[(outcome_s == 'miss') & (instruction == 'right')] = 0
```
```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
assert all(x.shape == (4, N_TIME) for x in outputs)
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'], ...],
```

iii. CONVERSION_NOTES Step 5: "hit → instructed side; miss → opposite side; ignore → no lick; encode left=0, right=1, no lick=2 ... Per-trial scalar", with the codes taken from the task spec's "Lick direction choice (left, right, no lick, per-trial)". Per-trial values are broadcast over time because the target format requires a single uniform `(n_output, n_timepoints)` array per trial (Step 6: "the three per-trial outputs are repeated across time and tongue category remains time-varying, producing a decoder-compatible 4×80 output matrix"). Final distribution: left 0.429 / right 0.422 / no lick 0.149.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings the task asks for: `'ignore'`, `'miss'`, `'hit'`. No derivation from lick events or reward streams.

ii.
```python
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
```

iii. CONVERSION_NOTES Step 2: "`outcome` values are hit/miss/ignore"; Step 5 maps "`outcome` → `output[1]` outcome ... Per-trial scalar; all required classes retained." Step 3 notes the file's `ignore` corresponds to the paper's "no response" trials: "No response is represented as ignore in the supplied trial table."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit, matching the task spec's ordering, and written into row 1 of the `(4, 80)` output array, tiled across all 80 bins. `output_values[1]` is `['ignore','miss','hit']`. No trials are relabelled or merged.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
```
```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]), ...
```

iii. CONVERSION_NOTES Step 5: "ignore=0, miss=1, hit=2 ... Per-trial scalar; all required classes retained", i.e. the codes follow the task spec's listed order. Step 9/10 use the resulting distribution as a consistency check against the raw release: converted 14.91/16.64/68.45% vs raw 173-session 14.84/16.47/68.69%, confirming the trial filter is not outcome-selective. The dictionary lookup will raise a `KeyError` on any unexpected string, so an unknown outcome value cannot be silently mis-coded.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, which contains the strings `'no early'` and `'early'`.

ii.
```python
early_s = np.asarray(trials['early_lick'][:]).astype(str)[trial_keep]
```

iii. CONVERSION_NOTES Step 2: "`early_lick` supplies early/no-early"; Step 5 maps "`early_lick` → `output[2]` early lick ... retained despite paper-analysis exclusion because it is required output." Step 3 notes the behavioural meaning: "Early licking during sample/delay triggers replay", so the event that sets the flag occurs before the go cue and therefore inside the −2.5 s window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binarised as `early_s == 'early'` → 1, everything else → 0, and written into row 2 of the output array, tiled across all 80 bins. `output_values[2]` is `['no','yes']`. Final distribution: no 0.885 / yes 0.115.

ii.
```python
early = (early_s == 'early').astype(np.int64)
```
```python
outputs = [np.stack((..., np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8) ...]
```
```python
'output_values':[..., ['no','yes'], ...],
```

iii. CONVERSION_NOTES Step 5: "no early=0, early=1 ... Per-trial scalar". Step 12 records an explicit investigation because early lick was the weakest decoded output (0.7483 vs 0.5 chance, i.e. 1.497×): "Raw `early_lick` values were independently checked on first/middle/last NWB sessions and exactly matched converted labels. Both classes have ample support: final fractions are no=0.8846, yes=0.1154 ... the 0.0034 shortfall from exactly 1.5× is statistical/architectural variation, not evidence of a conversion bug."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = tongue_x, tongue_y, tongue_likelihood, with matching `timestamps` (~294 Hz). Column 1 (y) provides the value and column 2 (the DeepLabCut likelihood) decides visibility. Camera0 is used because it is the only tongue stream present in all 174 sessions.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_t = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
y_all, like_all = tongue_data[:, 1], tongue_data[:, 2]
```

iii. CONVERSION_NOTES Step 2: "`acquisition/BehavioralTimeSeries` contains Camera0 jaw, nose, and tongue tracking in every session. Each is `(samples, 3)` with x, y, likelihood and timestamps normally spaced about 0.0034 s ... Camera0 tongue tracking is the uniform source for tongue y and visibility." Step 5 Decision 8: "Camera0 exists uniformly and is the designated side view." Optional whisker/Camera3/lick-port streams exist in only 3–20 sessions and were rejected for that reason.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) Visibility: a frame counts only if `y` and `likelihood` are finite and `likelihood >= 0.9`; the tracker still reports a position when the tongue is retracted, so low-confidence frames must be discarded. (2) Session thresholds: `p40` and `p60` are the 40th/60th percentiles of **all visible raw y samples in that session** (not of binned values, and not over the whole session including invisible frames). If a session has no visible sample at all, both are NaN and every bin becomes class 3. (3) Per bin: one camera sample — the one nearest the bin centre — is looked up and classified. No averaging, smoothing, or interpolation over the bin is performed.

ii.
```python
LIKELIHOOD_THRESHOLD = 0.9
MAX_VIDEO_GAP = 0.010
...
visible_all = np.isfinite(y_all) & np.isfinite(like_all) & (like_all >= LIKELIHOOD_THRESHOLD)
if visible_all.any():
    p40, p60 = np.percentile(y_all[visible_all], [40, 60])
else:
    p40 = p60 = np.nan
```
```python
def nearest_values(query, timestamps, values):
    """Nearest timestamp indices and gaps for sorted timestamps."""
    if len(timestamps) < 2:
        shape = query.shape
        return np.full(shape, np.nan), np.full(shape, np.inf), np.zeros(shape, dtype=np.int64)
    ix = np.searchsorted(timestamps, query)
    ix = np.clip(ix, 1, len(timestamps)-1)
    prev = ix - 1
    choose = np.where(np.abs(timestamps[prev]-query) <= np.abs(timestamps[ix]-query), prev, ix)
    return values[choose], np.abs(timestamps[choose]-query), choose
```

iii. CONVERSION_NOTES Step 5 Decision 9: "Compute p40/p60 separately for each session from all finite Camera0 y samples with likelihood ≥0.9, as explicitly requested 'over the session.' Do not compute thresholds per trial or only within the go window. If no visible samples exist, all tongue outputs are class 3." Decision 8 justifies the confidence cut: "A conservative 0.9 confidence threshold defines visible because no paper/code threshold is specified", and reports "Across usable sessions, 98.0% of centers have temporal coverage and 15.8% are visible at this confidence, consistent with intermittent tongue protrusion."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four-way scheme specified in the task, applied to the nearest-sample y value of each bin: 0 if `y < p40`, 1 if `p40 <= y <= p60`, 2 if `y > p60`, and 3 ("not visible") for any bin that has no usable sample — i.e. where the nearest camera frame is more than 10 ms away, or y/likelihood is non-finite, or the likelihood is below 0.9. The array is initialised to 3 so that "not visible" is the default and no bin can be left unset. `output_values[3]` is `['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']`. Final distribution: 0.062 / 0.032 / 0.065 / 0.841.

ii.
```python
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
nearest_like = like_all[nearest_i] if len(tongue_t) >= 2 else np.full(query.shape, np.nan)
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
tongue_class = np.full(query.shape, 3, dtype=np.int64)
tongue_class[visible & (y < p40)] = 0
tongue_class[visible & (y >= p40) & (y <= p60)] = 1
tongue_class[visible & (y > p60)] = 2
```

iii. The cut points are copied verbatim from the task's "per-session discretization: 0: < 40th percentile ... 3: not visible". CONVERSION_NOTES Step 5 Decision 8: "Missing coverage and low confidence both map to class 3." Step 7's plot review states "tongue category changes only at percentile/visibility boundaries", and Step 10 Check 2 re-derived the thresholds and classes from raw pynwb reads: "independently recomputed session p40/p60 from likelihood≥0.9 samples and nearest-frame visible/not-visible classes; all tested bins matched with `np.allclose()`."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output, so it is the only one needing real alignment. The camera timestamps are on the same session-absolute clock as the spikes and events, so each of the 80 go-anchored bin centres (`absolute_centers`) is used as a query time and the single nearest camera frame is looked up by `searchsorted`. This is point sampling at the bin centre rather than aggregation over the bin: of the ~15 camera frames that fall inside each 50 ms bin, only the one closest to the centre is consulted. A bin is marked visible only if that nearest frame lies within 10 ms of the centre, which also detects video dropouts and the region outside a truncated video stream.

ii.
```python
query = absolute_centers            # go[:,None] + CENTERS_REL[None,:]
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
nearest_like = like_all[nearest_i] if len(tongue_t) >= 2 else np.full(query.shape, np.nan)
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
```
```python
tongue_class = tongue_class[activity_keep]   # subset with the same trial mask as `rates`
```

iii. CONVERSION_NOTES Step 5 Decision 8: "A sample must be within 10 ms of the bin center (greater than the normal ~1.7-ms nearest-frame distance but detects gaps) and likelihood ≥0.9." Step 4 relates it to the reference: "Video alignment in the method code is by trial identity followed by overlap of relative time indices ... Use NWB timestamps and one-to-one go events for alignment." Step 10 Check 3 records the same mapping to `temporal_alignment_embed_and_ephys`. Note that the video is trial-gated in this dataset, so bins before trial start have no frames and fall into class 3 — the notes report 98.0% of bin centres have temporal coverage but only 15.8% are visible.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, each handled explicitly rather than by imputation:
- **Session never quality-controlled** (`SC017_20190216_162508_s4`: `classification`/`anno_name` NaN for all 1,852 units): the `.astype(str)` comparison yields no `'good'` units, `process_session` returns `None`, the file is recorded in `metadata['skipped_files']`, and the session is dropped.
- **Trials with no neural data** (acquisition gaps, free-water trials): detected as all-zero firing rates across every selected unit and dropped; if fewer than 2 trials survive the session is dropped.
- **Units invalid on some trials**: `is_good_trials` false anywhere ⇒ the unit is excluded, keeping the per-session neuron dimension rectangular instead of punching holes in the matrix.
- **Missing / low-confidence / out-of-coverage tongue frames**, including the truncated video session `SC066_20210413_112028_s6` (13,995 samples for 550 trials): assigned the explicit `'not visible'` class 3; `nearest_values` also guards the degenerate case of fewer than 2 timestamps by returning NaN/inf.
- **Electrode `location` not valid JSON**: falls back to the raw string; a unit with no linked electrode row becomes `'unknown'`.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
candidates = np.flatnonzero(classification == 'good')
...
unit_inds, classifier_good = selected_unit_indices(nwb.units)
if len(unit_inds) == 0:
    return None
```
```python
activity_keep = np.any(rates != 0, axis=(1, 2))
if activity_keep.sum() < 2:
    return None
```
```python
if len(timestamps) < 2:
    shape = query.shape
    return np.full(shape, np.nan), np.full(shape, np.inf), np.zeros(shape, dtype=np.int64)
```
```python
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
tongue_class = np.full(query.shape, 3, dtype=np.int64)
```
```python
try:
    loc = str(json.loads(loc).get('brain_regions', loc))
except (json.JSONDecodeError, TypeError):
    pass
out.append(loc)
```

iii. CONVERSION_NOTES Step 5 Decision 10: "`SC066_20210413_112028_s6` has only 13,995 tongue samples; neural/task trials remain valid, and out-of-coverage bins become class 3 rather than dropping the session." Decision 1 covers the unlabelled session: "exclude `SC017_20190216_162508_s4`, whose 1,852 units all have missing classifier/anatomy labels. This exactly reconciles the 174 files with the paper's 173 sessions." Decision 2 explains preferring unit exclusion over trial exclusion for `is_good_trials`: "The alternative of removing every trial invalid for any selected unit would remove 1,569 trials and unnecessarily alter output distributions. Session neuron identity/dimension remains fixed." The general principle recorded in Step 10 is that data which was never recorded is dropped, while a measurement that legitimately has no value (retracted/untracked tongue) becomes an explicit category: "Missing coverage and low confidence both map to class 3 ... uncovered bins are class 3, never fabricated."

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion takes 264.8 s for 174 files (~1.5 s/session, range 0.89–3.02 s) plus 11.8 s to pickle the 11.76 GB result — comfortably inside the 15-minute budget. Profiling a representative session (SC015_s1, 1.37 s total) shows the actual ordering: `coarse_regions` 0.58 s (42%), `bin_spikes` 0.33 s (24%), NWB open+read 0.20 s (15%), tongue array read 0.05 s, `selected_unit_indices` 0.02 s. The dominant cost is therefore the per-unit brain-region lookup, which dereferences a pandas DataFrame from the electrodes table once per unit — this is *not* the bottleneck the AI identified. The AI's notes name spike histogramming and file I/O instead; those are real but second and third.

ii.
```python
def coarse_regions(nwb, unit_inds):
    """Read coarse lateralized recording target through unit electrode links."""
    out = []
    for j in unit_inds:
        electrode_rows = nwb.units['electrodes'][int(j)]  # pynwb-dereferenced DataFrame
        ...
```
```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
```
```python
print(f'[{i}] {inf["identifier"]}: trials={inf["n_trials"]} neurons={inf["selected_units"]} '
      f'visible={inf["tongue_visible_fraction"]:.3f} time={res["elapsed"]:.2f}s', flush=True)
```

iii. CONVERSION_NOTES Step 6 lists the inefficiencies it anticipated — "Naive nested trial × unit calls to `np.histogram` would perform tens of millions of Python operations" and "Reopening NWB files for each stream or retaining full NWB objects would add substantial I/O/memory overhead" — and Step 7 gives the measured estimate: "Session conversion 0.94–1.34 s in sample ⇒ about 3–5 min for 173 sessions ... conservatively <10 min, below the 15-min optimization threshold." Step 9 reports the realised figure: "Completed in 246.29 s (4.10 min), including a 12.23-s pickle write" (264.8 s in the final rerun). The AI printed per-session timings to find bottlenecks as instructed, but never attributed cost to `coarse_regions`.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops over units remain, and two of them are avoidable:
- `coarse_regions` — one pandas DataFrame dereference and one `json.loads` per unit. The electrodes table and the unit→electrode index could be read once per session as arrays and mapped with a single vectorised lookup; this alone is ~40% of runtime.
- `selected_unit_indices` — one ragged HDF5 read of `is_good_trials` per candidate unit. The ragged target buffer plus its offsets could be read once and reduced with `np.minimum.reduceat`.
- `bin_spikes` — one `units['spike_times'][i]` read per unit. This is inherently ragged, but reading the flat `spike_times` target buffer once and slicing it with the offset array (as the reference does) avoids ~400 separate HDF5 reads per session.

Inside `bin_spikes` the per-trial and per-bin dimensions *are* already fully vectorised via `searchsorted` + a single `bincount`, which was the main optimisation the AI applied.

ii.
```python
for j in unit_inds:
    electrode_rows = nwb.units['electrodes'][int(j)]  # pynwb-dereferenced DataFrame
```
```python
for j in candidates:
    if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
        keep.append(int(j))
```
```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    trial_i = np.searchsorted(starts, spikes, side='right') - 1
    ...
    counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
    rates[:, k, :] = counts.astype(np.float32) / BIN
```

iii. CONVERSION_NOTES Step 6 documents the vectorisation the AI *did* do: "Verified the minimum go-cue spacing is 4.5827 s ... For each unit, spikes are assigned vectorially to the latest trial-window start and accumulated with one `np.bincount`, exactly matching independent non-overlapping histograms" and "Session-level timestamp grids, nearest video indices, and event vectors are computed with NumPy search/sort operations." The remaining per-unit loops are not discussed; because the total runtime landed at 4.4 minutes, well under the 15-minute threshold, no further optimisation was pursued (Step 7: "conservatively <10 min, below the 15-min optimization threshold").

## 10-c. What processing does the code repeat multiple times?

i. Essentially nothing is recomputed within a run: each NWB file is opened once inside a single `with` block, each stream is read once into a NumPy array, the bin grid (`EDGES_REL`, `CENTERS_REL`) is built once at module level, `absolute_centers` is computed once per session and reused by both inputs and the tongue lookup, and `nearest_values` is called once per session. The only genuinely repeated work is inside `bin_spikes`, where each unit allocates and zeroes a fresh `n_trials × 80` `bincount` array (up to 64,000 elements × ~400–900 units per session) and writes it into a strided slice `rates[:, k, :]` — the allocation, not the arithmetic, is repeated.

Across runs, the AI did repeat the entire full conversion several times (Step 10 iterations 2 and 3) as it revised the trial-filtering policy, but that is the instructed iteration protocol rather than redundancy in the script.

ii.
```python
counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
rates[:, k, :] = counts.astype(np.float32) / BIN
```
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
...
event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
...
query = absolute_centers
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
```

iii. CONVERSION_NOTES Step 6: "Each session is opened, processed, and closed sequentially ... Files are read once per conversion session; only compact converted arrays and metadata remain in memory", and the listed speedup "One NWB open/read per session — Avoids redundant 50-GB source scans." Because the tongue percentiles are per-session, they can be computed in the same single pass, so no second pass over the data is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount, all of it cheap but none of it used by the decoder:
- **Dead code**: `fully_observed_trial_mask` is fully implemented (including a per-unit `obs_intervals` loop) but never called — `trial_keep` is hard-coded to all-True. The related `info['excluded_unobserved_trials']` field is therefore always 0, which is misleading provenance.
- **Values kept only for plotting**: `y` (nearest-sample tongue y) and `nearest_like` are computed, trial-subset, and carried in the return value even when `--show-processing` is off; `plot_info` is built only under the flag, but the arrays are subset regardless.
- **Bulky unused metadata written into the pickle**: `info['unit_ids']` and `info['fine_region_labels']` store per-unit ids and 293-way Allen `anno_name` strings for all 68,888 selected units, and `info['original_trial_indices']` stores a row index for all 90,859 trials. `classifier_good`/`excluded_bad_trial_units` and `time_bin_centers_seconds` are likewise audit-only.
- The `x` column of the tongue array is read (the whole `(n_frames, 3)` block is materialised) but never used.

ii.
```python
def fully_observed_trial_mask(units, unit_inds, go):
    """Trials whose complete requested window is in every selected unit's observation intervals."""
    ...
    return keep     # never called
```
```python
y = y[activity_keep]
nearest_like = nearest_like[activity_keep]
original_trial_indices = original_trial_indices[activity_keep]
...
fine_regions = np.asarray(nwb.units['anno_name'][:]).astype(str)[unit_inds]
unit_ids = np.asarray(nwb.units.id[:])[unit_inds].tolist()
```
```python
info = dict(identifier=identifier, source_file=str(path), subject=subject,
            n_trials=len(go), source_trials=len(trials), original_trial_indices=original_trial_indices.tolist(),
            excluded_unobserved_trials=int(len(trials)-trial_keep.sum()),
            ... unit_ids=unit_ids, fine_region_labels=fine_regions.tolist(), ...)
```

iii. The extra metadata is deliberate, and justified in CONVERSION_NOTES as provenance for the sanity checks: Step 5 Decision 11 keeps fine anatomy out of the primary region list but retains it — "fine `anno_name` has 293 Allen labels and is retained in session metadata summaries but is too granular for primary region categories" — and Step 10 Check 2 relies on it: "Original trial mapping: stored `original_trial_indices` correctly maps every converted trial to its raw NWB row after gap exclusion." The dead `fully_observed_trial_mask` function is the residue of the rejected `obs_intervals` policy described in Step 10 Issue 3 ("Rejected that policy, restored all behavioral classes"); the notes do not mention that the function was left behind.
