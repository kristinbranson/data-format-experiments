# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `sub-*/*.nwb` file, sorted the paths, opened each once with `h5py`, converted it, and assembled retained sessions into the output dictionary.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
for index, path in enumerate(paths, start=1):
    session = _convert_session(path)
```

iii. The trajectory says it audited the NWB schema and all 174 files, then retained 173 because one lacked classifier-labelled good units. It regarded the one-file-per-session NWB layout as the complete dataset boundary.

## 1-b. How are the data split into subjects?

i. Subject is inferred from each file's parent directory (`sub-<id>`), then unique IDs are sorted and each session receives an integer `subject_idx`.

ii.
```python
subject_id = path.parent.name.removeprefix("sub-")
subjects = sorted({session["subject"] for session in converted_sessions})
subject_idx = np.asarray([subject_lookup[s["subject"]] for s in converted_sessions])
```

iii. The agent's audit found 28 mice and treated the DANDI directory identifier as the canonical subject grouping.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; output session order is sorted path order. A session with no good units is skipped.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
session = _convert_session(path)
if session is None:
    skipped_sessions.append(path.name)
```

iii. The trajectory explicitly reports 174 files and 173 retained sessions, consistent with one NWB per recording session.

## 1-d. How are the data split into trials?

i. Trial rows are indexed by `intervals/trials/id`; the code requires one go cue per source trial and carries source indices through filtering. Neural, input, and output lists then contain one array per retained trial.

ii.
```python
n_source_trials = len(nwb["intervals/trials/id"])
if len(all_go_times) != n_source_trials: raise ValueError(...)
valid_trial_idx = common_good_idx[~water_mask]
"neural": [rates[t] for t in range(n_trials)]
```

iii. The agent checked the one-to-one go-cue/trial mapping and used source indices to handle sessions whose behavior table extended beyond ephys acquisition.

## 1-e. How are trials filtered based on quality controls?

i. It intersects `is_good_trials` across all retained units, maps compact ephys columns to source trials using common observation intervals, excludes both auto- and free-water trials, and drops any remaining population-wide all-zero trial. Early-lick, ignored, and photostimulation trials are retained. It errors if fewer than two remain.

ii.
```python
common_good_mask = np.all(good_trial_matrix, axis=0)
water_mask = auto_water_all[common_good_idx] | free_water_all[common_good_idx]
valid_trial_idx = common_good_idx[~water_mask]
population_recorded = np.any(rates != 0, axis=(1, 2))
```

iii. The trajectory says validator warnings revealed behavior outside acquisition and differing probe coverage, motivating common `is_good_trials`. It cites the repository's regular-trial mask for water exclusions and treats population silence as a recording dropout. Task-required early/no-response/photo trials were deliberately preserved.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and its index, selected by `units/classification == "good"`; go-cue timestamps define trial windows.

ii.
```python
classification = _decode_array(nwb["units/classification"])
good_rows = np.flatnonzero(classification == "good")
all_spikes = nwb["units/spike_times"][:]
spike_ends = nwb["units/spike_times_index"][:]
```

iii. The agent says source papers/code established spike times and classifier-labelled good units as the published representation and QC convention.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes are assigned to a trial window and 50-ms bin with `searchsorted`, `floor`, and `bincount`; counts are divided by 0.05 to produce float32 firing rates in spikes/s. No smoothing or normalization is applied.

ii.
```python
time_bin = np.floor((spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10).astype(np.int64)
counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
rates[:, out_unit, :] = counts.reshape(n_trials, N_BINS) / BIN_SIZE_S
```

iii. The trajectory reports exact cross-checks against NumPy histograms and notes that source code defines rate as spike count divided by bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose published `classification` is `good` are retained; a session with none is skipped, and every retained good unit must have an anatomical annotation.

ii.
```python
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0: return None
if np.any(annotations == ""): raise ValueError(...)
```

iii. The agent attributes this choice to the region-specific published QC classifiers and reports 69,453 units in 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue timestamps define windows from -2.5 to +1.5 s; session-absolute spikes are placed directly in those windows.

ii.
```python
window_starts = go_times + OFF_START
window_stops = go_times + OFF_END
trial = np.searchsorted(window_starts, spikes, side="right") - 1
```

iii. The agent states spikes, video, events, and trial intervals share the NWB clock, so no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins across four seconds. Raw spike events are binned once; there is no subsequent rebinning.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1) * BIN_SIZE_S
```

iii. This directly follows the requested temporal window and resolution; the trajectory confirms centers from -2.475 to +1.475 s.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, per-trial go-cue time, and neural-bin centers. The last sample/tone onset before each go cue is selected.

ii.
```python
idx = np.searchsorted(sample_starts, go_times, side="right") - 1
tone_onset = sample_starts[idx]
```

iii. The agent notes early licks can replay the sample epoch, so the final preceding tone is behaviorally relevant.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each absolute bin center is expressed as elapsed seconds since that trial's selected tone onset.

ii.
```python
time_from_tone = (go_times[trial] + BIN_CENTERS - tone_onset[trial]).astype(np.float32)
```

iii. This is the direct clock difference; no further transformation is justified or applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact centers of the same go-aligned 50-ms bins as neural activity.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = go_times[trial] + BIN_CENTERS - tone_onset[trial]
```

iii. The shared go-cue clock and bin-center grid guarantee elementwise alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial `photostim_onset`, `photostim_duration`, and `start_time`, plus go time and bin centers.

ii.
```python
stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
photo_start = trial_start[trial] + stim_onset[trial]
```

iii. The agent recognized onset as trial-start-relative text with missing values and converted it to the common absolute clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Optional text fields are parsed to floats/NaN. A binary vector is one for centers in the half-open stimulation interval and zero otherwise.

ii.
```python
photo_on = np.zeros(N_BINS, dtype=np.float32)
photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. This supplies the requested time-varying on/off input; NaN trials naturally remain all zero.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-relative stimulation onset is made absolute using trial start, then compared with the same absolute go-aligned bin centers used for neural data.

ii.
```python
absolute_centers = go_times[trial] + BIN_CENTERS
photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. The shared NWB clock makes this direct comparison sufficient.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `trial_instruction` and `outcome`: hit means instructed side, miss means opposite, ignore means no lick.

ii.
```python
if outcome == "ignore": return 2
if outcome == "hit": return 0 if instruction == "left" else 1
if outcome == "miss": return 1 if instruction == "left" else 0
```

iii. The agent calls these authoritative trial labels and says this avoids rare missing or mistimestamped lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded 0=left, 1=right, 2=no lick and repeated over all 80 bins.

ii.
```python
trial_output[0, :] = _trial_choice(instruction[trial], outcome_text[trial])
```

iii. It is a per-trial categorical target; repetition lets it share a time-indexed output array with tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome_text = _decode_array(nwb["intervals/trials/outcome"])[valid_trial_idx]
```

iii. The raw field already provides exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to 0=ignore, 1=miss, 2=hit and are repeated across bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
trial_output[1, :] = outcome_map[outcome_text[trial]]
```

iii. The mapping matches `output_values`; repetition represents a trial-level target.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early_text = _decode_array(nwb["intervals/trials/early_lick"])[valid_trial_idx]
```

iii. The trial table explicitly records the requested label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` maps to 1 and everything else (the expected `no early`) to 0, repeated across bins.

ii.
```python
trial_output[2, :] = 1 if early_text[trial] == "early" else 0
```

iii. This implements no/yes as a trial-level categorical target, although unexpected strings are silently treated as no.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses columns 1 (y) and 2 (DLC likelihood) of side-camera `Camera0_side_TongueTracking/data`, plus its timestamps and go cues.

ii.
```python
y = tongue_data[:, 1]
likelihood = tongue_data[:, 2]
camera_times = nwb[f"{tongue_path}/timestamps"][:]
```

iii. The agent identified this as the session's tongue trace and used likelihood to represent visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with finite y and likelihood at least 0.9 are visible. Session cut points are percentiles of visible raw frames. Each neural-bin center samples the preceding camera frame, rejecting stale frames over 20 ms; invalid samples become category 3.

ii.
```python
visible_session = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= 0.9)
q40, q60 = np.percentile(y[visible_session], [40, 60])
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame &= (target_times - camera_times[frame]) <= 0.020
```

iii. The agent says preceding-frame sampling follows repository marker alignment, chose 0.9 as visible, and added a 20-ms gap guard. This differs from averaging video within neural bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th/60th percentiles of visible raw-frame y define 0 below q40, 1 from q40 through q60, 2 above q60, and 3 not visible.

ii.
```python
categories[visible & (sampled_y < q40)] = 0
categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
categories[visible & (sampled_y > q60)] = 2
```

iii. The percentile boundaries and per-session scope follow the task; the agent made raw visible frames, rather than 50-ms means, the percentile population.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every go-aligned neural-bin center, the immediately preceding camera frame is selected on the shared clock; a sample more than 20 ms old is unavailable.

ii.
```python
target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
```

iii. The agent states all streams share a clock and cites repository preceding-frame alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN text decodes to empty; a session without good units is skipped. Missing photostim text becomes NaN/all-zero. Ephys coverage is intersected across units; water and all-population-zero trials are removed. Missing/stale/low-confidence tongue samples become class 3, and an all-invisible session gets NaN thresholds.

ii.
```python
if isinstance(value, (float, np.floating)) and np.isnan(value): return ""
return np.nan if text in {"", "N/A", "nan"} else float(text)
categories = np.full(target_times.shape, 3, dtype=np.uint8)
```

iii. The trajectory shows iterative validator-driven investigation of recording gaps, compact masks, water trials, and one residual dropout; missing measurements are either excluded where neural acquisition is absent or explicitly categorized for tongue visibility.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike and camera datasets, per-unit spike binning, constructing the roughly 12-GB in-memory result, and pickling it are the dominant operations. The code itself does not time substeps.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
tongue_data = nwb[f"{tongue_path}/data"][:]
for out_unit, unit_row in enumerate(good_rows): ...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory reports multiple full conversions of a ~12-GB artifact and describes the full 50-GB source conversion; its runtime evidence points to I/O, neural binning, and serialization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit neural loop is necessary for ragged spike trains but could potentially use specialized ragged/event binning. The per-trial construction of inputs/outputs is readily vectorizable; session conversion remains serial. Annotation and subject index comprehensions are minor.

ii.
```python
for out_unit, unit_row in enumerate(good_rows):
    ...
for trial in range(n_trials):
    inputs.append(...)
    outputs.append(...)
```

iii. The agent emphasized vectorizing spike assignment within each unit and validated performance, but supplied no explicit rationale for leaving the simple per-trial loop or sessions serial.

## 10-c. What processing does the code repeat multiple times?

i. It recomputes absolute bin centers and allocates/stacks inputs and outputs once per trial; `_decode_array` repeatedly scans separate trial columns. During development, the trajectory records several complete reconversions after curation changes, though the final script processes each session once.

ii.
```python
for trial in range(n_trials):
    absolute_centers = go_times[trial] + BIN_CENTERS
    inputs.append(np.stack((time_from_tone, photo_on), axis=0))
```

iii. No special justification was given; the final implementation favors straightforward trial-local construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It bins neural rates before using them to identify population-wide zero trials, so rates for those trials are computed and then discarded. It also loads full-session tongue arrays even though only retained trial windows are output, and computes/stores extensive metadata not needed by decoder training.

ii.
```python
rates = _bin_good_units(...)
population_recorded = np.any(rates != 0, axis=(1, 2))
rates = rates[population_recorded]
tongue_data = nwb[f"{tongue_path}/data"][:]
```

iii. The trajectory explains the zero-rate pass as a response to a remaining validator warning. Other discarded work is not explicitly justified beyond enabling session-wide percentiles and provenance.
