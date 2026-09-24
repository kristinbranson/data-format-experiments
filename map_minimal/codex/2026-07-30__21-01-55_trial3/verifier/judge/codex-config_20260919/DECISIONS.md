# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively enumerates sorted `sub-*/*.nwb` files, opens each once with `h5py`, and reads units, trials, behavioral events, and tongue tracking directly from HDF5 paths. Sessions yielding no usable result are skipped.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```
```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
```

iii. The trajectory says the AI switched from `pynwb` to direct HDF5 because full-dataset scans were too slow. It considered one NWB file to be one session and reported 173 retained sessions after QC.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from each NWB file's parent directory by removing `sub-`; unique IDs are sorted and each session is mapped to its index.

ii.
```python
"subject": path.parent.name.replace("sub-", ""),
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI treated the DANDI directory layout as the subject identity source. Its notes say these directory names supply `subjects`.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Output session order follows the sorted path order; sessions with zero good units or fewer than two retained trials are omitted.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
    if session is not None:
        sessions.append(session)
```

iii. The AI inferred the file boundary as the session boundary and used sorting for reproducibility. It expected one zero-good-unit session to be excluded.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`, but all trial arrays and go cues are truncated to the width of `units/is_good_trials`. Retained trial indices are then used consistently for neural, input, and output lists.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. After observing silent late trials, the AI concluded that `is_good_trials` width represented ephys-covered trials and called truncation a critical alignment correction. It later also dropped fully zero binned trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes both `auto_water` and `free_water`, windows outside camera coverage, and all-zero-neural trials; it retains early-lick, ignore, and photostimulation trials. Sessions with fewer than two survivors are removed.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    continue
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. The AI deliberately retained early/ignore/stimulation trials because they are required targets or inputs, despite paper masks excluding them. It excluded water trials as nonstandard and added camera/all-zero exclusions as mechanical validity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from ragged `units/spike_times` and `spike_times_index`, restricted by `units/classification`, and are positioned using go-cue timestamps.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
```

iii. The AI identified `classification == "good"` as the published QC output and spike times as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` cumulative positions at every absolute bin edge are differenced into counts and divided by 0.05 s to yield Hz. Results are stored as `float16`; there is no smoothing or normalization.

ii.
```python
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. The AI aimed to mirror the paper's binned firing-rate computation while controlling the multi-gigabyte output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose decoded `classification` equals `good` are retained. A session with none is dropped; no additional unit metric or per-trial `is_good_trials` mask is applied.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None, stats
```

iii. The AI concluded this field was the Chen et al. QC classifier and avoided extra thresholds. It noted 69,453 such units in the released files versus 69,943 in the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative bin edges from -2.5 to +1.5 s are added to each absolute go-cue timestamp; absolute spikes are binned against those edges.

ii.
```python
go_abs = go_times[valid_trials]
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
flat_abs_edges = abs_edge_matrix.reshape(-1)
```

iii. The AI treated spikes and event timestamps as sharing the NWB session clock, so no interpolation or clock correction was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It creates 80 nonoverlapping 50-ms bins over `[-2.5, 1.5)` relative to go cue. Raw spike times are binned once; there is no further temporal rebinning.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
```

iii. After testing an endpoint-centered alternative, the AI chose literal exact-width bins because that directly follows the task and avoids unnecessary window loss.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial start, and go cue. The chosen tone is the last sample start between trial start and go cue; a missing tone falls back to `go - 1.85 s`.

ii.
```python
lo = np.searchsorted(sample_starts, trial_start, side="left")
hi = np.searchsorted(sample_starts, go_time, side="right")
return float(sample_starts[hi - 1]), False
```

iii. The AI reasoned that early licks can replay the sample epoch, making the last pre-go sample the behaviorally relevant tone. It instrumented a fallback, though reported that it was never used.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is expressed relative to go cue and subtracted from every go-relative bin center, producing seconds elapsed since tone onset.

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. The AI wanted zero to correspond to tone onset on the same time grid as the firing rates.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the identical 80 go-cue-relative `BIN_CENTERS` used by the neural bin edges.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. The shared grid was explicitly chosen so each input column corresponds to the same neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, `photostim_power`, trial start, and go cue.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])
stim_power = parse_optional_float_array(trials["photostim_power"][()])
```

iii. The AI described onset/duration as defining timing and positive power as confirming an actual stimulation trial.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Valid positive-power events are transformed from trial-relative to go-relative onset/offset. A bin is 1 when its center is in `[onset, offset)`, otherwise 0.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
stim_on = ((BIN_CENTERS >= on_rel) & (BIN_CENTERS < off_rel)).astype(np.float32)
```

iii. The task requests an on/off value at every timepoint, so the AI represented stimulation as a binary time series rather than a trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset and offset are converted to go-cue-relative seconds and compared with the same neural bin centers.

ii.
```python
on_rel = float(stim_onset[trial_idx] - (go_abs - float(trial_start[trial_idx])))
stim_on = ((BIN_CENTERS >= on_rel) & (BIN_CENTERS < off_rel))
```

iii. The AI followed the reference pattern of moving trial-relative stimulation timing onto the go-relative neural axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `left_lick_times`, `right_lick_times`, go cue, and trial stop. It selects the first post-go lick through trial stop; if none exists, it substitutes `trial_instruction`.

ii.
```python
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks, right_licks, go_abs, float(trial_stop[trial_idx]), trial_instruction[trial_idx]
)
```

iii. The AI considered choice ambiguous because the table lacks an explicit reported-side column. It preferred behavior events, but believed the target required a binary label and therefore used instructed side for 13,064 no-lick trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earliest qualifying side is coded left=0 or right=1 and repeated over all 80 bins. Only two output labels are declared; no-lick is not represented.

ii.
```python
return 0, False  # left
return 1, False  # right
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8)
```
```python
["left", "right"],
```

iii. The AI justified repetition because choice is per-trial, but its binary interpretation conflicts with the instruction's explicit no-lick category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from the trials-table `outcome` strings.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

iii. The raw field already has exactly the requested categories, so the AI did not re-derive it.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2, and the code is repeated across the trial's 80 bins; unexpected values raise an error.

ii.
```python
if outcome[trial_idx] == "ignore": outcome_code = 0
elif outcome[trial_idx] == "miss": outcome_code = 1
elif outcome[trial_idx] == "hit": outcome_code = 2
```

iii. The mapping follows the requested order, and repetition puts per-trial outputs into a uniform time-shaped matrix.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

iii. The AI retained these trials specifically because early lick is a required decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other value maps to 0; the result is repeated across 80 bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. The AI used the requested no/yes binary coding and a common time-shaped output format.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking/data` and that series' timestamps. It does not use column 2 tracking likelihood.

ii.
```python
tongue_data = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()])
tongue_y = tongue_data[:, 1]
```

iii. The AI identified this series as the side-camera tongue trace but elected to use y alone.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Raw session-wide y values define the 40th/60th percentiles. For each 50-ms bin, the last frame is selected; if no frame occurs, the most recent frame before the bin end is carried forward. No confidence filtering or averaging is performed.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
values[i] = tongue_y[end_idx - 1]
```

iii. The notes state this was intended as a synchronized representative sample per neural bin, with carry-forward preventing missing bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values default to 0, values from p40 through p60 inclusive become 1, and values above p60 become 2. Only three categories are declared; `not visible` is absent.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
cats[(values >= p40) & (values <= p60)] = 1
```

iii. The AI followed the numerical percentile bands but overlooked the required fourth visibility category and the tracking-likelihood semantics.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are searched at each trial's absolute neural edges, then one y sample is assigned per corresponding 50-ms interval; empty intervals carry the preceding sample.

ii.
```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
```

iii. The AI relied on the shared session clock and identical absolute bin edges. It rejected trial-table boundaries after deciding camera data were effectively continuous.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Optional numeric strings such as `N/A` become NaN; sessions without good units and sessions with fewer than two trials are dropped; missing tone or choice receives a fallback; camera-incomplete and all-zero-neural trials are dropped; empty tongue bins are forward-filled.

ii.
```python
if value in {"N/A", "nan", "NaN", ""}: out[i] = np.nan
return float(go_time - 1.85), True
return (0 if instruction == "left" else 1), True
```

iii. The trajectory emphasizes auditable fallback counters and mechanical exclusions. The AI used fallbacks to keep required labels dense, even where the reference instead preserves an explicit missing/not-visible category.

## 10-a. What are the most time-consuming steps of the code?

i. Full-session HDF5 reads, per-unit spike binning/searches, building the large in-memory result, pickling, and later decoder loading/training dominate. The AI reported the full conversion as a long unit-binning pass and the pickle as 5.5 GB.

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left")
```

iii. The trajectory explicitly switched to HDF5 for speed and described binning tens of thousands of units as the long phase.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-wise input/output construction, the 80-bin tongue loop, unit-to-trial rate copying, decoding/parsing loops, and summary loops could be more vectorized. The important spike-edge search is already vectorized across trials, but remains looped over ragged units.

ii.
```python
for row_idx, trial_idx in enumerate(valid_trials):
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The AI prioritized a clear ragged-unit implementation and memory control; it did not claim these inner assignment and tongue loops were fully optimized.

## 10-c. What processing does the code repeat multiple times?

i. Per-trial arrays are assembled in one pass, but each trial is revisited for label/input construction, each unit's rate matrix is copied trial-by-trial, and the completed outputs are traversed again to produce summary histograms. Bin constants are correctly reused globally.

ii.
```python
for row_idx, trial_idx in enumerate(valid_trials):
    input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
for session_outputs in data["output"]:
    for trial_output in session_outputs:
        choice_hist[int(trial_output[0, 0])] += 1
```

iii. The trajectory presents the conversion as a single file pass and considers the repeated traversals acceptable bookkeeping rather than repeated scientific processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `session_output_counts` is populated but never read; `BIN_STRIDE` is unused; `trial_idx` in the construction loop is unused; several detailed stats exist only for console summaries, not the converted dataset. Neural arrays are also cast through float32 before float16, which is necessary for division but transient.

ii.
```python
session_output_counts = Counter()
session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1
BIN_STRIDE = 0.05
```

iii. The AI said the reporting counters made fallback/filter decisions auditable, but this particular counter and constant never contribute to either output or reporting.
