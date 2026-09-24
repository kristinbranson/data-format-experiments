# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers all NWB session files with a sorted glob, excludes one named unresolved-QC session, and opens each remaining file once with `h5py`. It directly reads the subject, trial table, events, units, spikes, and tongue tracking arrays.

ii.
```python
all_paths = sorted(glob.glob("/app/data/sub-*/*.nwb"))
for session_idx, path in enumerate(valid_paths, start=1):
    result = process_session(path)
```
```python
with h5py.File(path, "r") as h5file:
    trial_start_times = np.asarray(h5file["intervals"]["trials"]["start_time"][()])
    spike_times_all = np.asarray(h5file["units"]["spike_times"][()])
```

iii. The notes say NWB is the available raw format and that one file represents one recording session. Sorting makes traversal deterministic; the single excluded file has no classifier-good units and explains the 174-file versus 173 analyzed-session discrepancy.

## 1-b. How are the data split into subjects?

i. The subject is read from each file's `general/subject/subject_id`, with the `sub-*` directory name as a fallback. During assembly, subjects are added in first-session order and every session receives an integer `subject_idx`.

ii.
```python
subject = h5file["general"]["subject"]["subject_id"][()]
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx[session_idx] = subject_to_idx[subject]
```

iii. The notes identify the NWB subject field as the canonical animal identifier and report 28 retained subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its filename without `.nwb` is the session name, and output session order follows the sorted path list. The known session with no classifier-good units is omitted.

ii.
```python
def session_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    ...
    return base
```
```python
if session_name == EXCLUDED_SESSION:
    continue
```

iii. The agent concluded that the NWB file boundary is already the session boundary and that excluding the unresolved-QC file reproduces the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Trial rows and go-cue events are treated as index-aligned. Instead of asserting equal lengths, the agent truncates every trial-level vector to the minimum available length, then applies one shared mask and indexes all streams with the retained trial indices.

ii.
```python
ntrials = min(len(trial_start_times), ..., len(go_times_abs))
trial_start_times = trial_start_times[:ntrials]
go_times_abs = go_times_abs[:ntrials]
...
kept_trial_indices = np.flatnonzero(keep_trials)
```

iii. The notes describe NWB trial rows and behavioral events as corresponding per trial. The truncation is an implicit robustness choice for mismatched lengths; no specific mismatch was documented.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes auto-water and free-water trials, trials without a tone, trials not marked good for every retained unit in `units/is_good_trials`, windows outside the aggregate good-unit spike span, and finally trials whose entire binned neural matrix is zero. Early-lick, ignore, and photostimulation trials are deliberately retained. Sessions with fewer than two survivors are dropped.

ii.
```python
keep_trials = ((~auto_water) & (~free_water) & np.isfinite(tone_onsets_abs)
               & neural_coverage_mask & spike_support_mask)
...
nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials])
```

iii. The agent says auto/free-water trials confound choice/outcome, while early-lick, ignore, and stimulation must remain because they are requested targets/inputs. Neural filters were intended to avoid fabricating zero activity where recording was absent. These choices produced 88,943 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `units/spike_times` and `spike_times_index` arrays, restricted by `units/classification`; go-cue timestamps define the analysis windows.

ii.
```python
classification = decode_string_array(h5file["units"]["classification"])
good_unit_indices = np.flatnonzero(classification == "good")
spike_times_all = np.asarray(h5file["units"]["spike_times"][()])
```

iii. The notes identify spike times as the raw neural representation and `classification == "good"` as the NWB analogue of the paper's classifier QC.

## 2-b. How is the `neural` data processed?

i. For every good unit, `searchsorted` obtains cumulative spike indices at every absolute bin edge. Adjacent differences are spike counts, divided by 0.05 s to obtain Hz and stored as `float16`. There is no smoothing, normalization, or baseline subtraction.

ii.
```python
insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
counts = np.diff(insertion_idx, axis=1)
neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)
```

iii. The agent states that this matches the reference firing-rate histogram concept while using the task-mandated non-overlapping 50 ms bins. `float16` was chosen to reduce the very large output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` string is exactly `good` are retained; no additional firing-rate or metric threshold is used. Sessions with no such unit are skipped.

ii.
```python
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    return None
```

iii. The notes explain that `classification` reflects the paper's region-specific QC classifiers and is much closer to the reported 69,943 units than `unit_quality`; the conversion retains 69,453.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each absolute go-cue timestamp, and spikes are counted directly between those absolute edges.

ii.
```python
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
```

iii. The agent says all NWB timestamps use the same session clock, so adding go-relative offsets is sufficient and requires no interpolation or clock correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50 ms bins covering `[-2.5, 1.5)` s. Raw spike times are binned once; no later temporal rebinning is performed.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1) * BIN_SIZE_S
```

iii. This directly follows the decoder instructions, superseding the reference paper's 40 ms sliding windows where necessary.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses trial `start_time`, absolute `sample_start_times`, and absolute go-cue times. The tone is the latest sample-start event after trial start and before the go cue.

ii.
```python
start_idx = np.searchsorted(sample_start_times_abs, trial_start_times, side="left")
end_idx = np.searchsorted(sample_start_times_abs, go_times_abs, side="left")
tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
```

iii. The agent notes that early licking can replay sample epochs, so event arrays cannot simply be paired by index; the last pre-go sample onset represents the operative tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is first expressed relative to go, then subtracted from each go-relative bin center. Trials with no valid tone are removed.

ii.
```python
tone_rel_s = tone_onsets_abs[kept_trial_indices] - go_keep
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
```

iii. The notes describe this as a continuous trajectory with zero at tone onset, preserving replay-extended trial timing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 go-relative bin centers as the neural firing rates.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
```

iii. The common go cue and bin grid guarantee one input value for each neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus the trial's go cue. `N/A` values become NaN.

ii.
```python
stim_on_rel_s[valid_stim] = (trial_start_keep[valid_stim]
    + photostim_onset_keep[valid_stim] - go_keep[valid_stim])
stim_off_rel_s = stim_on_rel_s + photostim_duration_keep
```

iii. The agent verified that onset is trial-start-relative and therefore must be shifted to the go-relative axis used by the decoder.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary value is set to one whenever a 50 ms bin has any overlap with the half-open stimulation interval; non-stimulated trials remain all zero.

ii.
```python
photostim[valid] = ((bin_starts < off) & (bin_ends > on)).astype(np.float32)
```

iii. The notes say an overlap signal preserves stimulation timing as a time-varying decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation bounds are converted to time from go and compared with the same go-relative edges used for neural bins.

ii.
```python
bin_starts = BIN_EDGES_REL[:-1][None, :]
bin_ends = BIN_EDGES_REL[1:][None, :]
```

iii. This puts stimulation and spikes on the shared go-cue clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed directly from absolute `left_lick_times` and `right_lick_times`, not from instruction and outcome. The earliest lick in the 1.5 s response window determines direction; absence of either yields `no lick`.

ii.
```python
left_idx = np.searchsorted(left_lick_times_abs, go_time, side="left")
right_idx = np.searchsorted(right_lick_times_abs, go_time, side="left")
...
labels[trial_idx] = "left" if left_time <= right_time else "right"
```

iii. The agent argues that actual lick events are the direct behavioral source and reports over 99.6% agreement with the instruction/outcome reconstruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Licks are searched in `[go, go+1.5)`, categorized as left/right/no lick, mapped to 0/1/2, and the per-trial code is repeated across all 80 bins.

ii.
```python
CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
trial_choice_int = np.asarray([CHOICE_TO_INT[x] for x in choice_labels])
outputs_2d[:, 0, :] = trial_choice_int[:, None]
```

iii. This follows the paper's 1.5 s response epoch and uses repetition only to package trial-level and time-varying targets in one output array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the NWB trial-table `outcome` strings.

ii.
```python
outcome = decode_string_array(h5file["intervals"]["trials"]["outcome"])
```

iii. The stored categories already exactly match the requested ignore/miss/hit output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers 0/1/2 and repeated across time.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
outputs_2d[:, 1, :] = trial_outcome_int[:, None]
```

iii. The fixed mapping follows the requested category order; repetition represents a per-trial target in the common `(outputs, time)` shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` strings.

ii.
```python
early_lick = decode_string_array(h5file["intervals"]["trials"]["early_lick"])
```

iii. The NWB table already supplies the relevant behavioral flag, and these trials are retained because this is a requested target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` and `early` are mapped to 0 and 1 and repeated across all bins.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
outputs_2d[:, 2, :] = trial_early_int[:, None]
```

iii. This is a direct categorical encoding of the per-trial flag.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 and 2 (y coordinate and likelihood) of `Camera0_side_TongueTracking`.

ii.
```python
tongue_data = np.asarray(h5file["acquisition"]["BehavioralTimeSeries"]
    ["Camera0_side_TongueTracking"]["data"][()])
tongue_y_raw = tongue_data[:, 1]
tongue_likelihood_raw = tongue_data[:, 2]
```

iii. The side camera matches the method paper, and likelihood supplies the visibility needed for the fourth category.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood below 0.9 are considered invisible. For each bin, the final timestamped frame within that bin is selected; its y value is categorized if visible, otherwise the bin remains class 3. No averaging or interpolation is applied.

ii.
```python
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
...
visible = np.isfinite(sampled_y) & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
classes = np.full(bin_starts_abs.shape, 3, dtype=np.int8)
```

iii. The notes cite reference marker alignment as using last-frame sampling and describe likelihood as strongly bimodal, making 0.9 a defensible visibility cutoff.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over all raw frames with likelihood at least 0.9. Visible sampled values are assigned 0 below p40, 1 from p40 through p60 inclusive, and 2 above p60; invisible/missing bins are 3.

ii.
```python
tongue_q40, tongue_q60 = np.quantile(tongue_y_raw[visible_global], [0.4, 0.6])
classes[visible & (sampled_y < visible_q40)] = 0
classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
classes[visible & (sampled_y > visible_q60)] = 2
```

iii. The agent says session-wide visible-frame percentiles implement the requested per-session discretization without allowing occluded tracking values to distort thresholds.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Absolute bin starts/ends are constructed from each go cue and the neural relative edges. The last camera frame inside each interval supplies the corresponding output bin.

ii.
```python
bin_starts_abs = go_times_abs[:, None] + BIN_EDGES_REL[:-1][None, :]
bin_ends_abs = go_times_abs[:, None] + BIN_EDGES_REL[1:][None, :]
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
```

iii. Camera, event, and spike timestamps share the NWB session clock, so identical absolute intervals align the streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte strings and optional numeric strings are decoded explicitly; `N/A` becomes NaN. Subject IDs fall back to directory names. Trial vectors are truncated to their common minimum. Missing tones, missing common neural coverage, unsupported spike windows, auto/free-water trials, and all-zero neural windows are dropped. A no-good-unit or under-two-trial session is dropped. Missing/low-confidence tongue samples become `not visible`, and blank anatomy becomes `Unknown`.

ii.
```python
if item != "N/A": out[idx] = float(item)
...
ntrials = min(...)
...
region_labels = np.asarray([label if str(label).strip() else "Unknown" ...])
```

iii. The agent's stated principle is to exclude records where neural data would otherwise be fabricated, while representing legitimate absence of tongue visibility as an explicit requested category.

## 10-a. What are the most time-consuming steps of the code?

i. The code's main costs are reading large NWB spike/video arrays, looping over good units to bin spikes with `searchsorted`, accumulating a large in-memory result, and serializing it. Optional diagnostic plotting adds work for two sessions.

ii.
```python
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
```

iii. Progress timing is recorded per session. The notes emphasize large spike buffers, tongue arrays, and neural binning as the scale-dependent operations; the full conversion completed within the allotted time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Tone selection and lick-choice reconstruction loop over trials, good-unit spike bounds are partly constructed in a loop, and neural binning loops over units. Trial dimensions inside spike binning and tongue alignment are already vectorized. The unit loop is difficult to eliminate because spike arrays are ragged; the small trial loops could be vectorized further.

ii.
```python
for trial_idx in range(len(go_times_abs)):
    ...
for trial_idx, go_time in enumerate(go_times_abs):
    ...
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
```

iii. The implementation favors clear per-trial logic where event counts vary, while applying array operations across all bins and trials in the expensive neural search.

## 10-c. What processing does the code repeat multiple times?

i. Each session repeats schema decoding, string conversion, masking, and construction of identical relative-bin expressions. Some retained arrays are copied when neural trials are formed and again converted while assembling output lists. When visualization is requested, quantities already computed for conversion are traversed again for plots. There is no second full data-processing pass.

ii.
```python
return [neural_stack[:, trial_idx, :].copy() for trial_idx in range(n_trials)]
...
input_data.append([trial.astype(np.float32, copy=False) for trial in session["inputs_2d"]])
```

iii. The agent characterized conversion as a single pass and precomputed the shared relative grid at module scope. Most repetition is therefore packaging or optional diagnostics rather than repeated scientific transformation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `trial_stop_times` and retained `trial_instruction` are loaded/sliced but never used for final conversion. Spike-span diagnostics, multiple exclusion counts, label summaries, raw tongue arrays, and plotting-oriented fields are computed and stored in intermediate session dictionaries but omitted from the final pickle. Optional plots are also not consumed by decoder training.

ii.
```python
trial_stop_times = np.asarray(h5file["intervals"]["trials"]["stop_time"][()])
trial_instruction_keep = trial_instruction[kept_trial_indices]
...
"classification_counts": classification_counts,
"tongue_y_visible_raw": tongue_y_raw[visible_global],
```

iii. These values support diagnostics, sanity checks, progress summaries, and `--show-processing`, but they are not part of the requested converted dataset or downstream decoder inputs.
