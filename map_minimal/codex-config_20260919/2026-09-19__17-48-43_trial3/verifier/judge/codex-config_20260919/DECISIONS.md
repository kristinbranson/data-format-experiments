# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates sorted NWB files matching `sub-*/*.nwb`, opens each twice with `h5py` (once to scan vocabularies and once to convert it), and reads trials, behavioral events, units, subject metadata, and tongue tracking directly from HDF5 paths.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
subjects, brain_regions, paths = scan_vocab(paths)
...
with h5py.File(path, "r") as nwb:
    trials = nwb["intervals/trials"]
    events = nwb["acquisition/BehavioralEvents"]
    units = nwb["units"]
```

iii. The trajectory says the agent inspected the NWB representation and intentionally used `h5py` because the archived files use an older NWB schema while the needed fields are standard HDF5 datasets. Sorting makes processing deterministic.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each file's `general/subject/subject_id`. A sorted unique vocabulary and lookup table are built during a preliminary scan; every retained session receives its subject's integer index.

ii.
```python
subjects.add(nwb["general/subject/subject_id"][()].decode("utf-8").strip())
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
subject = nwb["general/subject/subject_id"][()].decode("utf-8").strip()
return (..., subject_to_idx[subject], ...)
```

iii. The agent treated the NWB subject field as canonical and reported 28 subjects. No separate inference from filenames was necessary.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Converted sessions are appended in sorted path order; files with no classifier-approved units, or fewer than two retained trials, are omitted.

ii.
```python
for session, path in enumerate(paths, start=1):
    converted = convert_session(path, subject_to_idx, region_to_idx)
    ...
    if len(n) < 2:
        continue
    neural.append(n)
```

iii. The trajectory notes that excluding the single file with no classifier-approved units produces the paper's 173-session count. The file boundary already defines the session boundary.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`, and the code requires one go cue per behavioral trial. It initially retains the leading number of trials represented by the second dimension of `units/is_good_trials`, then removes trials with no binned activity among retained good units. Each retained trial becomes one array in each session list.

ii.
```python
n_behavior_trials = len(trials["id"])
all_go_times = events["go_start_times/timestamps"][:]
if len(all_go_times) != n_behavior_trials:
    raise ValueError(...)
n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
...
neural_trials = [rates[:, trial, :].copy() for trial in range(n_trials)]
```

iii. The agent discovered during verification that some behavioral trial tables extend past ephys acquisition. It therefore used `is_good_trials` as the recorded span so behavioral-only tails would not become artificial zero-neural trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps early-lick, ignore, free-water, and photostimulation trials in principle. It removes behavioral trials beyond the `is_good_trials` column count, then removes any trial whose 4-s window has no spikes from any **classifier-good** unit. Sessions with fewer than two surviving trials are removed.

ii.
```python
n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
...
has_neural_data = np.any(rates != 0, axis=(0, 2))
source_trial_idx = source_trial_idx[has_neural_data]
...
if len(n) < 2:
    continue
```

iii. The agent justified retaining behavioral classes explicitly requested as decoder targets and stimulation as a requested input. It excluded behavior-only tails and all-zero windows because these cannot supply neural input. The trajectory reports 90,734 retained trials. Although the comment says the all-zero check covers all units, it is actually applied after good-unit filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `units/spike_times_index`, restricted using `units/classification`; `go_start_times/timestamps` supplies trial alignment.

ii.
```python
classification = decode_strings(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
all_spikes = units["spike_times"]
ends = units["spike_times_index"][:].astype(np.int64, copy=False)
```

iii. The agent found that the NWB directly carries the paper's classifier result, so it did not recreate the QC model. It identified spike times as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. For each good unit, absolute spikes are assigned to nonoverlapping go-relative trial windows and then to 50-ms bins. `np.add.at` adds `1/0.05 = 20` per spike, producing float32 firing rates in Hz. There is no smoothing, normalization, or baseline correction.

ii.
```python
relative = candidate_spikes - go_times[candidate_trials]
bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
np.add.at(rates[output_unit],
          (candidate_trials[valid_bin], bin_idx[valid_bin]),
          np.float32(1.0 / BIN_SIZE_S))
```

iii. The trajectory says the supplied repository subtracts trial go times and divides spike counts by bin width. A dry run confirmed rates in 20-Hz increments.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose decoded `classification` equals `"good"` are retained. Sessions with zero such units are discarded; anatomical labels do not impose further QC.

ii.
```python
classification = decode_strings(nwb["units/classification"][:])
good = classification == "good"
if np.count_nonzero(good) == 0:
    continue
```

iii. The agent states this field is the paper's region-specific classifier output and the analysis population used by supplied preprocessing code. It retained 69,453 approved units and excluded one unclassified session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All timestamps remain on the common NWB session clock. Trial windows are placed at each go time; spikes are converted to `spike_time - go_time` and kept in `[-2.5, 1.5)`.

ii.
```python
window_starts = go_times + OFF_START_S
window_ends = go_times + OFF_END_S
...
relative = candidate_spikes - go_times[candidate_trials]
```

iii. The agent concluded that spikes and events share an absolute clock, so no additional clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins spanning `[-2.5, 1.5)` seconds around go onset. Raw point-process spikes are histogrammed directly to that resolution; no subsequent rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1) * BIN_SIZE_S
```

iii. These values directly implement the decoder instructions. The agent emphasized exact half-open tiling and consistent 80-bin trial shapes.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times/timestamps`, each retained trial's `start_time`, and its go cue. The selected tone is the last sample start before or at go that also occurs within the trial.

ii.
```python
sample_starts = events["sample_start_times/timestamps"][:]
tone_onsets = final_tone_onsets(trial_starts, go_times, sample_starts)
```

iii. The agent recognized that an early lick can replay the sample epoch, so it selected the final tone before go—the one relevant after any replay.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each neural bin center relative to go, the code adds the elapsed interval from selected tone onset to go, yielding seconds since tone at every bin center.

ii.
```python
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. The trajectory describes this as a simple clock-axis transformation with no other processing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 go-relative bin centers used by neural histograms, so column `k` describes the center of neural bin `k`.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
time_from_tone = BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
```

iii. The shared go-relative grid supplies alignment without resampling.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trials-table `photostim_onset`, `photostim_duration`, and `start_time`, plus the go-aligned absolute bin centers. String `"N/A"` values are parsed as NaN.

ii.
```python
photo_onset = parse_optional_floats(trials["photostim_onset"][:][source_trial_idx])
photo_duration = parse_optional_floats(trials["photostim_duration"][:][source_trial_idx])
absolute_photo_start = trial_starts + photo_onset
```

iii. The agent determined that stimulation onset is stored relative to trial start, so it must be converted to the common session clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A float32 binary series is one when a bin center lies in the half-open interval `[photo_start, photo_end)` and zero otherwise. NaN starts naturally yield all-zero non-stimulation trials.

ii.
```python
photo_on = ((absolute_centers >= absolute_photo_start[:, None])
            & (absolute_centers < absolute_photo_end[:, None])
            & np.isfinite(absolute_photo_start[:, None])).astype(np.float32)
```

iii. The agent chose a time-varying indicator because the task asks whether stimulation is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-relative stimulation fields are converted to absolute times and compared with the absolute centers of the exact neural bins.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
absolute_photo_start = trial_starts + photo_onset
```

iii. Both quantities are thereby placed on the common NWB clock; the binary columns correspond one-for-one to neural columns.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice from `left_lick_times/timestamps`, `right_lick_times/timestamps`, and each go cue, rather than from instruction and outcome fields.

ii.
```python
left_licks = events["left_lick_times/timestamps"][:]
right_licks = events["right_lick_times/timestamps"][:]
choice = event_choice(go_times, left_licks, right_licks)
```

iii. The final code treats observed response-epoch lick events as the direct behavioral source. Earlier trajectory text proposed recovering choice from instruction plus hit/miss, but the implemented decision changed to event timestamps.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick at or after go are located. The earlier event before `go + 1.5` gives left=0 or right=1 (ties favor left); if neither exists, choice=2. The scalar class is repeated through all 80 output columns.

ii.
```python
choice = np.full(len(go_times), 2, dtype=np.int8)
...
if left_time < response_end and left_time <= right_time:
    choice[trial] = 0
elif right_time < response_end:
    choice[trial] = 1
...
output_trial[0, :] = choice[trial]
```

iii. The response window follows the 1.5-s task response epoch. Tiling was chosen because output rows share a temporal array while choice is per trial.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the retained rows of trials-table `outcome`.

ii.
```python
outcome_text = decode_strings(trials["outcome"][:][source_trial_idx])
```

iii. The NWB labels already exactly match the requested categories, so no inference is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map as ignore=0, miss=1, hit=2; unknown values raise an error. The result is repeated across 80 bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
output_trial[1, :] = outcome[trial]
```

iii. The coding follows the requested output ordering, and tiling accommodates a per-trial label in a shared time-varying output matrix.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the retained rows of trials-table `early_lick`.

ii.
```python
early_text = decode_strings(trials["early_lick"][:][source_trial_idx])
```

iii. The NWB already supplies the relevant per-trial flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `"no early"` maps to 0 and `"early"` maps to 1, with validation against unexpected strings. The result is repeated through all time bins.

ii.
```python
unknown_early = sorted(set(early_text) - {"no early", "early"})
early = (early_text == "early").astype(np.int8)
output_trial[2, :] = early[trial]
```

iii. This directly supplies the requested no/yes codes and uses the same tiling convention as other trial-level outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses the side-camera `Camera0_side_TongueTracking` data and timestamps. Data column 1 is y-position and column 2 is tracking likelihood.

ii.
```python
tongue = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue["data"][:]
tongue_times = tongue["timestamps"][:]
tongue_y, visible, n_outliers = clean_tongue_y(tongue_data[:, 1], tongue_data[:, 2])
```

iii. The agent identified the side camera as the view used by the method paper and used likelihood to distinguish visible from occluded samples.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Samples are visible only if y and likelihood are finite and likelihood is at least 0.9. Among consecutive visible frames, transitions above mean speed plus five standard deviations mark destination frames as outliers; these are linearly interpolated when possible. The cleaned trace is sampled at the nearest video timestamp to every neural bin center.

ii.
```python
visible = np.isfinite(clean) & np.isfinite(likelihood) & (
    likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
cutoff = speed_sample.mean() + 5.0 * speed_sample.std()
clean[outlier] = np.interp(x[outlier], x[good], clean[good])
...
video_idx = nearest_indices(tongue_times, absolute_centers.ravel())
```

iii. The agent called 0.9 a conservative standard cutoff justified by strongly bimodal likelihoods. It attributed five-sigma velocity-outlier interpolation to the method paper and kept occlusions unimputed to preserve the requested not-visible class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed from all cleaned, visible **raw frames** in a session. Nearest-frame values are coded 0 below q40, 1 from q40 through q60 inclusive, 2 above q60, and 3 when the sampled frame is not visible.

ii.
```python
q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
tongue_class = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_class[sampled_visible & (sampled_y < q40)] = 0
tongue_class[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_class[sampled_visible & (sampled_y > q60)] = 2
```

iii. The agent followed the requested per-session percentile scope and four categories, but chose raw visible samples as the percentile population.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Absolute neural bin centers (`go + BIN_CENTERS`) are queried against camera timestamps, and the nearest single frame supplies each bin's label. There is no maximum-distance or trial-boundary check.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    n_trials, N_TIME)
```

iii. The agent relied on the streams' common NWB clock and used nearest-neighbor sampling to align camera frames to firing-rate bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. `"N/A"` optional photostimulation values become NaN; unknown categorical labels and count mismatches raise errors. Sessions without good units are skipped, behavior-only tails and all-zero good-unit windows are dropped, and sessions with fewer than two trials are skipped. Low-confidence/nonfinite tongue frames become not-visible; detected velocity outliers are interpolated. Output is first written to a temporary file and atomically replaced.

ii.
```python
valid = text != "N/A"
result[valid] = text[valid].astype(np.float64)
...
if len(all_go_times) != n_behavior_trials:
    raise ValueError(...)
...
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(temporary, output_path)
```

iii. The agent distinguished unusable neural records from meaningful missingness such as tongue occlusion. The trajectory shows the ephys-tail correction was prompted by verifier warnings and intended to avoid fabricating all-zero neural trials.

## 10-a. What are the most time-consuming steps of the code?

i. The code does not instrument step timings. Its largest work is reading every NWB twice, binning every good unit's spikes with indexing and `np.add.at`, loading/cleaning full-session video, building many per-trial arrays, and pickling the roughly 11.5-GiB result.

ii.
```python
for path in paths:
    with h5py.File(path, "r") as nwb: ...  # scan
for session, path in enumerate(paths, start=1):
    converted = convert_session(...)
...
for output_unit, source_unit in enumerate(good_indices):
    ...
    np.add.at(...)
```

iii. The trajectory reports conversion and full-artifact writing as substantial stages, but supplies no per-function profile. The implementation deliberately uses float32 to limit the largest array.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Explicit loops remain over files, units, trials in tone selection, and trials when packaging outputs. Session/file loops are natural boundaries; tone selection and output construction could be vectorized. The ragged-unit loop is harder to remove, though a histogram/search strategy could reduce `np.add.at` overhead.

ii.
```python
for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
    ...
for output_unit, source_unit in enumerate(good_indices):
    ...
for trial in range(n_trials):
    output_trial = np.empty((4, N_TIME), dtype=np.int8)
```

iii. The agent vectorized spike assignment across trials within each unit and camera lookup across all bin centers. It favored straightforward loops at ragged-data and output-list boundaries.

## 10-c. What processing does the code repeat multiple times?

i. Every usable NWB file is opened twice: `scan_vocab` decodes classifications, good-unit region annotations, and subject IDs, then `convert_session` rereads those same fields. Per-trial output allocation also repeats constant-label tiling.

ii.
```python
subjects, brain_regions, paths = scan_vocab(paths)
...
converted = convert_session(path, subject_to_idx, region_to_idx)
```

iii. The first pass establishes globally sorted subject and brain-region vocabularies before integer indices are emitted. The agent accepted duplicated I/O for stable mappings.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The preliminary scan's decoded classifications/annotations are discarded and reread. `clean_tongue_y` computes and stores an outlier count used only in metadata. Several session diagnostic counts and source details are likewise metadata rather than decoder inputs. Most importantly, rates are computed for all ephys-span trials before all-zero trials are removed.

ii.
```python
rates = bin_good_units(units, good_indices, go_times)
has_neural_data = np.any(rates != 0, axis=(0, 2))
rates = rates[:, has_neural_data, :]
...
"tongue_velocity_outliers_imputed": n_outliers,
"choice_counts": np.bincount(choice, minlength=3).astype(int).tolist(),
```

iii. These diagnostics improve provenance, while the two-pass scan makes stable categorical vocabularies possible. The agent did not explicitly identify them as waste; this follows from inspection of data flow.
