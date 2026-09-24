# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `data/sub-*/*.nwb` files, pre-opens every file with `h5py` to retain sessions having at least one `classification == "good"` unit, then opens each retained file again in `process_session` and reads trials, events, units, and tracking arrays directly from NWB HDF5 paths.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
        if np.any(good):
            valid.append(path)
...
with h5py.File(path, "r") as f:
    trials = f["intervals/trials"]
```

iii. The notes justify `h5py` as lower-overhead, explicit access to ragged NWB arrays and identify the one zero-good-unit session as unanalyzable.

## 1-b. How are the data split into subjects?

i. The parent directory name (for example, `sub-440956`) is the subject ID. During assembly, first occurrence order defines `subjects` and each session receives the corresponding `subject_idx`.

ii.
```python
subject_id = path.parent.name
...
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The agent says the directory subject ID is exact and follows sorted NWB paths; this yields the expected 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; its filename stem is the session ID, and sorted file order is preserved.

ii.
```python
session_id = path.stem
...
for i, path in enumerate(files, start=1):
    result = process_session(path)
    results.append(result)
```

iii. The notes identify NWB files as session-level sources and exclude only the file with no good units, producing 173 sessions.

## 1-d. How are the data split into trials?

i. Rows in `intervals/trials` define trials. The code requires the go-cue count to equal the table row count, applies one boolean mask consistently, and finally creates one neural/input/output array per retained row.

ii.
```python
n_trials_raw = len(trials["id"])
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
...
neural_trials = [firing_rates[:, i, :].copy() for i in range(n_trials)]
```

iii. The agent considers `go_start_times` the reliable one-per-trial anchor because sample and delay streams include replayed epochs.

## 1-e. How are trials filtered based on quality controls?

i. A trial is initially kept only if its complete `[go-2.5, go+1.5]` window lies within an interval from the first good unit's `obs_intervals`. After spike binning, any trial with zero spikes across every retained unit and bin is also removed. Early-lick, ignore, miss, and stimulation trials are retained; `free_water` is not explicitly filtered.

ii.
```python
return np.any((trial_start[:, None] >= starts) &
              (trial_end[:, None] <= ends), axis=1)
...
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
```

iii. The agent says full neural-window coverage avoids fabricated all-zero activity and that task-required behavioral categories must remain. It describes the post-binning removal as a defensive guard and reports 73,910 retained trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times`, `units/spike_times_index`, `units/classification`, and go-cue timestamps.

ii.
```python
classification = decode_str_array(f["units/classification"])
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes treat classified spike times as the NWB-native analogue of the reference classifier output.

## 2-b. How is the `neural` data processed?

i. For every good unit, `searchsorted` evaluates cumulative spike counts at all trial edges, adjacent counts are differenced, and counts are divided by 0.05 s to obtain Hz. Results are stored as `float16`; there is no smoothing or normalization.

ii.
```python
edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
counts = np.diff(edge_idx, axis=1)
firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The agent chose search-sorted, per-unit vectorized binning for speed and compact dtypes to reduce the pickle size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` is exactly `good` are retained; sessions with none are omitted during discovery. No individual metric thresholds or `unit_quality` filter are used.

ii.
```python
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
```

iii. The agent identifies this as the closest available equivalent to the external `goodunits` files and explains the 490-unit discrepancy as a release/classifier snapshot difference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Fixed relative edges from -2.5 to +1.5 s are added to each absolute go-cue timestamp before binning.

ii.
```python
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1)
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The agent chose the go cue because it is unique per trial and all NWB streams share the session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` s. Raw spikes are directly histogrammed into these bins; no subsequent rebinning is performed.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
```

iii. The agent explicitly follows the decoder requirement, noting that it supersedes the papers' 40-ms/3.4-ms analysis settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from a raw trial event. The code uses bin centers and a hard-coded canonical tone time of -1.85 s relative to the go cue.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16),
                     (n_trials, 1))
```

iii. The agent argues that replay makes `sample_start_times` irregular and that 0.65-s sample plus 1.2-s delay supports a canonical -1.85-s onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The constant -1.85-s offset is subtracted from each go-relative bin center, then the same 80-value vector is tiled over every trial and cast to `float16`.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16),
                     (n_trials, 1))
```

iii. The notes say this avoids disambiguating replay-related sample events and follows the nominal task structure.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the identical `REL_CENTERS` associated with the neural bin edges, so input column `k` corresponds to neural bin `k`.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The agent describes both inputs as fully time-varying arrays on the go-centered 80-bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus per-trial go times.

ii.
```python
onset_trial = parse_optional_float_array(photostim_onset_str)
duration = parse_optional_float_array(photostim_duration_str)
go_minus_start = go_times - start_times
```

iii. The notes establish that onset is stored relative to trial start and must be converted to go-relative coordinates.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `'N/A'` becomes NaN; onset is transformed to go-relative time, duration defines offset, and bins whose centers lie in `[onset, offset)` are marked 1, otherwise 0.

ii.
```python
onset_rel_go = onset_trial - go_minus_start
offset_rel_go = onset_rel_go + duration
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) &
       (REL_CENTERS < offset_rel_go[trial_idx])
stim[trial_idx, mask] = 1.0
```

iii. The agent keeps stimulation trials because stimulation is an explicit requested decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation times are put into go-relative coordinates and evaluated at the same 80 bin centers corresponding to neural bins.

ii.
```python
onset_rel_go = onset_trial - (go_times - start_times)
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & ...
```

iii. The notes say this mirrors the reference conversion of trial-relative stimulation timing into go-centered timing.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice uses `trial_instruction` and encoded `outcome`; for ignore trials it additionally searches session-wide left/right lick event streams inside the trial.

ii.
```python
choice_code = build_choice_array(..., left_lick_times=left_lick_times,
                                 right_lick_times=right_lick_times)
choice[hit_mask] = instructed[hit_mask]
choice[miss_mask] = 1 - instructed[miss_mask]
```

iii. The agent says hit and miss determine side, while ignore requires a lick-derived side when available and a fallback otherwise.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left/right are encoded 0/1. On ignore trials the first post-go lick is used, otherwise the first lick anywhere from trial start to stop, otherwise the instructed side. The scalar is repeated over 80 bins. No `no lick` class exists.

ii.
```python
if post_choice is not None:
    return post_choice
...
if any_choice is not None:
    return any_choice
return instructed_choice
...
np.full(N_BINS, choice_code[trial_idx], dtype=np.int16)
```

iii. The notes acknowledge that most ignores have no post-go lick but choose instructed side as a documented fallback, yielding a binary, roughly balanced target.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` strings.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
```

iii. The requested categories already exist in the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped `ignore=0`, `miss=1`, `hit=2`, then the trial value is repeated over all bins.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x]
                         for x in outcome_str])
```

iii. The agent follows the requested category order and represents trial-level outputs on the common time axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` strings.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
```

iii. The agent keeps early-lick trials because this flag is a requested output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1; the value is repeated across all 80 bins.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x]
                       for x in early_lick_str])
```

iii. This is the direct categorical encoding planned in the notes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses columns x, y, and likelihood plus timestamps from `Camera0_side_TongueTracking`.

ii.
```python
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()]
```

iii. The agent identifies the side-camera tongue series as the relevant unit-resolved behavioral stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with nonfinite coordinates or speed above mean plus five SD are interpolated. Frames with likelihood below 0.9 are not left missing; their y is replaced with the mean y of visible frames. One last-observed frame is sampled at each bin center.

ii.
```python
outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
y[outlier_mask] = np.interp(...)
visible_mask = np.isfinite(likelihood) & (likelihood >= 0.9)
y[~visible_mask] = mean_y
...
idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
return cleaned_y[np.clip(idx, 0, len(timestamps) - 1)]
```

iii. The agent calls this method-paper-inspired velocity cleaning and last-frame-carried-forward alignment. It imputes occlusion to avoid NaNs and to populate three classes.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed over all aligned, imputed bin-center samples retained in a session. Values below q40 are 0, between inclusive thresholds are 1, and above q60 are 2. There is no class 3 for not visible.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The agent says per-session thresholds should be based on samples entering the converted dataset; explicit comparisons preserve a middle class when thresholds tie.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every neural-bin center, the code chooses the most recent camera frame using `searchsorted(..., side="right") - 1`; indices outside camera coverage are clipped to the first/last frame. Thus it samples rather than averaging within the neural bin.

ii.
```python
abs_centers = go_times[:, None] + REL_CENTERS[None, :]
idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
idx = np.clip(idx, 0, len(timestamps) - 1)
return cleaned_y[idx]
```

iii. The agent believed the reference marker-alignment script used last-frame carry-forward and chose it over interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing optional stimulation values become NaN and therefore zero stimulation. A zero-good-unit session is excluded. Trials lacking full `obs_intervals` coverage or any spikes are excluded. Nonfinite tongue coordinates and velocity outliers are interpolated; low-likelihood tongue frames are mean-imputed, so no missing/not-visible tongue category remains. Structural mismatches raise errors.

ii.
```python
if value == "N/A":
    continue
...
y[outlier_mask] = np.interp(...)
y[occluded_mask] = mean_y
...
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
```

iii. The agent aimed for no NaNs and no all-zero neural trials, and documented imputation/filtering as defensive consistency handling.

## 10-a. What are the most time-consuming steps of the code?

i. The code times spike binning per session and total session processing. The likely dominant operations are reading large spike/video arrays, per-unit `searchsorted` binning, copying neuron-by-trial arrays, and writing the 4.5-GB pickle. The notes report roughly 0.7 s per sample session and a multi-minute full conversion.

ii.
```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
neural_time_s = time.perf_counter() - t_neural
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent emphasizes vectorized spike binning, trial prefiltering, and compact dtypes as its primary runtime/size optimizations.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike loop is hard to remove because spike trains are ragged, but its trial dimension is already vectorized. The loop over valid photostim trials could be broadcast; ignore-trial lick searches repeatedly scan full lick arrays; per-trial input/output construction could be replaced by array transposes/copies; string parsing and region mapping could also be vectorized or cached.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices): ...
for trial_idx in np.where(valid)[0]: ...
for trial_idx in ignore_trials: ...
for trial_idx in range(n_trials): ...
for unit_idx, label in enumerate(result.region_labels_per_unit): ...
```

iii. The notes claim all heavy work is vectorized and specifically defend the per-unit search approach, but do not discuss all remaining Python loops.

## 10-c. What processing does the code repeat multiple times?

i. Every valid NWB is opened during discovery and again for conversion. Photostim onset parsing and go-relative onset calculation occur once to build the input and again for diagnostics. Trial arrays are repeatedly copied/stacked during list conversion.

ii.
```python
with h5py.File(path, "r") as f:  # get_nwb_files
...
with h5py.File(path, "r") as f:  # process_session
...
onset_trial = parse_optional_float_array(photostim_onset_str)  # twice
```

iii. The agent says it reuses bin centers and session thresholds and avoids repeated percentile work, but its documentation overlooks the file pre-scan and diagnostic recomputation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Diagnostics are computed for every full-run session even when `--show-processing` is false, although most diagnostics are not put in the final dataset. Tongue x is copied, interpolated, and used for speed/outlier detection but only cleaned y is returned. `region_names` is calculated per session but never consumed by `build_dataset`. Several arrays are retained solely for optional plotting/logging.

ii.
```python
x[outlier_mask] = np.interp(...)
...
diagnostics = {"tracking_preview_raw": ..., "aligned_tongue_y_trial": ..., ...}
...
region_names=sorted(set(region_labels.tolist())),
```

iii. The agent justifies diagnostic plots and sanity checks during development, but does not condition diagnostic construction on plotting and does not identify these discarded computations in its efficiency notes.
