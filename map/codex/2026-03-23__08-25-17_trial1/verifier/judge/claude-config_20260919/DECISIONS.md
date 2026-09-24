# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB (HDF5) file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single sorted glob (`sub-*/*.nwb`, 174 files) and opens each file **directly with `h5py`** rather than with `pynwb`, deliberately bypassing the NWB object layer for speed. Inside one file it reads, in one pass: the trials table (`intervals/trials`: `start_time`, `stop_time`, `trial_instruction`, `early_lick`, `outcome`, `photostim_onset`, `photostim_duration`), the behavioural event streams (`acquisition/BehavioralEvents`: `go_start_times`, `sample_start_times`, `delay_start_times`, `left_lick_times`, `right_lick_times`), the side-camera tongue tracking series, and the unit table (`units`: `classification`, `anno_name`, `obs_intervals`, `is_good_trials`, ragged `spike_times` + `spike_times_index`). Each file is opened exactly once and closed before the numeric processing begins. `--sample` takes the first 12 files and stops after 2 successful sessions; `--full` processes all of them.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```
```python
    with h5py.File(file_path, "r") as h5:
        subject_id, session_id = get_session_identity(file_path)

        units = h5["units"]
        n_recorded_trials = int(units["is_good_trials"].shape[1])
        ...
        trials = h5["intervals"]["trials"]
        n_behavior_trials = int(len(trials["id"]))
        trial_start_all = trials["start_time"][:].astype(np.float64)
        ...
        go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
        sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
        ...
        spike_times = units["spike_times"][:].astype(np.float64)
        spike_times_index = units["spike_times_index"][:]
        spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```

iii. From CONVERSION_NOTES.md Step 6: "Used `h5py` directly instead of PyNWB for the conversion path" because "Full PyNWB object loading produces avoidable overhead and warning spam." Step 2 documents that the layout is DANDI-style with one NWB file per session per subject folder and that the trial/unit schema is identical across all 174 files (`n_unique_trial_schema = 1`, `n_unique_unit_schema = 1`), so a single uniform reader is sufficient.

## 1-b. How are the data split into subjects?

i. The subject id is taken from the **containing folder name** (`sub-440956` → `'440956'`), not from `general/subject/subject_id` inside the file. At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list. This yields 28 subjects with 3–10 sessions each, matching `dandiset.yaml`.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```
```python
    subjects = sorted({r.subject_id for r in results})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[r.subject_id] for r in results], dtype=np.int64),
```

iii. Step 5 of CONVERSION_NOTES.md lists the mapping as "`subject.subject_id` or folder name `sub-<id>` → `subjects`, `subject_idx`", i.e. the AI treated the two as interchangeable (they are: the folder name is derived from the NWB subject id). Step 9 records the sanity check that the converted data has 28 subjects, matching the data paper's "This study is based on data from 28 mice."

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. The session id is the file stem (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order follows the sorted file list (hence chronological within subject). A session is dropped only if it has zero units with `classification == 'good'`, or zero recorded trial columns. Exactly one session is dropped (`sub-440958_ses-20190216T162508`, whose `classification` field is NaN for all units), leaving 173 sessions. Session ids and source paths are stored in `metadata['session_ids']` / `metadata['source_files']`.

ii.
```python
        classification = np.char.lower(decode_str_array(units["classification"][:]))
        good_unit_idx = np.flatnonzero(classification == "good")
        if len(good_unit_idx) == 0:
            print(f"Skipping {session_id}: zero good units")
            return None
```
```python
        "session_ids": [r.session_id for r in results],
        "source_files": [r.source_file for r in results],
        "n_sessions": len(results),
```

iii. Step 4 of CONVERSION_NOTES.md: "One raw NWB session (`sub-440958_ses-20190216T162508...`) has zero units with `classification == good`; after excluding sessions with zero good units, raw data and papers agree at 173 analyzable sessions." Step 5 Key Decision 1 states session inclusion = "at least one `classification == good` unit and at least two trials." The AI also explicitly rejected a behaviour-based session filter (>65 % performance, ≥50 correct left/right trials) because applying it naively gave 152 sessions rather than the published 173.

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table, one row per behavioural trial, and the per-trial anchor is the go cue from `BehavioralEvents/go_start_times`, truncated to the number of trial rows (`[:n_behavior_trials]`). There is no assertion that the number of go events equals the number of trials (I verified externally that they are equal in all 174 files, so the truncation is harmless). Trials are then subset by the quality rules in 1-e; all per-trial arrays (start/stop, instruction, outcome, early lick, photostim) are subset with the same index array so they stay row-aligned.

ii.
```python
        trials = h5["intervals"]["trials"]
        n_behavior_trials = int(len(trials["id"]))
        ...
        go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
```
```python
        trial_start = trial_start_all[selected_trial_idx]
        trial_stop = trial_stop_all[selected_trial_idx]
        trial_instruction = trial_instruction_all[selected_trial_idx]
        early_lick = early_lick_all[selected_trial_idx]
        outcome = outcome_all[selected_trial_idx]
        photostim_onset = photostim_onset_all[selected_trial_idx]
        photostim_duration = photostim_duration_all[selected_trial_idx]
        go_times = go_times_all[selected_trial_idx]
```

iii. Step 5 maps `go_start_times.timestamps` to the "temporal alignment anchor" with the note "One go cue per trial." Step 10 records that the AI discovered, and fixed, an earlier bug in which it had assumed the ephys-backed trials were the *first* N rows of the behavioural trial table; it verified on three sessions (offsets 0, 1 and 126) that the converted trials line up with raw recomputation.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both motivated by spike-data availability rather than behaviour:

1. **Observation-window filter.** A trial is kept only if its *entire* `[-2.5 s, +1.5 s]` go-aligned window lies inside a single session-level interval `[min(obs_start), max(obs_stop)]` computed from `units/obs_intervals` "for the good units".
2. **All-zero-neural filter.** After binning, any trial whose whole `(n_units, 80)` matrix is exactly zero is dropped.

Sessions with <1 good unit are dropped (1-c). Early-lick, ignore, photostim and `free_water` trials are all deliberately *retained* as trial types, since early lick and outcome are decoder targets.

Net result: **51,346 trials kept** across 173 sessions (mean 297/session), versus 93,310 recorded ephys trials in the raw files — i.e. ~45 % of the recorded trials are discarded. (The human reference keeps 90,860.)

Critically, the code reads `units["obs_intervals"]` **without** `obs_intervals_index`. `obs_intervals` is a ragged flat array of shape `(n_units × n_trials, 2)`; indexing it with the good-unit indices returns arbitrary rows of that flat array, not each unit's interval list. Consequently `unit_obs_intervals[unit_pos]` is one arbitrary trial's `[start, stop]`, and in the 120/173 sessions that take the fallback branch (see 2-c) nearly every trial is judged "unobserved" for nearly every unit, its rates are overwritten with 0, and the trial is then either dropped by the all-zero filter or kept with ~99 % of its units forced to silence.

ii.
```python
def select_trial_indices(
    go_times_all: np.ndarray,
    good_unit_obs_intervals: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
```
```python
        unit_obs_intervals = units["obs_intervals"][good_unit_idx].astype(np.float64)   # NOTE: no obs_intervals_index
        selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
            go_times_all=go_times_all,
            good_unit_obs_intervals=unit_obs_intervals,
        )
```
```python
    nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
    dropped_zero_trials = int(np.sum(~nonzero_trial_mask))
    if dropped_zero_trials:
        neural_session = neural_session[nonzero_trial_mask]
        ...
```

iii. Step 5 Key Decision 8: "Retain early-lick, miss, hit, ignore, and photostimulation trials because these are either decoder outputs or decoder inputs. Do not apply the reference code's `regular_trial_mask` wholesale. However, do require that the full neural decoding window is actually supported by the raw session observation interval." Step 9 explains the trial-count shortfall as intentional: "The trial totals are intentionally lower because the converted dataset keeps only trials whose full `[-2.5, +1.5] s` go-aligned neural window is actually supported by the raw observation interval, which is stricter than the paper's behavior-table summary statistics." The all-zero-trial drop is recorded only as a per-session statistic (`dropped_all_zero_neural_trials`) and is not separately justified in the notes.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` (a single flat array of session-absolute spike times) together with `units/spike_times_index` to recover each unit's slice, restricted to units with `units/classification == 'good'`. `BehavioralEvents/go_start_times` supplies the per-trial anchor that places the bin edges. `units/obs_intervals` and `units/is_good_trials` are additionally used as validity masks (see 2-c).

ii.
```python
        spike_times = units["spike_times"][:].astype(np.float64)
        spike_times_index = units["spike_times_index"][:]
        spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```
```python
    for unit_pos, unit_idx in enumerate(good_unit_idx):
        spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
```

iii. Step 5 variable mapping: "`units/spike_times` for units with `classification == good` → `neural`; bin spikes into firing rates (Hz) in non-overlapping 50 ms bins over [-2.5, 1.5) s relative to go cue", with the reference functions cited as `sliding_histogram` / `process_one_area`. Spike times are the only neural representation in the files.

## 2-b. How is the `neural` data processed?

i. For each good unit, the 81 go-relative bin edges of every trial are flattened into one absolute-time array, `np.searchsorted` gives the running spike count at each edge, differencing gives the count per bin, and dividing by the 50 ms bin width converts to **Hz**. No smoothing, normalisation or baseline subtraction. Rates are cast to `float16` for storage. Immediately afterwards, rates for unit×trial pairs judged invalid are **overwritten with 0.0** (see 2-c), which for 120 of 173 sessions destroys essentially all of the real firing rates (measured mean rate ≈ 0.08–0.11 Hz in those sessions versus ≈ 5–9 Hz in the unaffected ones).

ii.
```python
    neural_session = np.zeros((n_trials, n_units, n_bins), dtype=np.float16)
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    flat_edges = abs_edges.reshape(-1)

    for unit_pos, unit_idx in enumerate(good_unit_idx):
        spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
        edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
        counts = np.diff(edge_indices, axis=1).astype(np.float32)
        fr = counts / bin_width
        ...
        if np.any(invalid_trials):
            fr[invalid_trials] = 0.0
        neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. Step 6: "Used vectorized `np.searchsorted` across all trial/bin edges per unit for spike binning" and "Stored neural firing rates as `float16` to reduce pickle size while remaining valid floating-point input for the decoder (training code converts to `float32`)." Step 10 Check 2 verifies one bin directly against the raw file: session `sub-440956_ses-20190207T120657`, trial 6, neuron 4, bin 11 → 20.0 Hz, `np.allclose == True`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels.

**Unit level:** keep only units with `units/classification == 'good'` (the spike-sorting QC classifier of Chen, Liu et al. 2023). No thresholds on individual metrics (`presence_ratio`, `amplitude_cutoff`, `isi_violation`, …) and the older `unit_quality` label is not used. This yields **69,453 good units** over 173 sessions (mean 401.5/session), which matches the NWB release exactly and is within 0.7 % of the white paper's 69,943.

**Unit×trial level:** the AI additionally masks out unit-trials it considers unobserved, by *zeroing the firing rates* rather than excluding them. The branch taken depends on whether `units/is_good_trials.shape[1]` happens to equal the number of selected trials:
- if equal (53/173 sessions): `invalid = ~is_good_trials[unit]`;
- otherwise (120/173 sessions): `invalid` is computed from the **mis-indexed** `unit_obs_intervals[unit_pos]` (see 1-e), which is not that unit's observation interval at all. I measured that this forces ~99.4 % of all unit-trials to all-zero in those sessions.

ii.
```python
        classification = np.char.lower(decode_str_array(units["classification"][:]))
        good_unit_idx = np.flatnonzero(classification == "good")
        if len(good_unit_idx) == 0:
            print(f"Skipping {session_id}: zero good units")
            return None
```
```python
        is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
        uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
        if uses_direct_is_good_trials:
            is_good_trials = is_good_trials_raw.copy()
        else:
            is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)
```
```python
        if uses_direct_is_good_trials:
            invalid_trials = ~is_good_trials[unit_pos]
        else:
            obs_start = unit_obs_intervals[unit_pos, 0]
            obs_stop = unit_obs_intervals[unit_pos, 1]
            valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
            invalid_trials = ~valid_obs
        if np.any(invalid_trials):
            fr[invalid_trials] = 0.0
```

iii. Step 5 Key Decision 2: "Use units with `classification == good` as the primary QC filter, because the papers and code describe classifier-selected good units as the analysis set." Step 4 notes the 69,453 vs 69,943 gap and attributes it to "release/version differences." Step 5 Key Decision 9: "Do not assume `units/is_good_trials.shape[1]` means 'use the first N behavioral trials'. In several sessions the ephys-backed trials form an offset block relative to the full behavioral trial table; use raw go times plus `obs_intervals` to select valid trials." The zeroing rule is recorded in metadata as `trial_selection_rule`. The trajectory shows the AI inspected `obs_intervals` shape at step 284 as `f['units']['obs_intervals'][good]` and concluded it was per-unit `(n_good, 2)`; it had printed `obs_intervals_index` earlier (step 160) but never used it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**. All NWB times share one session-absolute clock, so no resampling or offset correction is needed: the fixed grid of 81 go-relative edges is added to each trial's go-cue timestamp to give absolute edge times, and spikes are binned against those edges directly.

ii.
```python
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    flat_edges = abs_edges.reshape(-1)
    ...
        edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
```
```python
        "temporal_alignment_event": "Go cue onset",
        "off_start": WINDOW_START_S,
        "off_end": WINDOW_END_S,
```

iii. Step 5: go cue is the "temporal alignment anchor … Per-trial go cue absolute timestamp; subtract from neural/behavioral timestamps", and Step 3 notes the reference preprocessing also treats spike times as go-cue referenced ("in Susu's data, spike_times are relative to go cue time"). The `--show-processing` plots overlay the neural raster, inputs, outputs and raw event times on a common "time from go cue" axis to demonstrate there is no misalignment.

## 2-e. How is the `neural` data temporally binned/resampled? (time bin size / rebinning)

i. 80 non-overlapping **50 ms** bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and session, giving `(n_units, 80)` per trial. The grid is built once per session by `bin_edges_and_centers()` from module constants. No sliding windows, no smoothing, no secondary rebinning; `metadata['time_bin_size'] = 50.0` ms.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

iii. Step 5 Key Decision 4: "Use 50 ms non-overlapping bins and convert to firing rates in Hz. This intentionally departs from the paper's 40 ms / 3.4 ms preprocessing and 200 ms / 10 ms choice-decoder windows only because the user's decoder task explicitly requires 50 ms bins." Step 4 records the same trade-off ("Preserve the core reference principle of go-cue alignment and classifier-based unit curation, but deliberately re-bin to 50 ms").

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch tone onsets) plus the trial's go cue. For each trial the AI takes the **first** sample-start event that falls within `[trial_start, trial_stop]`. If a trial contains no sample event, it falls back to a hard-coded nominal offset of −1.85 s relative to the go cue and counts the fallback in `stats['sample_onset_fallbacks']`.

ii.
```python
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85
```
```python
    sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
    sample_onset_abs = np.empty(n_trials, dtype=np.float64)
    sample_onset_fallbacks = 0
    for trial in range(n_trials):
        events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
        if len(events):
            sample_onset_abs[trial] = events[0]
        else:
            sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
            sample_onset_fallbacks += 1
    sample_onset_rel_go = sample_onset_abs - go_times
```

iii. Step 5 Key Decision 5: "Use the earliest sample-start event in each trial as tone onset. This preserves replay-induced timing shifts visible in early-lick trials and is the most faithful raw-data representation of 'time from tone onset'." Step 5's mapping table adds: "Normal trials give ~−1.85 s tone onset; early-lick trials can have earlier tone onset because replay extends trial structure."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for every 50 ms bin, the value is the bin centre (relative to the go cue) minus the tone onset (relative to the go cue) — i.e. seconds elapsed since that trial's tone onset, negative before the tone. Stored as `float32` in row 0 of the `(2, 80)` input array. Observed range over the full dataset: `[-1.5, 11.9]` s.

ii.
```python
    for trial in range(n_trials):
        inp = np.zeros((2, n_bins), dtype=np.float32)
        inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Step 5 mapping: "For each bin, set value to `(bin_center_rel_go − sample_onset_rel_go)` in seconds." Step 10 Check 2 reports the whole `time_from_tone_onset_s` vector for a spot-checked trial matched a direct raw recomputation exactly (`np.allclose == True`, max abs diff 0.0).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input uses the same go-cue-referenced bin centres (`bin_centers_rel`) that define the neural bin edges, so bin *k* of the input covers exactly the interval of bin *k* of the firing rates. The only per-trial quantity added is the scalar tone-to-go gap.

ii.
```python
    bin_edges_rel, bin_centers_rel = bin_edges_and_centers()
    ...
    inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
    ...
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. Implicit in the design (one shared grid, defined once); the `--show-processing` panel "Constructed Inputs" plots the input against time-from-go with a vertical marker at the tone onset, and Step 7 records that "sample onset / delay onset / go cue alignment matched the raw event times in inspected trials."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset` and `photostim_duration` (stored as strings, `'N/A'` on unstimulated trials), combined with `trials.start_time` (the onset is measured from trial start) and the trial's go cue to re-express the stimulation window on the go-cue axis. The `photostim_start_times` / `photostim_stop_times` event streams were used only to cross-check the trial-table values.

ii.
```python
        photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
        photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
```
```python
        if str(photostim_onset[trial]) != "N/A":
            stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
            stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. Step 5 mapping: "`photostim_onset`, `photostim_duration`, `photostim_power` trial-table columns → `input[1]` … Trial-table onset/duration matched the event-series timestamps in spot checks." Step 3 records the paper's expectations: photoinhibition on ~25 % of randomly interleaved trials, during the last 0.5 s of the delay, always ending before the go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary, time-varying series rather than a per-trial flag: a bin is 1 if its centre lies in `[onset, onset + duration)` on the go-cue axis, else 0. Trials with `'N/A'` onset keep the all-zero row created by `np.zeros`. Stored as `float32` in row 1; observed range `[0, 1]`.

ii.
```python
        inp = np.zeros((2, n_bins), dtype=np.float32)
        ...
        if str(photostim_onset[trial]) != "N/A":
            stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
            stim_rel_off = stim_rel_on + float(photostim_duration[trial])
            inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
            photostim_trial_count += 1
```

iii. Step 5 Key Decision 6: "Represent photostimulation as a time-varying binary series, not a static trial label, because the decoder task requests the on/off state at every time point." Step 10 Check 2: the full `photostim_on` vector for a spot-checked trial matched raw recomputation exactly.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are converted from trial-start-relative to go-cue-relative (`trial_start + onset − go`) and then compared against the same `bin_centers_rel` grid used for the firing rates, so no separate alignment step is needed.

ii.
```python
            stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
            stim_rel_off = stim_rel_on + float(photostim_duration[trial])
            inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. Same shared-grid argument as 3-c; the processing plot shades the stimulation window (`axvspan`) over the input traces and the raw event-timing panel so that a reader can confirm the shaded window coincides with the raw `photostim_start_times`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored in the file. The AI derives it from the **raw lick event streams** `BehavioralEvents/left_lick_times` and `right_lick_times`, using a three-level hierarchy per trial: (1) the direction of the first lick after the go cue and before trial end; (2) if there is none, the direction of the first lick anywhere in the trial; (3) if there is no lick at all, the trial's `trial_instruction`. Counts of which rule fired are recorded per session in `stats['choice_source_counts']` (e.g. 296 / 12 / 60 of 368 trials in the first session).

ii.
```python
def infer_choice_for_trial(trial_start, trial_stop, go_time, instruction, left_lick_times, right_lick_times):
    ...
    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")

    left_any = left_lick_times[left_start:left_stop]
    right_any = right_lick_times[right_start:right_stop]
    if len(left_any) or len(right_any):
        ...
        return (0, "any_lick") if first_left < first_right else (1, "any_lick")

    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. Step 5 mapping: "First post-go lick direction inferred from left/right lick events → `output[0]` (`choice`) … Paper's choice variable is actual lick direction on responded trials." Key Decision 7: "For hit/miss trials, decode actual lick direction from lick-event timing. For ignore trials with no lick, use the fallback hierarchy documented above. This is the only unavoidable ambiguity created by the target spec relative to the raw data." (I verified independently that on responded trials the lick-derived choice agrees 100 % with `trial_instruction × outcome`.)

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as a **two-class** variable, `0 = left`, `1 = right`, written into row 0 of the `(4, 80)` `int8` output array and repeated across all 80 bins. `output_values[0] = ['left', 'right']`. There is **no "no lick" class**: the 15.5 % of trials whose outcome is `ignore` are assigned a direction from an earlier (pre-go) lick or, failing that, from the instructed side. Resulting distribution: left 0.495 / right 0.505.

ii.
```python
    choice_trials = np.empty(n_trials, dtype=np.int8)
    choice_source_counts = {"post_go_lick": 0, "any_lick": 0, "instruction_fallback": 0}
    for trial in range(n_trials):
        choice_val, source = infer_choice_for_trial(...)
        choice_trials[trial] = choice_val
        choice_source_counts[source] += 1
```
```python
    for trial in range(n_trials):
        out = np.empty((4, n_bins), dtype=np.int8)
        out[0] = choice_trials[trial]
```
```python
        "output_values": [
            ["left", "right"],
            ...
```

iii. The AI treats the no-lick case as an "unavoidable ambiguity" requiring a fallback ("This fallback is required because ignore trials have no post-go lick but the decoder spec still requires a binary choice output", Step 5 mapping table). Nothing in CONVERSION_NOTES.md or the trajectory shows the AI considering a third "no lick" category; searching the trajectory for "no lick" / "not visible" returns only this fallback discussion.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains the strings `'hit'`, `'miss'`, `'ignore'`. No derivation.

ii.
```python
        outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
        ...
        outcome = outcome_all[selected_trial_idx]
```

iii. Step 5 mapping: "`trials/outcome` → `output[1]` … Use exact trial-table labels", noting consistency with the paper's hit/error/no-response categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are lower-cased and mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit`, then written to row 1 of the output array and repeated across all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`. Converted distribution: ignore 0.155 / miss 0.093 / hit 0.752.

ii.
```python
    outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
    outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
    ...
        out[1] = outcome_trials[trial]
```

iii. Step 5 mapping: "Map `ignore -> 0`, `miss -> 1`, `hit -> 2`", i.e. exactly the ordering given in the Decoder Task spec. Step 9 notes the outcome distribution differs from the paper's 84 % correct rate "because decoder keeps early/stim/ignore structure" (the paper's figure is for control, no-early trials only).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column (`'early'` / `'no early'`). No derivation from lick times.

ii.
```python
        early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
        ...
        early_lick = early_lick_all[selected_trial_idx]
```

iii. Step 5 mapping: "`trials/early_lick` → `output[2]` … Consistent with reference code's `early_lick_trials` mask. Keep all trials; do not exclude early trials because early lick itself is a decoder target."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped through a fixed dictionary to `0 = no`, `1 = yes`, written to row 2 of the output array and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Converted distribution: no 0.888 / yes 0.112.

ii.
```python
    early_map = {"no early": 0, "early": 1}
    early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
    ...
        out[2] = early_trials[trial]
```

iii. Step 5 mapping (as above). Step 9 lists the 0.888/0.112 split as an "expected difference" from the papers, which exclude early-lick trials from most analyses, "decoder target requires retention."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` (~294 Hz). Only column 1 (`tongue_y`) is used for the output. Column 2 (`tongue_likelihood`) is read into `tongue_likelihood` and carried into the diagnostic plot payload, but is **deliberately not used** to gate or mask the values.

ii.
```python
        tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        tongue_values = tongue_group["data"][:].astype(np.float64)
        tongue_y = tongue_values[:, 1]
        tongue_likelihood = tongue_values[:, 2]
        tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. Step 2 documents the three-column layout. Step 5 mapping states the rule explicitly: "Use `tongue_likelihood` only for QC diagnostics, not thresholding, to stay close to reference code" — the reference `align_markers_between_lims` is cited as aligning markers without a likelihood mask.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial and each 50 ms bin, the AI takes the **last camera frame whose timestamp falls in that bin** (`searchsorted` on the bin's upper edge, minus one) and uses that frame's raw `tongue_y` value; bins containing no frame are marked NaN. All valid binned values of the session (across its kept trials) are pooled and their 40th/60th percentiles become the class edges. No averaging, no likelihood masking, no interpolation. I measured that only ~7–13 % of camera frames have `likelihood ≥ 0.5`, so the large majority of the binned values are the tracker's output while the tongue is retracted.

ii.
```python
def bin_tongue_y(tongue_timestamps, tongue_y, go_times, bin_edges_rel):
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1

    clipped_end = np.clip(end_idx, 0, len(tongue_y) - 1)
    valid = end_idx >= start_idx
    binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid
```
```python
    valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
    tongue_p40 = float(np.percentile(valid_values, 40))
    tongue_p60 = float(np.percentile(valid_values, 60))
```

iii. Step 5 mapping: "within each 50 ms bin take the last available `tongue_y` sample; discretize within session using 40th and 60th percentiles over all aligned binned values … Reference marker alignment uses the last frame within each time step rather than averaging." Key Decision 10: "Discretize after session-level alignment and binning, using percentiles over the kept binned tongue-y values from that session, exactly as requested by the decoder task."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. **Three** classes only: `0` = below the session's 40th percentile, `1` = between the 40th and 60th percentiles (inclusive on both edges), `2` = above the 60th percentile. `output_values[3] = ['lt_p40', 'p40_to_p60', 'gt_p60']`. The spec's fourth class (`3: not visible`) is **not implemented**, and since visibility is never tested, no bin can be labelled "not visible". Bins with no camera frame at all (NaN, ~0.2–0.3 % of bins) silently keep the array's initial value and are therefore coded as class `0`. Converted distribution: 0.410 / 0.195 / 0.395.

ii.
```python
    tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
    tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
    tongue_discrete[tongue_y_binned > tongue_p60] = 2
```
```python
        "tongue_discretization_rule": "per-session 40th and 60th percentiles over aligned binned tongue_y",
```

iii. Step 5 Key Decision 10 (per-session percentiles "exactly as requested by the decoder task"). No justification is offered anywhere for omitting the "not visible" category, and the trajectory contains no discussion of it. The AI's Step 12 review notes the low tongue decoding accuracy (0.447 vs 3-class chance 0.333) but concludes "Class balance is acceptable for all outputs" and leaves it.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues, so the identical go-relative edge grid is applied: `abs_edges = go_times[:, None] + bin_edges_rel[None, :]`, and `searchsorted` on the camera timestamps assigns frames to bins. Bin *k* of the tongue output therefore spans exactly the same interval as bin *k* of the firing rates. It is the only genuinely time-varying output (the other three are per-trial values broadcast across bins).

ii.
```python
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

iii. Step 6 lists "Used vectorized timestamp-to-bin alignment for tongue tracking" as one of the speed-ups; the `--show-processing` plot overlays the raw tongue trace and the binned step function on a shared time-from-go axis with the two percentile thresholds drawn, explicitly to demonstrate correct alignment and discretisation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, handled differently:

- **Session never quality-controlled** (`classification` is NaN for all units): `decode_str_array` turns the non-string entries into `'nan'`, no unit matches `'good'`, and the session is skipped with a printed message (1 session).
- **Zero recorded trial columns**: session skipped.
- **Trials with no spike coverage**: excluded by the observation-window filter, and any remaining trial whose neural matrix is entirely zero is dropped (this is what removes the `free_water` trials — but also ~41 k further trials, because of the `obs_intervals` indexing defect described in 1-e/2-c).
- **Unit×trial pairs judged unobserved**: firing rates *overwritten with 0.0 Hz* rather than the trial or unit being removed — i.e. missing data is represented as genuine silence.
- **Trial with no sample-start event**: tone onset replaced by the nominal −1.85 s offset, counted in `stats['sample_onset_fallbacks']`.
- **Trial with no lick at all**: choice filled from `trial_instruction` (see 5-a).
- **Bin with no camera frame**: NaN, excluded from the percentile computation, then implicitly labelled class `0`.
- **Unit with an empty `anno_name`**: raises `ValueError` (never triggered in practice, since such sessions have no good units).

ii.
```python
        if len(good_unit_idx) == 0:
            print(f"Skipping {session_id}: zero good units")
            return None
```
```python
            sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
            sample_onset_fallbacks += 1
```
```python
        if np.any(invalid_trials):
            fr[invalid_trials] = 0.0
    ...
    nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```
```python
        brain_region_names = decode_str_array(units["anno_name"][good_unit_idx])
        if np.any(brain_region_names == ""):
            raise ValueError(f"{session_id}: found kept good units with empty anno_name")
```

iii. Step 5 Key Decision 9 and Step 10 "Issues Found and Resolved" describe the trial-mapping fix as the main robustness measure: "Replaced that assumption with raw go-time plus `obs_intervals` selection … use `is_good_trials` directly only when its column count matches the selected trials." The choice and tone fallbacks are justified as the minimum needed to fill a required output/input on trials where the raw event is absent. There is no discussion of the risk that overwriting rates with 0 Hz fabricates data, and the resulting 45 % trial loss is presented in Step 9 as intentional strictness rather than investigated.

## 10-a. What are the most time-consuming steps of the code?

i. File I/O dominates: `units["spike_times"][:]` pulls the whole session spike buffer (up to ~11.5 M doubles), the tongue tracking array is ~680 k × 3, and `units["is_good_trials"][good_unit_idx, :]` is a fancy-indexed read of an `(n_units, n_trials)` bool table. After I/O, the per-unit `searchsorted` loop over 81 × n_trials edges is the main compute cost. Measured: 265.99 s total for 174 files, mean 1.54 s per kept session, ranging from ~0.5 s to ~8.4 s roughly with unit count; pickling the 3.7 GB output adds to that. Timing is printed per session (elapsed + running total) but not broken down per processing stage within a session.

ii.
```python
        print(
            f"[{idx}/{len(target_files)}] {result.session_id}: "
            f"{result.stats['n_good_units']} good units, {result.stats['n_trials']} trials, "
            f"{elapsed:.2f}s (running total {running:.2f}s)"
        )
```

iii. Step 6 identifies the bottlenecks as "Full PyNWB object loading produces avoidable overhead" and "Per-spike/per-trial Python loops would be too slow for the full dataset", and Step 7 estimates ~1.6 s/kept session → "~4.6 minutes for 173 kept sessions", comfortably inside the 15-minute budget in the instructions, so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The two expensive loops are already collapsed as far as the ragged storage allows: spike binning loops over units only (all trials × bins are searched in one flattened `searchsorted`), and tongue binning is fully vectorised over trials and bins. What remains are four per-trial Python loops that could all be vectorised with array operations: the sample-onset lookup, the input construction, the choice inference (which does six `searchsorted` calls per trial into the lick arrays), and the output assembly. The human reference vectorises all four. They are cheap relative to I/O, so the practical saving would be small.

ii.
```python
    for trial in range(n_trials):
        events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
        ...
    for trial in range(n_trials):
        inp = np.zeros((2, n_bins), dtype=np.float32)
        ...
    for trial in range(n_trials):
        choice_val, source = infer_choice_for_trial(...)
        ...
    for trial in range(n_trials):
        out = np.empty((4, n_bins), dtype=np.int8)
```
Already vectorised:
```python
        edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
```

iii. Step 6 lists the vectorisations that were done ("ragged-array indexing", "vectorized `np.searchsorted` across all trial/bin edges per unit", "vectorized timestamp-to-bin alignment for tongue tracking") and Step 7 reports the resulting runtime as acceptable; the remaining per-trial loops are not discussed, presumably because the measured total (~4.4 min) was already well inside budget.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and no session is processed twice, so there is no large-scale recomputation. Smaller repetitions: `bin_edges_and_centers()` is rebuilt per session (and the bin centres are computed a third time in `build_dataset` for `metadata['bin_centers_s']`); `infer_choice_for_trial` re-runs `searchsorted` into the full lick arrays six times per trial instead of once per array; `event_slices_for_trials` is run for the delay stream on every session although the result is only consumed by the optional plot; and the string trial columns are decoded/lower-cased over all behavioural trials before being subset to the kept trials.

ii.
```python
def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
```
```python
        "bin_centers_s": (np.arange(WINDOW_START_S + BIN_WIDTH_S / 2.0, WINDOW_END_S, BIN_WIDTH_S)).astype(np.float32),
```
```python
    left_start = np.searchsorted(left_lick_times, trial_start, side="left")
    left_go = np.searchsorted(left_lick_times, go_time, side="left")
    left_stop = np.searchsorted(left_lick_times, trial_stop, side="right")
    right_start = np.searchsorted(right_lick_times, trial_start, side="left")
    ...
```

iii. Not discussed explicitly in CONVERSION_NOTES.md; the notes only claim the single-pass design ("Avoid unnecessary file I/O" was addressed by reading each file once with `h5py`). The per-session tongue percentiles genuinely require nothing more than one pass, so no second pass over the data is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, all cheap relative to I/O:
- `tongue_likelihood` is read and carried in the plot payload but never used to affect any output (by explicit decision, 8-a).
- `bin_tongue_y` returns `tongue_valid`, which is assigned and never used.
- `delay_start_times` and the per-trial delay slice indices are computed for every session but consumed only by the optional `--show-processing` plot.
- `units["is_good_trials"][good_unit_idx, :]` is always read (and `is_good_trials` is materialised as an all-ones array) even in the 120/173 sessions that take the other branch and never consult it.
- `trial_stop`, `trial_instruction`, `early_lick` and several `stats` entries (`fraction_invalid_unit_trials`, `mean_good_trial_fraction_per_good_unit`, `used_direct_is_good_trials`) are maintained through the mask-subsetting block but do not enter the saved dataset.
- Rates are computed in `float32` then downcast to `float16`, and the decoder upcasts them again to `float32` at training time.

ii.
```python
    tongue_y_binned, tongue_valid = bin_tongue_y(...)   # tongue_valid never used again
```
```python
    delay_slice_starts, delay_slice_ends = event_slices_for_trials(delay_start_times, trial_start, trial_stop)
```
```python
            "tongue_likelihood_window": tongue_likelihood[tongue_window],
```
```python
        else:
            is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)
```

iii. Not discussed in CONVERSION_NOTES.md. The retained diagnostics are consistent with the instruction to produce `--show-processing` plots and per-session statistics, so most of this work is intentional instrumentation rather than waste; the unused `tongue_valid` return and the always-read `is_good_trials` table are simple leftovers of the Step 10 refactor.
