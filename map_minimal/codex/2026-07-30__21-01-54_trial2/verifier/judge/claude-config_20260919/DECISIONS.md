# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is read directly from the DANDI NWB release mounted at `/app/data`, which stores one NWB file per session under `sub-<subject_id>/`. The AI enumerates every session with a single sorted glob (`sub-*/*.nwb`, 174 files) and opens each file **once** with raw `h5py` rather than `pynwb`, reading the HDF5 groups directly (`intervals/trials`, `units`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `general/subject`). Ragged NWB columns (`spike_times`, `obs_intervals`) are sliced manually using their companion `*_index` datasets. String columns are decoded with a helper that handles `bytes`/`np.bytes_`. Everything needed for a session (trials table, behavioural events, tongue tracking, units, spikes) is pulled inside one `with h5py.File(...)` block, and results are accumulated into per-session lists that are assembled into the target dictionary at the end.

ii.
```python
def convert_dataset(data_dir, exclude_auto_free=True):
    data_dir = Path(data_dir)
    nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not nwb_paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
...
    for session_number, path in enumerate(nwb_paths, start=1):
        with h5py.File(path, "r") as f:
            trials = f["intervals/trials"]
            n_trials = len(trials["id"])
            ...
            be = f["acquisition/BehavioralEvents"]
            tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
            units = f["units"]
```

```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
```

iii. From the trajectory: the AI first explored the files with `pynwb` (steps 21–37) to learn the schema, then explicitly *switched to h5py* for speed — step 38: "I'm switching to faster HDF5-level inspection so I can find the exact inclusion rule rather than guessing which session to drop." It verified whole-dataset counts against the reference texts (174 files, 69,453 classifier-good units, 28 subjects) before committing to the loader, and documented the 174-file vs. 173-session question in `CONVERSION_NOTES.md`. Note that brain-region labels are taken from the *full* `anno_name` string (293 distinct labels, e.g. `"Orbital area, lateral part, layer 5"`) rather than being collapsed to a coarse region name.

## 1-b. How are the data split into subjects?

i. Each session's animal is read from the NWB file's own subject record, `general/subject/subject_id` (a numeric string such as `'440956'`). The AI builds the `subjects` list in first-encounter order using a dict, and appends the corresponding index to `subject_idx` in the same order that sessions are appended to `neural`/`input`/`output`. Because files are globbed as `sub-*/...` and sorted, first-encounter order is also alphabetical. The result is 28 subjects with 3–10 sessions each.

ii.
```python
            subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
            subject = str(subject)
            if subject not in subject_to_idx:
                subject_to_idx[subject] = len(subjects)
                subjects.append(subject)
            subject_idx.append(subject_to_idx[subject])
```
```python
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The AI did not discuss this at length; the trajectory shows it inspected `general/subject` when mapping the NWB schema and then used the canonical in-file subject identifier. The directory name (`sub-440956`) is derived from the same id, so no separate grouping step is needed. The `CONVERSION_NOTES.md` records the resulting 28 subjects as a check against the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as exactly one session — no grouping or splitting is performed. Session order in all output lists follows the sorted file list (which, because the filename embeds the acquisition timestamp, is chronological within each subject). A session is dropped only if it has no classifier-`good` units, or if fewer than 2 trials survive curation. 173 of 174 files reach the output.

ii.
```python
    nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
    ...
    for session_number, path in enumerate(nwb_paths, start=1):
        with h5py.File(path, "r") as f:
```
```python
            if len(good_unit_indices) == 0:
                continue
```
```python
            if n_keep_trials < 2:
                continue
```
Per-session identity is recorded in the side-car summary file, not in the pickle:
```python
            summary["per_session"].append(
                {
                    "session_file": path.name,
                    "subject": subject,
                    "trials_raw": int(n_trials),
                    "trials_kept": int(n_keep_trials),
                    ...
```

iii. Step 96: "The full conversion completed cleanly and, importantly, it converged to `173` usable sessions once classifier-good units were enforced." The AI reconciled the 174-file/173-session discrepancy against `methods.txt` and documented it in `CONVERSION_NOTES.md` and in a `reference_summary_discrepancy_note` metadata field. Caveat: unlike the reference, the converted pickle's `metadata` contains **no** `session_info` (session ids are only in `/app/conversion_summary.json`), so session identity is not recoverable from the delivered dataset.

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI checked in advance (step 55) that `go_start_times` has exactly one event per trial in all 174 sessions, and it hard-asserts this per session at conversion time, so trial *k* of the table corresponds to go cue *k*. It does not attempt to re-derive trial boundaries from event streams (it had observed that `sample_start_times`/`delay_start_times` can have several entries per trial because early licks replay those epochs).

ii.
```python
            trials = f["intervals/trials"]
            n_trials = len(trials["id"])
            trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
            trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
```
```python
            go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
            if len(go_starts) != n_trials:
                raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
```

iii. The AI ran an explicit dataset-wide check before writing the converter (step 55/56: `go_count_mismatches 0`), establishing that the trials table and the go-cue stream are one-to-one, which makes the trials table an unambiguous definition of a trial.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied in this order:
1. **`auto_water == 0` and `free_water == 0`** — the "regular trial" definition taken from the method paper's own code (`population_decoding_utils.py`: "No early lick, no auto water, no free water"). This removes 1,338 auto-water and 2,450 free-water trials.
2. **Trials with valid ephys**, built from `units/obs_intervals` of the first good unit mapped back onto `trials.start_time`/`stop_time`, intersected with `units/is_good_trials` requiring *every* good unit to be flagged good on that trial. This is stricter than an `obs_intervals`-only filter (92,801 vs 93,310 trials dataset-wide).
3. **Post-hoc removal of trials whose binned neural matrix is entirely zero** (2 trials in the whole dataset).
4. **Sessions with fewer than 2 surviving trials are dropped.**

Early-lick, `ignore`, and photostim trials are deliberately **kept** because they are required decoder outputs/inputs. Net result: 89,068 of 94,370 trials retained (94.4%).

ii.
```python
            trial_keep = np.ones(n_trials, dtype=bool)
            if exclude_auto_free:
                trial_keep &= auto_water == 0
                trial_keep &= free_water == 0
```
```python
            is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
            ...
                n_recorded_trials = min(len(first_obs_intervals), is_good_trials.shape[1])
                common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
                interval_starts = first_obs_intervals[:n_recorded_trials, 0]
                interval_stops = first_obs_intervals[:n_recorded_trials, 1]
                mapped_idx = np.searchsorted(trial_starts, interval_starts, side="left")
                valid = mapped_idx < n_trials
                valid &= np.isclose(trial_starts[mapped_idx], interval_starts, atol=1e-4)
                valid &= np.isclose(trial_stops[mapped_idx], interval_stops, atol=1e-4)
                if np.sum(valid) == n_recorded_trials:
                    recorded_trial_mask[mapped_idx[common_obs_mask]] = True
                else:
                    recorded_trial_mask[:n_recorded_trials] = common_obs_mask
            trial_keep &= recorded_trial_mask
```
```python
            nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
            zero_only_trials = int((~nonzero_trial_mask).sum())
            session_neural = session_neural[nonzero_trial_mask]
            ...
            if n_keep_trials < 2:
                continue
```

iii. The auto/free-water exclusion is taken directly from the method paper's regular-trial definition (step 66: "exclusion of `auto_water`/`free_water` trials while keeping early/ignore trials because those are required decoder targets"). The ephys mask was arrived at iteratively and empirically: step 104 — "some late trials in a few sessions are entirely zero across every neuron … so I'm checking the NWB `is_good_trials` masks"; step 107 — "`is_good_trials` only spans the first `160` or `206` trials, while the behavior table continues to `480+`"; step 135 — "in some sessions, the valid ephys trials are a trailing block, not a leading block. The NWB `obs_intervals` table gives the exact recorded trial start/stop windows, so I'm switching the mask construction to map those intervals back onto the behavioral trial table." The final all-zero sweep is justified in step 145: "they carry no neural information and there's no reason to leave them in once they're identified." The 2-trial minimum comes from the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (with `units/spike_times_index` to de-ragged it), restricted to units with `units/classification == 'good'`, plus `acquisition/BehavioralEvents/go_start_times/timestamps` to place the bins. No other neural representation is used.

ii.
```python
            units = f["units"]
            classification = _decode_str_array(units["classification"]).reshape(-1)
            good_unit_indices = np.flatnonzero(classification == "good")
```
```python
            for pos, unit_idx in enumerate(good_unit_indices):
                unit_spikes = _extract_unit_spikes(units, int(unit_idx))
```

iii. The AI enumerated the `units` group (step 40) and confirmed that spike times are the only neural data in the file; it also verified that the NWB files "already contain the post-QC classifier labels the paper describes" (step 30), so unit selection could be done from the file itself.

## 2-b. How is the `neural` data processed?

i. Spike counts in non-overlapping 50 ms bins, divided by the bin width to give firing rate in Hz, stored as `float32`. For each good unit, the flat array of absolute bin edges for all kept trials is passed to a single `np.searchsorted`, and differencing adjacent positions gives the per-bin count. No smoothing, no normalisation, no baseline subtraction, no spike-time truncation beyond the window.

ii.
```python
            abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
            ...
            for pos, unit_idx in enumerate(good_unit_indices):
                unit_spikes = _extract_unit_spikes(units, int(unit_idx))
                spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
                spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
                session_neural[:, pos, :] = spike_counts
```

iii. Documented in `CONVERSION_NOTES.md` as "Neural data: firing rates in Hz from non-overlapping spike-count bins" and in metadata as `"non-overlapping 50 ms spike-count bins converted to firing rates (Hz)"`. This mirrors the method paper's `sliding_histogram(..., rate=True)`, which the AI read at step 16 and which returns `binSpikes/bin_width`; the paper's version uses a sliding window with a stride, but here bin width = stride = 50 ms is fixed by the task instructions, giving non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit inclusion is exactly the NWB spike-sorting QC classifier verdict: keep units where `units/classification == 'good'`; no individual metric (isi_violation, presence_ratio, amplitude_cutoff, …) is thresholded, and the older `unit_quality` label is not used. A session with zero such units is dropped. 69,453 good units are retained across 173 sessions (median ~390/session, min 90, max 923).

ii.
```python
            classification = _decode_str_array(units["classification"]).reshape(-1)
            anno_name = _decode_str_array(units["anno_name"]).reshape(-1)
            good_unit_indices = np.flatnonzero(classification == "good")

            if len(good_unit_indices) == 0:
                continue
```

iii. Step 30: "I've confirmed the NWB files already contain the post-QC classifier labels the paper describes." The AI compared `classification` against `unit_quality` across the whole dataset (step 42) before choosing, and matched its total against the white paper: `CONVERSION_NOTES.md` records 69,453 classifier-good units vs. the 69,943 quoted in `methods.txt`, and flags the difference rather than tuning the filter to fit. It identified `sub-440958_ses-20190216T162508` as the single session with zero good units and dropped it, resolving the 174-file/173-session mismatch.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go cue onset (`go_start_times`). All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: the fixed grid of 81 edges relative to the go cue is added to each trial's go-cue time to produce absolute bin edges, and spikes are binned against those edges.

ii.
```python
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
```
```python
            abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
            abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
```
```python
        "temporal_alignment_event": "go cue onset",
        "off_start": float(WINDOW_START_S),
        "off_end": float(WINDOW_END_S),
```

iii. `CONVERSION_NOTES.md`: "Alignment event is go cue onset from `acquisition/BehavioralEvents/go_start_times/timestamps`. Go cue count matches trial count in each converted session." Step 77 reports a sanity check on a real session: "80 bins, resolved tone onset at `-1.85 s`, expected unit counts". Note that the method paper's own exported data already have spike times expressed relative to the go cue, which the AI read at step 17 ("in Susu's data, spike_times are relative to go cue time"), so aligning to the go cue reproduces the reference convention.

## 2-e. What is the temporal resolution of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial, spanning −2.5 s to +1.5 s relative to the go cue. The grid is defined once at module level and reused for every trial and every session, so all trials have identical shape `(n_neurons, 80)`. There is no rebinning of an intermediate representation: spikes go from raw times straight into the final 50 ms bins. `time_bin_size` is stored as 50.0 ms, along with the explicit edge and centre lists.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)
```
```python
            "time_bin_size": float(BIN_SIZE_S * 1000.0),
            "n_timepoints": int(N_BINS),
            "bin_edges_s": BIN_EDGES_REL.tolist(),
            "bin_centers_s": BIN_CENTERS_REL.tolist(),
```

iii. The window and bin width are taken verbatim from the Decoder Task section of the instructions. The `+ 1e-9` guard on `np.arange` is there to avoid a floating-point short-by-one edge, the same numerical concern the method paper's `sliding_histogram` handles with an `np.allclose` check. `CONVERSION_NOTES.md`: "Each kept trial is represented with 80 bins of width 50 ms, aligned to go cue onset, spanning `-2.5 s` to `+1.5 s`."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets), combined with each trial's go cue and, for a fallback branch, `trials.start_time`. For each trial the tone is the **last** `sample_start_times` event at or before that trial's go cue; if that event falls before the trial's own start, the AI re-searches within `[trial_start, go]` and raises if nothing is found.

ii.
```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    valid = sample_idx >= 0
    sample_onsets = np.full(go_starts.shape, np.nan, dtype=np.float64)
    sample_onsets[valid] = sample_starts[sample_idx[valid]]

    invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
    if np.any(invalid):
        for i in np.where(invalid)[0]:
            mask = (sample_starts >= (trial_starts[i] - 1e-9)) & (sample_starts <= (go_starts[i] + 1e-9))
            candidates = sample_starts[mask]
            if len(candidates) == 0:
                raise RuntimeError(f"Could not resolve sample onset for trial {i}.")
            sample_onsets[i] = candidates[-1]
    return sample_onsets
```

iii. `CONVERSION_NOTES.md`: "In these NWB files, `sample_start_times` can contain replayed sample epochs after early licks, so naive one-to-one trial matching is wrong. For each trial, tone onset is resolved as the last `sample_start_times` timestamp occurring before that trial's go cue, with a fallback constrained to the trial's own start-to-go interval." The AI had verified at steps 37/55 that `sample_start_times` has more entries than trials while `go_start_times` does not.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying channel: for each trial, the tone time is expressed relative to that trial's go cue (`tone_rel`, normally ≈ −1.85 s), and the input value at each bin is the bin centre minus `tone_rel`, i.e. seconds elapsed since the tone at the centre of that bin. Stored as `float32` in row 0 of the `(2, 80)` input array, named `time_from_tone_onset_s`. The realised range over the dataset is [−1.5, 11.9] s, identical to the reference solution's [−1.525, 11.894].

ii.
```python
            for local_idx, trial_idx in enumerate(keep_trial_indices):
                tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
                time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
                ...
                input_trial = np.vstack([time_from_tone, stim_on]).astype(np.float32)
```
```python
        "input_names": ["time_from_tone_onset_s", "photostimulation_on"],
        ...
            "input_processing": {
                "time_from_tone_onset_s": "bin center minus final sample-start time for that trial",
```

iii. The AI treats this as a per-bin continuous value rather than a binary onset marker; its stated rationale (`CONVERSION_NOTES.md`) is simply "Input channel `time_from_tone_onset_s` is computed at each bin center", which is what the instructions request ("Time from **tone onset** in seconds (continuous, time-varying)"). Step 77 records a per-session sanity check of the resolved value (−1.85 s).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the same grid that defines the neural bins: `BIN_CENTERS_REL` are the centres of the same 80 go-cue-relative bins used to count spikes, so element *k* of the input is by construction the same 50 ms interval as column *k* of the firing-rate matrix. No interpolation or shifting.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
...
            abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]   # neural
            abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
                time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. No separate justification is given; alignment is implicit in reusing the single module-level grid for every stream.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials table columns `photostim_onset` and `photostim_duration` (stored as strings, with `'N/A'` on unstimulated trials), together with `trials.start_time` (the onsets are measured from trial start) and the go cue (to express them on the bin axis). `photostim_power` is also parsed but never used.

ii.
```python
            photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
            photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
            photostim_power = _parse_optional_float_array(trials["photostim_power"])
```
```python
def _parse_optional_float_array(dataset):
    raw = _decode_str_array(dataset).reshape(-1)
    out = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for i, value in enumerate(raw):
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
    return out
```

iii. `CONVERSION_NOTES.md`: "Trial photostim timing uses `intervals/trials/photostim_onset` and `photostim_duration`. `photostim_onset` is interpreted relative to trial start, consistent with the NWB event timing." The AI inspected stimulated trials directly (step 48, printing `start_time`, `stop_time`, `photostim_onset`, `photostim_duration`, `photostim_power`) before fixing the interpretation, and had also read the method paper's convention that laser on/off times are "with reference to trial start time".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying channel (row 1 of the input array): 1 where the bin centre falls in `[stim_start, stim_stop)` in absolute time, 0 elsewhere. Unstimulated trials have NaN onset/duration and are skipped entirely, leaving an all-zero row. Stored as `float32`.

ii.
```python
                stim_on = np.zeros(N_BINS, dtype=np.float32)
                if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
                    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
                    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
                    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```
```python
                "photostimulation_on": "1 if bin center falls within trial photostim interval, else 0",
```

iii. `CONVERSION_NOTES.md`: "`photostimulation_on` is 1 when the bin center falls inside the photostim interval." The instructions ask for "Whether **photostimulation** is on at every time point (discrete, time-varying)", so a per-bin binary indicator rather than a per-trial flag is the literal requirement; photostim trials are explicitly retained ("keep_photostim_trials": True in the metadata).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Rather than converting the stimulus window to go-cue-relative time, the AI converts the *bin centres* to absolute time (`abs_centers = go + BIN_CENTERS_REL`) and compares them with the absolute stimulus start/stop derived from `trial_start + photostim_onset`. Since `abs_centers` is the same grid used to build `abs_edges` for the spikes, the resulting indicator is bin-for-bin aligned with the firing rates.

ii.
```python
            abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
                    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
                    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
                    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. Everything in the NWB file is on a single session-absolute clock, so the AI's choice to do the comparison in absolute time avoids having to re-reference the onset twice; the result is numerically the same as re-expressing the stimulus relative to the go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Only from `trials.trial_instruction` — the side the tone **instructed** the animal to lick. The actual lick direction is not derived: `outcome` is not consulted when forming `choice`, and the `left_lick_times`/`right_lick_times` behavioural event streams (which the AI had enumerated at step 37) are never read. The result is a two-valued output, `left = 0` / `right = 1`, with no "no lick" class.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
```
```python
            trial_instruction = _decode_str_array(trials["trial_instruction"]).reshape(-1)
...
                choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```
```python
        "output_values": [
            ["left", "right"],
            ...
            "choice": "mapped from trial_instruction to match the paper code's left/right trial label",
```

iii. `CONVERSION_NOTES.md`: "The requested output says lick direction choice. The reference code uses the paper's left/right trial label (`trial_type` in the original code path), which corresponds to `trial_instruction` in the NWB release. To preserve ignore trials and stay consistent with the reference processing, `choice` is encoded from `trial_instruction`." The AI had indeed seen `sess_dict['trial_type'] = 1*(trial_type=='l')` in `preprocessing_DJ_2022Aug.py` (step 13/16). It had also seen, in the same file, `behavior_lick_directions` — "ndarray, (n_trials). Each element is a list of 0 (lick left) and 1 (lick right)" — i.e. the reference pipeline does carry actual lick direction separately from trial type, but the AI did not use the NWB equivalent.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A dictionary lookup per trial mapping the instruction string to 0/1, tiled across all 80 bins so the value sits in row 0 of the `(4, 80)` integer output array (`int16`). No `no lick` category exists, so `ignore` trials (14.9% of the data) are labelled with the instructed side, and `miss` trials (16.6%, where the animal licked the *opposite* side) are labelled with the side it did not lick. Dataset-wide the output splits 48.6% left / 51.4% right.

ii.
```python
                choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
...
                output_trial = np.vstack(
                    [
                        np.full(N_BINS, choice_value, dtype=np.int16),
                        np.full(N_BINS, outcome_value, dtype=np.int16),
                        np.full(N_BINS, early_value, dtype=np.int16),
                        tongue_cat,
                    ]
                )
```

iii. The only justification offered is consistency with the method paper's `trial_type` variable (see 5-a); the AI does not discuss the resulting loss of the "no lick" class or the mislabelling of error trials, and its `task_description` re-labels the variable as "left/right trial type", i.e. it is aware the quantity is the instruction, not the choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials.outcome`, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` (verified across the dataset at steps 25/42).

ii.
```python
            outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
```

iii. The trials table stores the outcome explicitly with the three categories requested by the instructions, so nothing has to be derived.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2`, exactly the coding given in the instructions; the per-trial value is tiled across all 80 bins into row 1 of the output array, and `output_values[1] = ['ignore', 'miss', 'hit']`. Realised distribution: 14.9% ignore, 16.6% miss, 68.5% hit — identical to the reference.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
```
```python
                outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```
```python
            ["ignore", "miss", "hit"],
```

iii. `CONVERSION_NOTES.md`: "Mapped directly from NWB `outcome`: `ignore -> 0`, `miss -> 1`, `hit -> 2`." Per-trial values are repeated across bins so that all four outputs can live in one `(n_output, n_timepoints)` array, as the target format prefers time-varying outputs.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `trials.early_lick`, whose only values in this dataset are `'no early'` and `'early'`.

ii.
```python
            early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
```

iii. The flag is stored explicitly per trial, so no derivation from lick times is needed; the AI confirmed the value set when it first inspected the trials table (step 25).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early → 0`, `early → 1` via a fixed dictionary, tiled across all 80 bins into row 2. `output_values[2] = ['no', 'yes']`. Realised distribution 88.4% / 11.6%, matching the reference. Early-lick trials are explicitly retained by the trial filter even though the data paper excludes them, because the instructions require this output.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
```
```python
                early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
```
```python
                "keep_early_lick_trials": True,
```

iii. Step 66: "exclusion of `auto_water`/`free_water` trials while keeping early/ignore trials because those are required decoder targets." The lick that sets the flag occurs in the sample or delay epoch, i.e. inside the −2.5 s window, so a per-trial constant is still decodable from the extracted window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of `data` (`tongue_y`) plus the matching `timestamps`, and the go cue for windowing. Column 2 of the same array (`tongue_likelihood`, the tracker's confidence that the tongue is visible in that frame) is read into the array slice but **never used** — the AI slices only `data[:, 1]`.

ii.
```python
            tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
            tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
            tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
            tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
```

iii. The AI inspected the series (step 32, printing `description`, shape, timestamps and the first row) and verified at step 31 that tongue tracking exists in all 174 sessions. It grepped the method paper's code for `likelihood` (step 54) and found the reference pipeline's marker handling (`align_markers.py`) uses `tongue_y` without any confidence gating, which is presumably why the likelihood channel was dropped; this is not explicitly argued anywhere in the notes or trajectory.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps. (1) Per session, the 40th and 60th percentiles of the **raw, unfiltered** full-session `tongue_y` trace (all camera frames, ~10⁶ per session) are computed once. (2) Per trial, each 50 ms bin is assigned the value of the **last camera frame falling inside that bin** (`searchsorted` on the camera timestamps at the bin end, then a check that the sample is at or after the bin start); bins containing no frame keep the initialised value `0.0`. No averaging, no interpolation, no NaN handling, and no visibility masking.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled
```
```python
            tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
```

iii. `CONVERSION_NOTES.md`: "The reference alignment code uses the most recent marker sample within each time window rather than interpolation. For each 50 ms neural bin, tongue y is sampled as the last camera sample within that bin." This is a faithful reading of `align_markers.py`, which the AI read at step 52 and which does `marker_vecs[i, trial, :] = _this_embed_array[:, np.where(mask)[0][-1]]` into a zero-initialised array. The percentile step is described as "Per-session discretization uses the full-session tongue-y distribution". No rationale is given for computing the percentiles over frames in which the tongue is not protruding; in practice the tongue is visible in only ~10–14% of frames, so the 40th and 60th percentiles collapse to within a fraction of a pixel of each other (e.g. 280.994 vs 281.241 in `sub-440956_ses-20190208T133600`, vs 276.3/289.9 if only visible frames are used).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 for `y < q40`, 1 for `q40 ≤ y ≤ q60`, 2 for `y > q60`, where `q40`/`q60` are the per-session percentiles from 8-b. The instructions' fourth category, `3: not visible`, is **not implemented** — `output_values[3]` has length 3. Because unfilled bins carry the sentinel value `0.0` and `q40` is typically ~280 px, every bin with no camera frame is silently assigned class 0 ("below 40th percentile"). Realised distribution 39.6% / 18.6% / 41.8%, against the reference's 9.7% / 5.1% / 10.3% / 75.0% not-visible.

ii.
```python
                tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
                tongue_cat = np.zeros(N_BINS, dtype=np.int16)
                tongue_cat[tongue_y_trial >= tongue_q40] = 1
                tongue_cat[tongue_y_trial > tongue_q60] = 2
```
```python
            ["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"],
```
```python
                "tongue_y_position": "per-session 40/60 percentile bins from the full tongue_y session trace",
```

iii. `CONVERSION_NOTES.md` states the three-way rule verbatim from the instructions ("`< 40th percentile -> 0`, `40th to 60th percentile -> 1`, `> 60th percentile -> 2`") but stops there; neither the notes nor the trajectory mention the "3: not visible" category from the Decoder Task specification, and there is no discussion of what happens to bins with no camera coverage.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes, so alignment is done by building the *same* go-cue-relative bin edges (`go + BIN_EDGES_REL`) and searching the camera timestamp array against them. Bin *k* of the tongue output therefore covers exactly the interval of bin *k* of the firing rates. No interpolation or clock correction is applied. The video is trial-gated (off during the inter-trial interval), so on trials whose go cue arrives less than 2.5 s after trial start the leading bins have no frames — those fall through to the `0.0` sentinel and hence to class 0.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
```
```python
                tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```

iii. Implicit in reusing the module-level grid; the AI's sanity check at step 77 reported "sensible tongue ranges" for a real session. The "last sample in the window" convention is taken from the method paper's `align_markers_between_lims`, which uses a window of one frame period ending at each output time point.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases, handled in four different ways:
- **Session never quality-controlled** (`classification`/`anno_name` are NaN): decodes to a non-`'good'` string, the session has no good units and is skipped — this is exactly the one dropped file.
- **Trials outside the ephys recording** (behaviour continues after the probe recording stops, or starts before it): dropped via the `obs_intervals` × `is_good_trials` mask, with a fallback branch if the interval→trial mapping cannot be verified to 1e-4.
- **Trials that are all-zero after binning** (2 in the whole dataset): detected and dropped post hoc.
- **Missing photostim fields** (`'N/A'`): parsed to NaN and treated as "no stimulation" (all-zero indicator row).
- **Unresolvable tone onset**: the fallback search raises a `RuntimeError` rather than guessing (it never fires on this dataset).
- **Bins with no camera frame** (and frames in which the tongue is not protruding): *not* treated as missing — the bin silently takes the value `0.0` and is then classified as "below 40th percentile".

ii.
```python
            if len(good_unit_indices) == 0:
                continue
```
```python
                if np.sum(valid) == n_recorded_trials:
                    recorded_trial_mask[mapped_idx[common_obs_mask]] = True
                else:
                    recorded_trial_mask[:n_recorded_trials] = common_obs_mask
```
```python
            nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
            ...
            if n_keep_trials < 2:
                continue
```
```python
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
```
```python
            if len(candidates) == 0:
                raise RuntimeError(f"Could not resolve sample onset for trial {i}.")
```
```python
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
```

iii. The ephys-coverage handling was reached by iterating on verifier warnings (steps 104 → 107 → 135 → 145), each time tracing the warning to a concrete data property rather than suppressing it: "That points to behavioral trials extending past the valid ephys interval"; "in some sessions, the valid ephys trials are a trailing block, not a leading block". The philosophy is stated in `CONVERSION_NOTES.md` — where nothing was recorded, exclude; and the reference-vs-local unit-count discrepancy is "documented in the dataset metadata instead of being hidden". The one gap is video coverage, where missing data is imputed with a sentinel that is indistinguishable from a real low tongue position.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion of 174 sessions took roughly 5 minutes end to end (trajectory step 84 → 96, ≈4 min 47 s, including writing the 11.7 GB pickle). Within a session the dominant costs are, in order: (1) one HDF5 read per good unit for that unit's `spike_times` slice plus the `searchsorted` of 81×n_trials edges against it — this runs 69,453 times dataset-wide and scales with unit count (90–923 per session); (2) reading the tongue-tracking arrays (~10⁶ frames × 3 columns per session) and the full-session `np.percentile`; (3) the per-trial Python loop that builds inputs/outputs and calls `_bin_tongue_y` once per trial, each call doing a `searchsorted` against the ~10⁶-element camera timestamp array; (4) `pickle.dump` of the 11.7 GB result. `np.diff`/`searchsorted` on the flattened edge array means the trial dimension is already vectorised inside the unit loop.

ii.
```python
            for pos, unit_idx in enumerate(good_unit_indices):
                unit_spikes = _extract_unit_spikes(units, int(unit_idx))
                spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
                spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
                session_neural[:, pos, :] = spike_counts
```
```python
    with open(args.full_out, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI never profiled the conversion or discussed its cost; it did, however, switch from `pynwb` to `h5py` explicitly for speed early on (step 38), and it prints a progress counter every 10 sessions so the run can be monitored (steps 85–95 show it polling that counter). The whole run comfortably fits the time budget, so no optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four loops are avoidable, the largest being the per-trial output/input loop:
- `for local_idx, trial_idx in enumerate(keep_trial_indices)` — computes `time_from_tone`, `stim_on`, the tongue bins and the three categorical rows one trial at a time. All four are pure array arithmetic over `(n_trials, 80)` and could be computed for the whole session at once (the reference solution does exactly this with broadcasting).
- `_bin_tongue_y` is called once per trial and re-runs `searchsorted` over the entire session camera timestamp vector each time; a single global bin index plus one `bincount`/`searchsorted` would cover all trials.
- `_parse_optional_float_array` and `_decode_str_array` loop in Python over every trial/unit string; `np.asarray(...).astype('U')` plus vectorised comparison would do the same work.
- the `for i, unit_idx in enumerate(good_unit_indices)` region-mapping loop could be a dict lookup over `np.unique`.

The per-unit spike loop is *not* avoidable: each unit has a different number of spikes, so there is no single sorted array to search — that is inherent to the ragged storage.

ii.
```python
            for local_idx, trial_idx in enumerate(keep_trial_indices):
                tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
                time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
                ...
                tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
```
```python
            good_region_idx = np.empty(len(good_unit_indices), dtype=np.int64)
            for i, unit_idx in enumerate(good_unit_indices):
                region_name = str(anno_name[unit_idx])
```
```python
    for i, value in enumerate(raw):
        if value in ("N/A", "", None):
            continue
        out[i] = float(value)
```

iii. Not discussed in the trajectory. The AI did vectorise the expensive part — `abs_edges`/`abs_centers` are built once per session as `(n_trials, 81)` arrays outside the unit loop, so the per-trial Python loops only touch cheap 80-element operations and cost a small fraction of the ~5 min runtime.

## 10-c. What processing does the code repeat multiple times?

i. Little genuine recomputation, with three exceptions:
- `_bin_tongue_y` re-searches the full session camera timestamp array on every trial, instead of computing one global frame→bin index for the session.
- The session-level neural array is fully materialised and then re-filtered and re-copied twice (the all-zero mask, then `np.ascontiguousarray` per trial), and `session_input`/`session_output` lists are rebuilt by list comprehension after the same mask.
- `_decode_str_array` is applied separately to six trials-table columns and two unit columns, each doing its own Python-level decode pass.

Each NWB file is opened exactly once, the bin grid is built once at module level, and the sample pickle is derived from the already-converted in-memory dictionary (`_subset_data`) rather than by re-reading any file.

ii.
```python
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1   # once per trial, over all session frames
```
```python
            nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
            session_neural = session_neural[nonzero_trial_mask]
            session_input = [session_input[i] for i in np.flatnonzero(nonzero_trial_mask)]
            session_output = [session_output[i] for i in np.flatnonzero(nonzero_trial_mask)]
            ...
            neural_sessions.append([np.ascontiguousarray(session_neural[i]) for i in range(n_keep_trials)])
```
```python
    sample_sessions = list(range(min(args.sample_sessions, len(data["neural"]))))
    sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
```

iii. Reusing the in-memory result for the sample pickle is deliberate — the AI wanted a small artifact for fast verifier/decoder iteration (steps 97–103) without paying for a second conversion pass. During development the *whole conversion* was rerun three times (steps 84, 110, 138, 147) as the trial-curation rule was corrected, but that is iteration, not repetition inside the script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount, none of it expensive:
- `photostim_power` is parsed (a Python loop over every trial string) and never used.
- `anno_name` is decoded for **all** units, though only the good ones are needed.
- `trial_stops` is read and used only for the `obs_intervals` consistency check and the summary's `mean_trial_duration_s`.
- Per-session diagnostic statistics (`mean_trial_duration_s`, `mean_tone_onset_rel_go_s`, `tongue_q40/q60`, `photostim_trials`, …) are computed and written to `conversion_summary.json`, which the decoder never reads.
- The script always builds and pickles a second, sample dataset (`/app/sample_data.pkl`, 3 sessions × 32 trials) in addition to the required output.
- `metadata` carries the full 81-element `bin_edges_s` and 80-element `bin_centers_s` lists, which are redundant with `time_bin_size`/`off_start`/`off_end`.

ii.
```python
            photostim_power = _parse_optional_float_array(trials["photostim_power"])
```
```python
                    "mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
                    "mean_tone_onset_rel_go_s": float(np.mean(kept_tone_onsets - go_starts[trial_keep])),
```
```python
    sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
    with open(args.sample_out, "wb") as f:
        pickle.dump(sample_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The sample dataset and summary json are intentional validation artifacts, listed in `README.md` and used throughout the trajectory to check the format and train a quick decoder before committing to the full run (steps 97–103, 155–160). The remaining items (`photostim_power`, full-column `anno_name` decode, redundant metadata) are unused leftovers with negligible cost.
