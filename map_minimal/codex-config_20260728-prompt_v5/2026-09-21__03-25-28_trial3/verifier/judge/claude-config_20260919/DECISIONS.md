# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is read directly from the published NWB distribution in `/app/data`, which stores one NWB file per session under `sub-<subject_id>/`. The AI enumerates every file with a single sorted `Path.glob("sub-*/*.nwb")`, and opens each one once with `pynwb.NWBHDF5IO` inside a `with` block. Everything else (subject, trials, units, behavioural events, video tracking) is pulled out of that single open file: `nwb.subject.subject_id`, `nwb.trials.to_dataframe()`, `nwb.units.to_dataframe()`, `nwb.acquisition['BehavioralEvents']`, `nwb.acquisition['BehavioralTimeSeries']`. All 174 files are visited; sessions are appended to the output in sorted-path order. One session is dropped (no `classification == 'good'` units), giving 173 sessions / 89,544 trials.

ii.
```python
NWB_GLOB = "sub-*/*.nwb"
...
    session_paths = sorted(Path(data_dir).glob(NWB_GLOB))
    for session_idx, path in enumerate(session_paths, start=1):
        print(f"[{session_idx}/{len(session_paths)}] Converting {path.name}", flush=True)
        session_data = convert_session(path, brain_region_to_idx)
```

```python
def convert_session(path, brain_region_to_idx):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        trials = nwb.trials.to_dataframe()

        units_df = nwb.units.to_dataframe()
```

iii. The AI first inventoried the repository and the NWB schema (`rg --files /app`, then a probe printing `nwb.acquisition`, `nwb.units.colnames`, `nwb.trials.colnames`) and concluded at step 19 that "the NWB schema is usable directly: trials already expose `photostim`, `early_lick`, `outcome`, and instruction fields, and units carry `anno_name`". It then ran a full-dataset pass over all 174 files (step 36/61) to confirm every session has tongue tracking and to count good units per session before committing to the loader. One file per session means the glob is the complete session list, so no other index is needed.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the NWB file's own `nwb.subject.subject_id` field (a numeric string such as `'440956'`), not from the directory name. During assembly the AI maintains a `subject_to_idx` dict, appending each newly-seen subject to `subjects` in order of first appearance (which, because the file list is sorted, is ascending subject id). `subject_idx` records one index per session, in the same order as `neural`/`input`/`output`. The subject id is also duplicated into `metadata['session_info']`.

ii.
```python
        return {
            "subject": str(nwb.subject.subject_id),
            ...
        }
```

```python
        subject = session_data["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
        ...
        data["subject_idx"].append(subject_to_idx[subject])
    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```

iii. The AI's exploration (step 18) printed `nwb.subject.subject_id` as part of its first schema probe and used it from then on without further comment — it is the canonical animal identifier inside the file, and the `sub-<id>` folder name is derived from it, so no separate grouping step was needed. The verifier reported 28 subjects, matching the dandiset.

## 1-c. How are the data split into sessions?

i. No splitting is performed: one NWB file is treated as one session. Session order in the output is the sorted file order (which, because the filename embeds the acquisition timestamp `ses-YYYYMMDDThhmmss`, is chronological within each subject). Each session is identified in metadata by `path.stem` (the filename without extension) plus the full source path, and per-session bookkeeping (trial counts at each filtering stage, good-unit count, tongue percentiles) is recorded in `metadata['session_info']`.

ii.
```python
        session_info = {
            "session_id": path.stem,
            "source_file": str(path),
            "subject": str(nwb.subject.subject_id),
            "n_trials_total": int(len(trials)),
            "n_trials_recorded": int(recorded_trials.sum()),
            "n_trials_kept": int(len(trials_kept)),
            "n_trials_nonzero_neural": int(nonzero_trial_mask.sum()),
            "n_good_units": int(good_unit_mask.sum()),
            "tongue_visible_p40": p40,
            "tongue_visible_p60": p60,
        }
```

```python
        data["metadata"]["session_info"].append(session_data["session_info"])
```

iii. The AI treated the file boundary as the session boundary implicitly from its first probe. At step 78 it explicitly tied the count back to the literature: "the one NWB session with zero published-good units explains why the paper reports 173 analyzable sessions out of 174 files", and at step 90 confirmed the export contains 173 sessions. It did not use `nwb.identifier` (e.g. `SC015_20190207_120657_s1`, which carries the paper's mouse name) as the session id, preferring the filename stem.

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table (`nwb.trials.to_dataframe()`), one row per behavioural trial, with `start_time`/`stop_time` defining the trial extent. The go cue for trial *i* is element *i* of the `go_start_times` event stream, i.e. the AI assumes a strict one-to-one, ordered correspondence between trials-table rows and go-cue events, and indexes the go-cue array with the trial keep-mask. Other trial-phase events (`delay_start_times`) are *not* assumed to be one-per-trial and are instead looked up by time within `[start_time, stop_time]`.

ii.
```python
        go_times = np.asarray(
            nwb.acquisition["BehavioralEvents"].time_series["go_start_times"].timestamps[:],
            dtype=np.float64,
        )
        go_times_kept = go_times[keep_trials]
```

```python
def last_event_per_trial(event_times, trial_starts, trial_stops):
    left = np.searchsorted(event_times, trial_starts, side="left")
    right = np.searchsorted(event_times, trial_stops, side="right")
    out = np.full(trial_starts.shape, np.nan, dtype=np.float64)
    valid = right > left
    out[valid] = event_times[right[valid] - 1]
    return out
```

iii. The AI verified the multiplicity of the trial-phase events empirically before writing the converter (step 32): it counted `sample_start_times` and `delay_start_times` falling inside each trial and found counts of 1–12 per trial, concluding at step 34 that "tone onset is not a single fixed offset on early-lick trials because the task replays the sample epoch". Go cues, by contrast, were treated as exactly one per trial; the code contains no assertion of this, but a length mismatch would raise on the boolean indexing `go_times[keep_trials]`.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied in sequence:

1. **Ephys coverage.** A trial is kept only if some interval in the first good unit's `obs_intervals` matches *both* its `start_time` and its `stop_time` to within 1e-4 s. This was added after the AI discovered whole blocks of all-zero trials in its first export.
2. **`auto_water == 0`** — automatic-water trials excluded.
3. **`free_water == 0`** — free-water trials excluded.
4. **Non-empty spikes.** After binning, any trial whose firing-rate matrix is identically zero across all good units is dropped.

A session is dropped if fewer than 2 trials survive step 3, or fewer than 2 survive step 4. Early-lick and `ignore` (no-response) trials are deliberately **kept**, contrary to the paper's own "regular trials" definition, because they are required decoder outputs. No behavioural-performance filter is applied. Result: 89,544 trials over 173 sessions, with 0 all-zero trials remaining.

ii.
```python
def recorded_trial_mask_from_obs_intervals(trials, obs_intervals, atol=1e-4):
    obs_intervals = np.asarray(obs_intervals, dtype=np.float64)
    if obs_intervals.ndim != 2 or obs_intervals.shape[1] != 2:
        return np.ones(len(trials), dtype=bool)

    trial_starts = trials["start_time"].to_numpy(dtype=np.float64)
    trial_stops = trials["stop_time"].to_numpy(dtype=np.float64)
    keep = np.zeros(len(trials), dtype=bool)
    for obs_start, obs_stop in obs_intervals:
        keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)
    return keep
```

```python
        first_good_unit = units_df.loc[good_unit_mask].iloc[0]
        recorded_trials = recorded_trial_mask_from_obs_intervals(trials, first_good_unit["obs_intervals"])
        keep_trials = (
            recorded_trials
            & (trials["auto_water"].to_numpy() == 0)
            & (trials["free_water"].to_numpy() == 0)
        )
        trials_kept = trials.loc[keep_trials].copy()
        if len(trials_kept) < 2:
            return None
```

```python
        neural_trials = build_neural_trials(units_df, good_unit_mask, go_times_kept)
        nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
        if nonzero_trial_mask.sum() < 2:
            return None
```

iii. Three distinct justifications appear in the trajectory.

- On keeping early-lick/ignore trials (step 29): "The paper's analysis code excludes early/no-response/auto-water/free-water when predicting neural activity from video, but this decoder explicitly asks for early lick and ignore/no-lick outputs, so I'm treating those as a task-driven exception." The `auto_water`/`free_water` half of that same rule was kept, so the AI adopted the paper's curation everywhere it did not conflict with the decoder spec.
- On `obs_intervals` (step 100, after the verifier flagged all-zero trials): "Some NWB files contain the full behavior session, but the units' `obs_intervals` only cover the recorded subset of trials; I was keeping all behavior trials instead of intersecting with the actual ephys trial intervals." It confirmed the fix removed "essentially all of the bogus empty trials".
- On the residual zero trials (step 110): "only two session-terminal trials remain fully zero. Those are almost certainly residual boundary artifacts, so I'm trimming them explicitly rather than carrying verifier warnings into the final dataset."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute spike times, read via `units_df["spike_times"]`), restricted to units with `classification == 'good'`, together with `BehavioralEvents/go_start_times`, which supplies the per-trial alignment time used to place the bin edges. `units/anno_name` supplies the per-neuron brain-region label, and `units/obs_intervals` is used for trial curation only.

ii.
```python
def build_neural_trials(units_df, keep_mask, go_times_kept):
    good_units = units_df.loc[keep_mask]
    ...
    for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
        spikes = np.asarray(spike_times, dtype=np.float64)
```

```python
        units_df = nwb.units.to_dataframe()
        good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
```

iii. Spike times are the only neural representation in the NWB files, so no choice was involved; the AI's probe at step 22/28 confirmed the `units` columns (`classification`, `anno_name`, `avg_firing_rate`, `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `is_good_trials`) and it selected `spike_times` + `classification` + `anno_name` as the ones it needed.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to firing rate in Hz by dividing by the bin width, stored as **float16**. For each good unit, the 81 absolute bin edges for every trial are built as one `(n_trials, 81)` array, flattened, and searched once with `np.searchsorted(..., side="left")`; differencing adjacent edge positions gives the per-bin spike count. No smoothing, no normalisation, no baseline subtraction, no trial-averaging. Output per trial is a `(n_neurons, 80)` array.

ii.
```python
    neural = np.empty((ntrials, nneurons, ntime), dtype=np.float16)
    abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]

    for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
        spikes = np.asarray(spike_times, dtype=np.float64)
        edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
        counts = np.diff(edge_idx, axis=1)
        neural[:, unit_idx, :] = (counts / BIN_SIZE_S).astype(np.float16, copy=False)

    return [neural[trial_idx] for trial_idx in range(ntrials)]
```

iii. The AI located the reference implementation's binning helper (step 63: `rg -n "def sliding_histogram"` in `/app/code/VideoAnalysisUtils/`) and read `preprocessing_DJ_2022Aug.py` before writing its own, then summarised its plan (step 69) as "go-aligned 50 ms firing-rate bins over `[-2.5, 1.5)`". Its closing report states "firing rates from spike counts / 0.05". The float16 choice is not discussed in the trajectory; it halves the payload (5.5 GB vs the reference's 11.9 GB) and is lossless here because every stored value is a multiple of 20 Hz well inside float16's exactly-representable range.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single criterion: `units/classification == 'good'`, the verdict of the published spike-sorting QC classifier. No thresholds are applied to any individual metric (`presence_ratio`, `amplitude_cutoff`, `isi_violation`), and the older `unit_quality` label is not used. The per-unit/per-trial `is_good_trials` mask was examined and deliberately **not** applied. A session with zero good units returns `None` and is dropped. Result: 173 sessions, mean 401 neurons/session (min 90, max 923), ~69.5k neurons total.

ii.
```python
        units_df = nwb.units.to_dataframe()
        good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
        if good_unit_mask.sum() == 0:
            return None
```

iii. The AI investigated the alternative QC signals directly. It counted, over the first 20 sessions, how many good units carry any `False` in `is_good_trials` (steps 51/52/59/65/66) and found a single session with 8 of 169 units affected (mean good-trial fraction 0.9928). It concluded at step 62: "the published session-level unit QC is the `classification == 'good'` label, and the newer per-unit `is_good_trials` mask is only occasionally nontrivial and doesn't appear in the reference analysis code. I'm using that as metadata sanity-checking, not as a hard exclusion rule." It cross-checked the resulting session count against the literature at step 78: the one zero-good-unit file "explains why the paper reports 173 analyzable sessions out of 174 files".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to go-cue onset, with no resampling or offset correction: spike times, event timestamps and video timestamps all live on one session-absolute clock, so alignment is just an additive shift. The fixed relative edge grid `BIN_EDGES_REL_S` (−2.5 … +1.5 s) is broadcast onto each trial's go-cue time to produce absolute edges, and the spikes are binned against those.

ii.
```python
BIN_EDGES_REL_S, BIN_CENTERS_REL_S = make_time_grid()
...
    abs_edges = go_times_kept[:, None] + BIN_EDGES_REL_S[None, :]
```

```python
            "temporal_alignment_event": "Go cue onset",
            "off_start": WINDOW_START_S,
            "off_end": WINDOW_END_S,
```

iii. The instructions specify go-cue alignment; the AI restated it in its plan (step 67: "go-cue alignment, 50 ms bins") and noted from its reading of the reference code (step 14) that "their raw ephys arrays are go-cue aligned", i.e. the requested alignment already matches the reference pipeline. It verified alignment sanity on a photostim session (step 73) by checking that the photostim bins land in the expected late-delay positions relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins (`time_bin_size: 50.0` ms in metadata), 80 non-overlapping bins spanning −2.5 s to +1.5 s around the go cue, identical for every trial and every session. The grid is built once at module import as 81 edges and 80 centres. No rebinning, no overlapping/sliding windows, no downsampling of the neural stream; the two other streams that have a native rate (video at ~294 Hz, event times) are brought *onto* this grid rather than the grid being adapted to them.

ii.
```python
BIN_SIZE_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5


def make_time_grid():
    bin_edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S / 2, BIN_SIZE_S)
    bin_centers = bin_edges[:-1] + BIN_SIZE_S / 2
    return bin_edges, bin_centers
```

```python
            "time_bin_size": 50.0,
            "time_bin_centers_s": BIN_CENTERS_REL_S.tolist(),
```

iii. Window and bin width are dictated by the Decoder Task section of the instructions; the AI restated them in its plan and closing report. Building the grid once as module-level constants guarantees the format requirement that "time bins should be the same size for all trials and sessions" — the verifier confirmed `T_min = T_max = 80`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **Not** from the tone event stream. The AI derives tone onset from `BehavioralEvents/delay_start_times`: it takes the *last* delay-epoch onset falling inside the trial's `[start_time, stop_time]` and subtracts a hard-coded 0.65 s sample-epoch duration. If a trial contains no delay event at all, it falls back to a hard-coded `go_cue − 1.85 s`. `sample_start_times` — which is the actual record of tone onsets — is loaded nowhere in the final script.

ii.
```python
        delay_times = np.asarray(
            nwb.acquisition["BehavioralEvents"].time_series["delay_start_times"].timestamps[:],
            dtype=np.float64,
        )
        delay_last = last_event_per_trial(
            delay_times,
            trials_kept["start_time"].to_numpy(dtype=np.float64),
            trials_kept["stop_time"].to_numpy(dtype=np.float64),
        )
        tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
```

iii. The AI compared both candidate derivations. At step 48 it noted "on cleanly mapped trials the final sample-epoch onset sits exactly 1.85 s before the go cue, including early-lick replays where the last replayed sample is the valid one. I'm auditing the few mismatches now before I hard-code anything, because if they reflect an NWB bookkeeping quirk I'd rather derive tone onset from events than assume a constant." Its audit (steps 49/56) found that in the first session, 9 of 368 trials disagree: the last `sample_start_times` entry sits 1.99–4.25 s before the go cue while `delay_last − 0.65` gives exactly 1.85 s. It then verified (step 55) that the delay-based rule yields a single value, `−1.85`, for all 368 trials of that session, and adopted it, describing the choice at step 53 as "derive tone onset from the final valid pre-go task structure". The mismatching trials are those where only the *delay* epoch was replayed after an early lick (their event sequence is one sample onset followed by several delay onsets), so no tone was actually played 1.85 s before the go cue on those trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The absolute time of each bin centre is computed as `go_cue + bin_centre`, and the tone onset (3-a) is subtracted, giving a continuous, time-varying, per-trial ramp in seconds. Stored as row 0 of the `(2, 80)` float32 input array. No clipping, no normalisation, and negative values (bins before the tone) are retained. Because the delay-based rule collapses to `go − 1.85 s` on essentially every trial of the sessions checked, the resulting ramp is close to constant across trials; the dataset-wide range is −1.525 s to 3.925 s.

ii.
```python
        tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
        time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

```python
            input_trial = np.vstack([time_from_tone[i], photostim[i]]).astype(np.float32, copy=False)
```

iii. The instructions call for a continuous, time-varying "time from tone onset in seconds", so no discretisation is applied. The AI's own summary describes the input as "continuous time-from-tone". The two constants (0.65 s sample duration, 1.85 s tone-to-go interval) were read off its empirical probes of the first eight sessions (step 58: `go − last sample onset` equals 1.85 s on 359/368, 471/480, 515/518 … trials), not from the methods text.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the same `BIN_CENTERS_REL_S` grid and the same `go_times_kept` array that define the neural bin edges are used to build the input, so bin *k* of the input covers exactly the interval of bin *k* of the firing rates. Both arrays are also filtered by the same trial masks (`keep_trials`, then `nonzero_trial_mask`) in the same order.

ii.
```python
        time_from_tone = (go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]) - tone_onset[:, None]
```

```python
        trials_kept = trials_kept.iloc[nonzero_trial_mask].copy()
        go_times_kept = go_times_kept[nonzero_trial_mask]
        time_from_tone = time_from_tone[nonzero_trial_mask]
        photostim = photostim[nonzero_trial_mask]
        tongue_state = tongue_state[nonzero_trial_mask]
        neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask) if keep]
```

iii. No separate justification appears; all NWB streams share one global clock, so laying the single go-cue-relative grid over every stream is sufficient. The AI checked the result by hand on a photostim session (step 73), printing "example time from tone first/last" alongside the photostim bin indices.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The `photostim_onset` and `photostim_duration` columns of the trials table, which are stored as **strings**, with the literal `'N/A'` on non-stimulated trials, and which are measured relative to the trial's `start_time`. `trials['start_time']` and the trial's go-cue time are used to re-express them on the go-cue-relative axis. The `photostim_start_times`/`photostim_stop_times` event streams are not used.

ii.
```python
    for i, (_, trial) in enumerate(trials_kept.iterrows()):
        onset = trial["photostim_onset"]
        duration = trial["photostim_duration"]
        if onset == "N/A" or duration == "N/A":
            continue
        onset_abs = float(trial["start_time"]) + float(onset)
        offset_abs = onset_abs + float(duration)
```

iii. The AI inspected the photostim columns on a session known to contain stimulation (step 54, `sub-480133_ses-20201225T114244`), printing `photostim_onset`, `photostim_duration`, `photostim_power` alongside `start_time`/`stop_time`, and computing each stimulated trial's onset relative to the go cue before writing the converter. That probe established both the string encoding with `'N/A'` sentinels and the trial-start-relative reference frame.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0.0/1.0 float32) time series, one value per bin: a bin is 1 if its centre falls in `[onset, offset)` expressed relative to the go cue, else 0. Non-stimulated trials are left as an all-zero row (the array is pre-allocated with `np.zeros` and the loop `continue`s). Stored as row 1 of the input array. This is a per-time-point flag, as the instructions require, not a per-trial flag.

ii.
```python
def build_photostim_input(trials_kept, go_times_kept):
    photostim = np.zeros((len(trials_kept), len(BIN_CENTERS_REL_S)), dtype=np.float32)
    for i, (_, trial) in enumerate(trials_kept.iterrows()):
        onset = trial["photostim_onset"]
        duration = trial["photostim_duration"]
        if onset == "N/A" or duration == "N/A":
            continue
        onset_abs = float(trial["start_time"]) + float(onset)
        offset_abs = onset_abs + float(duration)
        onset_rel = onset_abs - go_times_kept[i]
        offset_rel = offset_abs - go_times_kept[i]
        photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
    return photostim
```

iii. Stated in the plan at step 67 ("binary photostim-on") and in the closing report ("Inputs: continuous time-from-tone and binary photostim-on"), following the instruction that time-varying inputs should be represented as binary time series. The AI validated it at step 73 by picking a session with stimulation, counting the stimulated trials and printing "example photostim bins on" — confirming the on-bins land where expected in the delay epoch. The verifier confirms the final range is exactly `[0.0, 1.0]`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are converted from trial-start-relative to go-cue-relative by adding `start_time` and subtracting the trial's go-cue time, then compared against the same `BIN_CENTERS_REL_S` grid used for the neural bins. Trial ordering is kept in lockstep with the neural data by the shared masks.

ii.
```python
        onset_rel = onset_abs - go_times_kept[i]
        offset_rel = offset_abs - go_times_kept[i]
        photostim[i] = ((BIN_CENTERS_REL_S >= onset_rel) & (BIN_CENTERS_REL_S < offset_rel)).astype(np.float32)
```

iii. No separate rationale is recorded beyond the step-54 probe establishing that photostim times are stored relative to `start_time` rather than to the go cue, which made the re-expression necessary.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB files, so choice is reconstructed from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side; a miss means it licked the opposite side; an `ignore` means it did not lick, which becomes a third class.

ii.
```python
def choice_from_instruction_and_outcome(instruction, outcome):
    if outcome == "ignore":
        return 2
    if instruction == "left":
        return 0 if outcome == "hit" else 1
    if instruction == "right":
        return 1 if outcome == "hit" else 0
    raise ValueError(f"Unexpected trial instruction: {instruction!r}")
```

```python
            choice = choice_from_instruction_and_outcome(
                str(trial["trial_instruction"]),
                str(trial["outcome"]),
            )
```

iii. The AI enumerated the trials-table columns and their value counts at step 20 (`outcome`, `trial_instruction`, `early_lick`, `auto_water`, …), which established that instruction × outcome fully determines the licked side and that only three outcome strings occur. It raises on any unexpected instruction string rather than silently defaulting. It did not attempt to read the lick event streams to confirm that `ignore` really means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, with `output_values[0] = ['left', 'right', 'no lick']`. The per-trial scalar is broadcast across all 80 bins with `np.full` so that all four outputs can share one `(4, 80)` int8 array per trial.

ii.
```python
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_position"],
        "output_values": [
            ["left", "right", "no lick"],
            ...
        ],
```

```python
            output_trial = np.vstack(
                [
                    np.full(len(BIN_CENTERS_REL_S), choice, dtype=np.int8),
                    np.full(len(BIN_CENTERS_REL_S), outcome, dtype=np.int8),
                    np.full(len(BIN_CENTERS_REL_S), early, dtype=np.int8),
                    tongue_state[i],
                ]
            )
```

iii. The AI read the decoder implementation before deciding on the representation (steps 30/31/43/44/45), reasoning at step 43: "I'm checking the decoder implementation itself because it constrains how much of each label should be time-varying. If a label is per-trial in the task spec, the cleanest match is usually to broadcast it across the aligned bins, but I want to confirm there isn't a better-supported representation hidden in the trainer." Its plan (step 67) then fixed "broadcasting per-trial `choice`, `outcome`, and `early_lick` across bins". The resulting class balance is 42.9% left / 42.2% right / 14.8% no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` required by the instructions. No derivation from reward/lick streams.

ii.
```python
            outcome = outcome_to_int(str(trial["outcome"]))
```

iii. Established by the step-20 probe, which printed `trials['outcome'].value_counts(dropna=False)` and showed only those three values. Since the trials table is authoritative and already matches the requested categories, no reconstruction was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2`, matching the order given in the instructions; anything else raises. The scalar is broadcast over all 80 bins as row 1 of the output array, and `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
def outcome_to_int(outcome):
    mapping = {"ignore": 0, "miss": 1, "hit": 2}
    if outcome not in mapping:
        raise ValueError(f"Unexpected outcome: {outcome!r}")
    return mapping[outcome]
```

iii. The code assignment follows the ordering in the instructions ("Outcome (ignore, miss, hit)"). The strict `ValueError` is the AI's chosen guard against silently mis-coding an unexpected string. Final balance: 14.8% ignore / 16.7% miss / 68.5% hit.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds the strings `'no early'` and `'early'`. The lick events themselves are not consulted.

ii.
```python
            early = early_lick_to_int(str(trial["early_lick"]))
```

iii. The step-20 probe printed `trials['early_lick'].value_counts()` and confirmed the two-valued string encoding, so the flag is taken at face value. The AI separately established (steps 32/34) that early licks trigger sample/delay epoch replays *before* the go cue, i.e. the physical event that sets the flag falls inside the −2.5 s analysis window even though the label is stored per trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' → 0`, `'early' → 1`; anything else raises. Broadcast across all 80 bins as row 2 of the output array; `output_values[2] = ['no', 'yes']`.

ii.
```python
def early_lick_to_int(value):
    mapping = {"no early": 0, "early": 1}
    if value not in mapping:
        raise ValueError(f"Unexpected early_lick value: {value!r}")
    return mapping[value]
```

iii. The coding follows the instructions ("Early lick (no, yes)"). Like the other per-trial labels it is broadcast rather than left as a scalar, per the instruction "If at all possible, make it time-varying" and the AI's step-43 reading of the trainer. Final balance: 88.4% no / 11.6% yes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` sampled at ~294 Hz with its own `timestamps`. Column 1 is the y-position; column 2 is the DeepLabCut tracking likelihood, used as the visibility indicator.

ii.
```python
def build_tongue_states(tongue_ts, go_times_kept):
    tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
    timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
    y_all = tracking[:, 1]
    likelihood_all = tracking[:, 2]
```

```python
        tongue_ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
        tongue_state, p40, p60 = build_tongue_states(tongue_ts, go_times_kept)
```

iii. The AI searched the reference code for tongue/DeepLabCut handling (step 25) and probed the series itself (steps 23/27/33/57/60): it printed the per-column min/max/quantiles, the median inter-frame interval (0.0034 s), and the likelihood distribution, finding it strongly bimodal ("likelihood quantiles [5.2e-05 1.22e-04 1.0 1.0 1.0]", with only ~6–15% of frames above 0.9). It also confirmed in a full-dataset pass (step 36/61) that tracking is present in all 174 sessions. That bimodality is what it used to justify treating the likelihood channel as a visibility flag: step 53, "treat tongue visibility from the DeepLabCut confidence channel rather than raw coordinates".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps, all point-sampled rather than averaged:

1. Session-level class edges: take every frame in the session with `likelihood >= 0.9`, and compute the 40th and 60th percentiles of *those raw frames'* y-values. If no frame is visible, the edges are NaN and every bin becomes class 3.
2. Per bin, find the last camera frame at or before the bin centre (`searchsorted(..., side="right") - 1`); that single frame supplies both the y-value and the likelihood for the bin. No averaging over the ~15 frames that fall inside each 50 ms bin.
3. Discretise that frame's y against the session edges; if its likelihood is below threshold, or there is no preceding frame, the bin is class 3.

The per-session `p40`/`p60` are recorded in `metadata['session_info']`.

ii.
```python
TONGUE_VISIBILITY_THRESHOLD = 0.9
...
    visible_all = likelihood_all >= TONGUE_VISIBILITY_THRESHOLD
    visible_y = y_all[visible_all]
    if visible_y.size == 0:
        p40 = np.nan
        p60 = np.nan
    else:
        p40, p60 = np.percentile(visible_y, [40, 60])
```

```python
    abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
    frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
    frame_idx = frame_idx.reshape(abs_centers.shape)
```

iii. The percentile scope (per-session, 40/60) is set by the instructions. The visibility threshold was chosen from the empirical likelihood distribution the AI measured at steps 57/60 — the channel is effectively binary (medians ~1e-4, 99th percentile 1.0), so 0.9 and 0.5 select nearly identical frame sets. Restricting the percentiles to visible frames is justified by the fact that the tracker still emits a y-coordinate when the tongue is retracted, so including those frames would corrupt the edges. The AI's closing report states: "time-varying tongue state from side-camera tongue tracking using DLC confidence `>= 0.9` for visibility and per-session visible-frame `y` percentiles". Nothing in the trajectory discusses averaging within a bin versus sampling one frame per bin; sampling is at least self-consistent with taking the percentiles over raw frames. Measured on session 1, this labels 12.6% of bins visible, against 20.6% for a bin-mean scheme; dataset-wide 84.1% of bins are class 3.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified: `0` = y below the 40th percentile, `1` = y in `[p40, p60]` inclusive, `2` = y above the 60th percentile, `3` = not visible (likelihood below threshold, no preceding frame, or no visible frame anywhere in the session). The array is pre-filled with `3` so the "not visible" class is the default, and the three visible classes are written only where `visible` is true. `output_values[3] = ['<40th percentile', '40th to 60th percentile', '>60th percentile', 'not visible']`.

ii.
```python
    tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
    valid = (frame_idx >= 0) & (frame_idx < len(timestamps))
    if np.any(valid):
        y = y_all[frame_idx[valid]]
        likelihood = likelihood_all[frame_idx[valid]]
        visible = likelihood >= TONGUE_VISIBILITY_THRESHOLD
        state = np.full(y.shape, 3, dtype=np.int8)
        if visible_y.size > 0:
            state[visible & (y < p40)] = 0
            state[visible & (y >= p40) & (y <= p60)] = 1
            state[visible & (y > p60)] = 2
        tongue_state[valid] = state
```

iii. The three-way percentile split and the fourth "not visible" category are given verbatim in the Decoder Task section of the instructions; the AI implemented them literally, using inclusive bounds on both sides of the middle class. Resulting balance: 6.2% / 3.2% / 6.5% / 84.1%, i.e. the visible bins split ~39/20/41, as the 40/60 percentile definition predicts.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with spikes and events, so alignment is a direct lookup: the absolute time of each go-cue-relative bin centre is computed, and `np.searchsorted` on the camera timestamps returns the index of the last frame at or before that instant. No interpolation and no per-stream offset. Bins whose centre precedes the first camera frame get index −1 and fall into the "not visible" class. There is no maximum-gap check, so a bin whose centre falls in a period when the camera was off is labelled from the most recent earlier frame, however old — empirically the windows are fully covered in the session I checked (0% of bin centres more than one frame-interval away from their source frame). Trial ordering is kept in lockstep with the neural data by the shared `nonzero_trial_mask`.

ii.
```python
    abs_centers = go_times_kept[:, None] + BIN_CENTERS_REL_S[None, :]
    frame_idx = np.searchsorted(timestamps, abs_centers.ravel(), side="right") - 1
    frame_idx = frame_idx.reshape(abs_centers.shape)

    tongue_state = np.full(abs_centers.shape, 3, dtype=np.int8)
    valid = (frame_idx >= 0) & (frame_idx < len(timestamps))
```

```python
        tongue_state = tongue_state[nonzero_trial_mask]
```

iii. The AI confirmed at step 27 that the series carries explicit `timestamps` (median Δt 0.0034 s) rather than a start-time/rate pair, which is what makes the plain `searchsorted` lookup valid. Beyond that it offers no separate alignment rationale — the single go-cue-relative grid is applied to the video exactly as it is to the spikes, which guarantees bin-for-bin correspondence.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, handled in three different ways — drop, substitute a sentinel category, or fall back to a task constant:

- **Session never quality-controlled** (`classification` is NaN for every unit): `astype(str)` turns NaN into `'nan'`, no unit matches `'good'`, and `convert_session` returns `None`, so the session is dropped. This is the single file that takes 174 → 173.
- **Trials with no ephys coverage**: dropped via `obs_intervals` matching (1-e).
- **Trials that are all-zero after binning**: dropped explicitly.
- **Sessions left with fewer than 2 usable trials**: dropped (the format requires ≥2 trials per session).
- **Missing/blank anatomical annotation**: the region label becomes the string `"Unknown"` rather than being dropped, so the neuron is still included.
- **Missing tongue tracking**: frames below the likelihood threshold, bins before the first frame, and sessions with no visible frame at all become the explicit `3 = not visible` class; the percentile edges degrade to `None` in `session_info` rather than raising.
- **Missing delay event on a trial**: `tone_onset` falls back to the hard-coded `go − 1.85 s`.

Conversely, anything *unexpected* rather than missing — an unknown `outcome`, `early_lick`, or `trial_instruction` string — raises a `ValueError` instead of being coerced.

ii.
```python
        good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
        if good_unit_mask.sum() == 0:
            return None
```

```python
        anno_names = units_df.loc[good_unit_mask, "anno_name"].astype(str).str.strip().replace("", "Unknown")
```

```python
    if visible_y.size == 0:
        p40 = np.nan
        p60 = np.nan
```

```python
        tone_onset = np.where(np.isnan(delay_last), go_times_kept - 1.85, delay_last - 0.65)
```

```python
    return tongue_state, float(p40) if visible_y.size else None, float(p60) if visible_y.size else None
```

iii. The overall pattern is: exclude where the measurement is genuinely absent (no spikes, no QC), and represent as an explicit category where the measurement legitimately has no value (tongue retracted). The AI diagnosed the largest missing-data case itself from the verifier's warnings rather than from the file's documentation — step 95: "a nontrivial set of late trials are completely all-zero across every 'good' unit, which strongly suggests trials outside the ephys coverage are still present rather than a formatting bug" — and it kept a per-session audit trail (`n_trials_total`, `n_trials_recorded`, `n_trials_kept`, `n_trials_nonzero_neural`) in `session_info` so each drop is traceable. It verified the end state at step 117: "sessions with zero trials 0, total zero trials 0, total sessions/trials 173 89544".

## 10-a. What are the most time-consuming steps of the code?

i. Three costs dominate, all I/O- or memory-bound:

1. **`nwb.units.to_dataframe()`**, called once per session. This materialises *every* unit's ragged columns — `spike_times`, `obs_intervals`, `is_good_trials` — into memory, including the ~75% of units that fail QC and are never used. It is the single largest avoidable cost in the script.
2. **`tongue_ts.data[:]`**, which reads the whole `(n_frames, 3)` tracking array (~680k × 3 per session) and casts it to float32.
3. **The per-unit `searchsorted` loop** in `build_neural_trials`: one binary search per bin edge per good unit (81 × n_trials searches per unit), plus the final `.astype(np.float16)` per unit.

Downstream, pickling the 5.5 GB result is a further one-off cost. The full conversion of 174 sessions ran in roughly 4–5 minutes of wall time in the agent's environment, comparable to the reference implementation.

ii.
```python
        units_df = nwb.units.to_dataframe()
```

```python
    tracking = np.asarray(tongue_ts.data[:], dtype=np.float32)
    timestamps = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
```

```python
    for unit_idx, spike_times in enumerate(good_units["spike_times"].to_list()):
        spikes = np.asarray(spike_times, dtype=np.float64)
        edge_idx = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(ntrials, -1)
```

iii. The AI never profiled the script and the trajectory contains no discussion of runtime cost; its only performance-related observations are throughput checks while watching the conversion scroll past ("The throughput stayed stable through the largest 2020–2021 block, so I'm not changing the converter", step 88). The choice of `to_dataframe()` is a convenience/readability choice — it makes `units_df.loc[good_unit_mask, "anno_name"]` and `good_units["spike_times"]` natural — paid for with extra I/O, and it was fast enough that the AI never revisited it.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain, three of which are avoidable:

1. `recorded_trial_mask_from_obs_intervals` loops over every observation interval and ORs a full-length `np.isclose` comparison each time — an O(n_obs × n_trials) scan where a single `np.isin` on rounded start times (plus one on stop times) would be O(n log n).
2. `build_photostim_input` loops over trials with `trials_kept.iterrows()`, the slowest pandas iteration idiom, to fill one row at a time. The onsets/offsets could be converted column-wise and the whole mask built with one broadcast comparison against `BIN_CENTERS_REL_S`.
3. The final assembly loop again uses `iterrows()` over all trials to `np.vstack` a 2-row input and a 4-row output per trial, and to map three label strings through dictionaries one trial at a time; all three mappings are vectorisable.
4. `nonzero_trial_mask` is a Python list comprehension calling `np.any` per trial; `neural.any(axis=(1, 2))` on the stacked array does the same in one call.

The per-unit loop in `build_neural_trials` is *not* avoidable — each unit has a different number of spikes, so there is no single sorted array to search — and the per-trial dimension inside it is already vectorised by flattening the edge array.

ii.
```python
    keep = np.zeros(len(trials), dtype=bool)
    for obs_start, obs_stop in obs_intervals:
        keep |= np.isclose(trial_starts, obs_start, atol=atol) & np.isclose(trial_stops, obs_stop, atol=atol)
```

```python
    for i, (_, trial) in enumerate(trials_kept.iterrows()):
        onset = trial["photostim_onset"]
```

```python
        for i, (_, trial) in enumerate(trials_kept.iterrows()):
            input_trial = np.vstack([time_from_tone[i], photostim[i]]).astype(np.float32, copy=False)
            choice = choice_from_instruction_and_outcome(...)
```

```python
        nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. No rationale is recorded; the AI wrote the converter in a single pass (step 70) with per-trial loops as the straightforward expression of the logic, and never returned to optimise them because total runtime was acceptable. The scalar helper functions (`choice_from_instruction_and_outcome`, `outcome_to_int`, `early_lick_to_int`) that force the per-trial loop do buy something in exchange: they raise on unexpected label strings, which a vectorised dictionary lookup would have to re-implement.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and each session is processed in a single pass, so nothing expensive is recomputed. What is repeated is small and per-session:

- `units_df.loc[good_unit_mask]` is evaluated three separate times (for `first_good_unit`, for `anno_names`, and again inside `build_neural_trials`), each producing a new sub-frame.
- `trials_kept.iterrows()` is walked twice — once in `build_photostim_input`, once in the final assembly loop — so the same rows are re-materialised as pandas Series.
- The tongue likelihood threshold comparison is applied twice: once over all frames (`visible_all`, for the percentiles) and again over the sampled frames (`visible`, for the classes).
- `str()`/`astype(str)` conversions are re-applied per trial for the three label columns.

Notably, no second pass over the data is needed to establish the tongue percentile edges, because they are per-session and are computed inside the same session pass.

ii.
```python
        first_good_unit = units_df.loc[good_unit_mask].iloc[0]
        ...
        anno_names = units_df.loc[good_unit_mask, "anno_name"].astype(str)...
        ...
        neural_trials = build_neural_trials(units_df, good_unit_mask, go_times_kept)
```

```python
def build_neural_trials(units_df, keep_mask, go_times_kept):
    good_units = units_df.loc[keep_mask]
```

iii. Not discussed in the trajectory. The repeats are consequences of structuring the converter as several small helpers that each take `units_df`/`trials_kept` and re-derive their own view, which keeps the helpers independently testable — the AI did in fact exercise `convert_session` on single sessions in isolation (steps 72/73) before the full run.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work produce values that are never used, or are computed and then thrown away:

- **Non-good units are fully loaded.** `nwb.units.to_dataframe()` pulls the spike times, observation intervals and per-trial good-flags of every unit in the session; roughly three quarters of them are then discarded by `good_unit_mask`. This is by far the largest waste in the script.
- **Inputs and tongue states are computed for trials that are subsequently dropped.** `time_from_tone`, `photostim` and `tongue_state` are all built over the full `trials_kept` set, then re-indexed by `nonzero_trial_mask`; the work done for the dropped trials is discarded.
- **`delay_times` is read in full** for every session even though only the last event per trial is used.
- **`tracking` is cast to float32 in its entirety** (all three columns, whole session) when only two columns are read, and only at ~80 sampled indices per trial.
- **Metadata that the decoder ignores**: `time_bin_centers_s` (a redundant restatement of the fixed grid), `tongue_visible_p40`/`p60`, `n_trials_total`/`n_trials_recorded`/`n_trials_nonzero_neural`, `source_file`. These are cheap and have documentary value, but nothing downstream reads them.
- **Brain regions are kept at full anatomical-annotation granularity** (e.g. `"Orbital area, lateral part, layer 5"`), producing several hundred distinct region strings rather than the coarse region names the format description suggests (`e.g. ALM, V1`); the decoder does not use `brain_region_idx` at all.

ii.
```python
        units_df = nwb.units.to_dataframe()
        good_unit_mask = units_df["classification"].astype(str).to_numpy() == "good"
```

```python
        trials_kept = trials_kept.iloc[nonzero_trial_mask].copy()
        go_times_kept = go_times_kept[nonzero_trial_mask]
        time_from_tone = time_from_tone[nonzero_trial_mask]
        photostim = photostim[nonzero_trial_mask]
        tongue_state = tongue_state[nonzero_trial_mask]
```

```python
            "time_bin_centers_s": BIN_CENTERS_REL_S.tolist(),
            "tone_visibility_threshold": TONGUE_VISIBILITY_THRESHOLD,
```

iii. The AI's stated intent (step 69) was to use "detailed per-unit anatomical annotations", so the fine-grained region labels are deliberate rather than accidental. The discarded per-trial work is a direct consequence of adding the all-zero-trial filter late (step 111) as a patch applied *after* the inputs and outputs had already been computed, rather than folding it into the earlier `keep_trials` mask. The extra metadata fields are consistent with the instruction to "add other relevant fields, e.g. `session_info`".
