# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script walks the NWB release directory tree under `/app/data`, iterating over sorted subject folders and sorted `.nwb` files inside each folder. Each NWB file is treated as one candidate session and is passed to `convert_session`; the top-level `main()` loop accumulates all converted sessions into the final dataset.

ii. 
```python
def load_sorted_nwb_paths(data_root: str) -> list[str]:
    paths = []
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for filename in sorted(os.listdir(subject_path)):
            if filename.endswith(".nwb"):
                paths.append(os.path.join(subject_path, filename))
    return paths

all_paths = load_sorted_nwb_paths(args.data_root)
for idx, path in enumerate(all_paths, start=1):
    result = convert_session(path)
```

iii. The trajectory says the agent counted `174` NWB files and deliberately used sorted file order for reproducibility. `CONVERSION_NOTES.md` frames the NWB collection as the canonical source and describes a full-dataset conversion over all source files.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file, then unique subject IDs are sorted and stored in `subjects`. Each converted session gets a `subject_idx` entry pointing back to that sorted subject list.

ii. 
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))

subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["subject"]])
```

iii. There is no separate explicit justification in the notes beyond following the released dataset organization. The trajectory treats each `sub-...` directory as one mouse and reports `28` subjects from the file layout.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session IDs are taken from the filename stem, and session ordering follows the sorted path order produced by `load_sorted_nwb_paths()`.

ii. 
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

for idx, path in enumerate(all_paths, start=1):
    session_id = get_session_id_from_path(path)
    result = convert_session(path)
```

iii. The notes describe the released dataset as `174` source NWB files and the kept dataset as `173` sessions after one exclusion. The trajectory repeatedly refers to “one NWB file = one session” while reconciling the file count with the paper-level session count.

## 1-d. How are the data split into trials?

i. Trials are not taken blindly from the full behavior table. The script first reads the full behavioral trial `start_time`/`stop_time` arrays, then uses the first good unit’s `obs_intervals` and `units/is_good_trials.shape[1]` to identify only the ephys-covered trials. It matches those intervals back to behavioral trial rows by rounded `(start, stop)` pairs and uses the resulting `trial_indices` for all downstream trial-level arrays.

ii. 
```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
behavior_trial_start = np.asarray(trial_group["start_time"][:], dtype=np.float64)
behavior_trial_stop = np.asarray(trial_group["stop_time"][:], dtype=np.float64)
...
first_unit_obs = obs_intervals[obs_start:obs_stop]
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
...
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
```

iii. The trajectory records a bug investigation where later trials in some sessions had all-zero neural data because behavior tables extended past the ephys recording. The agent justified the `obs_intervals` mapping as necessary to exclude behavior-only trials and match the actual recorded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. The code keeps only ephys-covered trials by the `obs_intervals` mapping, but it does **not** apply the reference “regular trial” mask that excludes early-lick, no-response/ignore, auto-water, free-water, or photostimulation trials. It intentionally retains those trial types after session inclusion.

ii. 
```python
trial_indices = np.asarray(trial_indices, dtype=np.int64)
...
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
...
"inclusion_rules": [
    "Include NWB sessions with at least one unit whose classification is 'good'.",
    "Keep all trials after session inclusion so outcome=ignore, early-lick, and photostim conditions remain available for decoding.",
]
```

iii. `CONVERSION_NOTES.md` explicitly says this was a deliberate deviation from some reference analysis masks because the decoder specification requires `Photostimulation`, `Outcome`, and `Early lick` labels, including ignored and early-lick trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from raw unit spike timestamps in `units/spike_times` and `units/spike_times_index`, filtered by `units/classification == "good"`, then assigned to trials using behavioral trial intervals and aligned by per-trial go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii. 
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
...
go_events = np.asarray(behavioral_events["go_start_times"]["timestamps"][:], dtype=np.float64)
...
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    trial_start=trial_start,
    trial_stop=trial_stop,
    go_times=go_times,
)
```

iii. The notes describe the neural data as “spike counts per 50 ms bin, converted to firing rates” and emphasize that units are filtered using the NWB `classification` field to match the published classifier-based QC workflow.

## 2-b. How is the `neural` data processed?

i. The script processes spikes in session time rather than reconstructing per-trial spike arrays first. For each good unit it assigns spikes to trials with `searchsorted`, subtracts the matched trial’s go-cue time, keeps spikes in `[-2.5, 1.5)`, bins them into 50 ms bins with `np.add.at`, and converts counts to firing rates by multiplying by `1 / 0.05`.

ii. 
```python
trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
...
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The trajectory says the agent chose this implementation because the NWB stores spikes in session time and assigning each spike once would be much faster than nested Python loops over trials. The notes state the intended output is firing rates, not raw counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by the NWB spike-sorting QC label `classification == "good"`. Sessions with zero good units are dropped entirely. No additional per-unit filter such as a photostim-trial minimum is applied for the decoder conversion.

ii. 
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. `CONVERSION_NOTES.md` ties this directly to the paper’s classifier-based QC white paper and notes that excluding the one zero-good-unit session reproduces the paper-level `173` kept-session count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is the go cue. The script enforces one go-cue event per retained trial, subtracts that trial’s go-cue timestamp from each spike time, and keeps spikes only in the `[-2.5 s, +1.5 s)` window around go cue.

ii. 
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
...
"temporal_alignment_event": "Go cue onset",
"off_start": WINDOW_START_S,
"off_end": WINDOW_END_S,
```

iii. The instructions explicitly required go-cue alignment, and both the notes and trajectory say the agent confirmed the reference code treats go cue as time zero for neural, lick, and stimulation timing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use non-overlapping 50 ms bins across the 4 s window from `-2.5` to `+1.5` s, giving `80` time bins. There is no second-stage rebinning after the initial binning.

ii. 
```python
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```

iii. This follows the decoder instructions directly. The notes also state that the conversion uses 50 ms bins for all aligned streams.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from the behavioral event stream `sample_start_times` and the per-trial go-cue times, then combined with the fixed bin centers. The script treats the relevant tone onset as the last `sample_start_times` event before each trial’s aligned go cue.

ii. 
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The trajectory says the agent intentionally used the “final sample/tone onset before the aligned go cue,” which matters in trials with sample/delay replays after early licks. The notes summarize this as bin centers expressed relative to the final sample/tone onset before go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each retained trial, the code finds the last sample-start event before go cue, computes its time relative to go cue, and subtracts that offset from every 50 ms bin center. The result is a continuous time-varying signal: negative before the tone onset and positive after it.

ii. 
```python
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
...
np.vstack([
    time_from_tone[trial_idx],
    photostim[trial_idx],
]).astype(np.float32)
```

iii. The decoder task explicitly asked for “Time from tone onset in seconds (continuous, time-varying),” so the agent justified using a continuous value per bin rather than a binary onset indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined directly on the same `BIN_CENTERS_S` grid used for neural spike binning, so every neural time bin has a same-index time-from-tone value.

ii. 
```python
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
...
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
...
neural_trials.append(...)
input_trials.append(np.vstack([time_from_tone[trial_idx], photostim[trial_idx]]))
```

iii. The agent’s notes say all streams were reconciled onto one shared 50 ms grid so neural, photostim, and tongue variables stayed exactly aligned per trial.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `Photostimulation` is derived from the per-trial behavioral table fields `photostim_onset` and `photostim_duration`. The code does not use photostim power or type because the target decoder input only asks whether stimulation is on at each time bin.

ii. 
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
...
photostim, stim_trial_count = bin_photostim_series(
    trial_start=trial_start,
    go_times=go_times,
    onset_values=photostim_onset,
    duration_values=photostim_duration,
)
```

iii. `CONVERSION_NOTES.md` says the target variable is a binary “photostim on” trace, so onset and duration were sufficient. The trajectory also notes that reference code stored stimulation as on/off times relative to go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code parses `"N/A"` as no-stimulation, converts onset and duration to floats otherwise, shifts onset from trial-relative coordinates into go-cue-relative coordinates, computes the corresponding off time, and marks a bin as `1` whenever that photostim interval overlaps the bin.

ii. 
```python
onset = as_float_or_none(onset_values[trial_idx])
duration = as_float_or_none(duration_values[trial_idx])
if onset is None or duration is None:
    continue
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The notes describe the output as “binary per bin, 1 if the photostim interval overlaps the bin.” This choice follows naturally from the decoder requirement “whether photostimulation is on at every time point.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim trace is sampled on the same 50 ms bins used for neural data, after converting trial-relative stim timing into go-cue-relative timing.

ii. 
```python
bin_left = BIN_EDGES_S[:-1][None, :]
bin_right = BIN_EDGES_S[1:][None, :]
...
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
...
input_trials.append(
    np.vstack([
        time_from_tone[trial_idx],
        photostim[trial_idx],
    ]).astype(np.float32)
)
```

iii. The notes say all streams were put on a common go-cue-aligned 50 ms grid, which is the entire justification here.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code derives choice from the left- and right-lick event timestamp streams in `acquisition/BehavioralEvents`, plus the per-trial instructed side as a fallback when a trial has no licks at all.

ii. 
```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
instructions = decode_array(trial_group["trial_instruction"][:])[trial_indices].astype(str)
...
choice, choice_sources = compute_choice_labels(...)
```

iii. The trajectory explicitly says the trial table did not contain behavioral choice directly, only the instructed side, so the agent decided to reconstruct choice from lick timestamps and documented fallback counts for edge cases.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is assigned as `0` for left and `1` for right. The preferred rule is the first post-go lick in the trial. If no post-go lick exists, the code falls back to the first lick anywhere in the trial. If there are no licks at all, it falls back again to the instructed side. The final scalar choice label is then repeated across all time bins in the trial.

ii. 
```python
if left_post.size or right_post.size:
    left_first = left_post[0] if left_post.size else np.inf
    right_first = right_post[0] if right_post.size else np.inf
    choice[trial_idx] = 0 if left_first < right_first else 1
    ...
if left_all.size or right_all.size:
    ...
    choice[trial_idx] = 0 if left_first < right_first else 1
    ...
choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
...
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8)
```

iii. The notes and trajectory both justify this as a pragmatic way to produce the required binary left/right label even on ignore or no-lick trials, while still preferring genuine post-go behavior when available.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` is taken directly from the trial table field `intervals/trials/outcome`.

ii. 
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
```

iii. No deeper justification was needed beyond the decoder spec, because the NWB trial table already stores the paper-level outcome categories directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps the NWB strings to integer categories `ignore=0`, `miss=1`, `hit=2`, then repeats the per-trial scalar label across all 50 ms bins.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
...
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8)
```

iii. This exactly follows the decoder instructions, and the notes list the same mapping.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The question appears to contain a typo: the code aligns `output` *Outcome*, not any distance-to-reward-zone variable. `Outcome` is aligned by repeating the trial-level category across the same 50 ms bins used for the neural data.

ii. 
```python
output_trials.append(
    np.vstack([
        np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8),
        np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
        np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
        tongue_disc[trial_idx],
    ])
)
```

iii. The agent justified repeating per-trial outputs across bins because the validator accepts time-varying outputs and because it wanted all decoder outputs to share the neural time axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is derived directly from the trial table field `intervals/trials/early_lick`.

ii. 
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
```

iii. No special justification appears beyond using the existing NWB trial annotation and matching the decoder label definition.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `"no early"` to `0` and `"early"` to `1`, then repeats that trial-level value across the 50 ms bins for the trial.

ii. 
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
...
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` lists the same mapping and explains that early-lick trials were intentionally retained because the decoder output explicitly requires them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera tongue tracking time series `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using both the `data` array and its `timestamps`. The code specifically takes column `1` from `data` as the raw tongue `y` coordinate.

ii. 
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. The notes cite the video-tracking reference code and say the conversion uses the side-camera tongue `y` position because that is the variable requested by the decoder task.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script computes session-wide 40th and 60th percentiles from the full raw tongue `y` trace, then for each trial/bin it samples the last tongue frame that falls inside that 50 ms bin. If a bin has no new frame, it falls back to the latest sample before the bin end.

ii. 
```python
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
...
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

iii. The notes justify this by citing the reference marker-alignment style: use the last sample inside each interval. The trajectory says the agent reconciled the higher-rate video stream onto the shared 50 ms neural grid.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After binning the continuous `y` values, the code discretizes them session-wise into `0` for below the 40th percentile, `1` for between the 40th and 60th percentiles inclusive, and `2` for above the 60th percentile.

ii. 
```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. This exactly follows the decoder instructions; the notes repeat the same three-category definition.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue position is aligned to go cue by building per-trial bin edges `go_times + BIN_EDGES_S`, then sampling the tongue trace into those exact bins. The resulting discretized sequence has one value per neural bin.

ii. 
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
...
output_trials.append(
    np.vstack([
        ...,
        tongue_disc[trial_idx],
    ])
)
```

iii. The trajectory states that the agent intentionally forced neural, photostim, and tongue labels onto the same 50 ms grid so they would stay exactly aligned trial-by-trial.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly validates aggressively and only has a few explicit fallbacks. It treats `"N/A"`/empty photostim fields as missing stimulation, skips sessions with zero good units, maps ephys-covered trials back to behavior rows to avoid behavior-only trials, uses the last available tongue frame when a 50 ms bin has no sample, and raises `ValueError` if key assumptions fail (for example, not exactly one go cue per retained trial, no sample event before go cue, unmatched `obs_intervals`, or missing tongue time series).

ii. 
```python
if value in ("N/A", "", None):
    return None
...
if n_good_units == 0:
    return None
...
if "BehavioralTimeSeries" not in f["acquisition"]:
    raise ValueError(...)
...
fallback_missing = int(np.sum(right_idx <= left_idx))
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
...
raise ValueError(f"Could not match obs_interval {key} ...")
```

iii. The trajectory shows these checks were added while debugging zero-neural trials and alignment inconsistencies. The notes describe them as sanity checks to prevent silent corruption and to preserve alignment without introducing NaNs.

## 10-a. What are the most time-consuming steps of the code?

i. The main hotspot is spike binning: `bin_spike_counts_for_good_units()` loops through every unit, slices that unit’s full spike train, assigns spikes to trials, computes relative times, and updates bins with `np.add.at`. The per-trial choice reconstruction is also expensive because it scans the global left/right lick timestamp arrays separately for every trial.

ii. 
```python
for unit_idx, unit_end in enumerate(spike_times_index):
    spikes = spike_times_flat[unit_start:unit_end]
    ...
    trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
    ...
    np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)

for trial_idx in range(len(trial_start)):
    left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
    right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
```

iii. The trajectory explicitly says the agent optimized around spike assignment because full conversion would otherwise be too slow. Its own notes identify end-to-end full conversion and validation as the dominant workload.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python loops that could be vectorized further: the per-trial loop in `compute_choice_labels()`, the per-trial loop in `bin_photostim_series()`, and the final packaging loop that builds `neural_trials`, `input_trials`, and `output_trials` trial-by-trial.

ii. 
```python
for trial_idx in range(len(trial_start)):
    ...

for trial_idx in range(n_trials):
    ...

for trial_idx in range(n_trials):
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The trajectory only explicitly defends the spike-binning loop as a pragmatic performance compromise; it does not defend these smaller per-trial loops, so they appear to be convenience choices rather than deliberate optimizations.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the lick timestamp arrays once per trial to recover choice, repeatedly constructs per-trial constant vectors for `choice`, `outcome`, and `early_lick`, and loops over all trials again after binning to package outputs into Python lists. It also decodes multiple string arrays from NWB one by one with the same helper.

ii. 
```python
left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
...
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8)
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8)
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8)
...
decode_array(trial_group["trial_instruction"][:])
decode_array(trial_group["outcome"][:])
decode_array(trial_group["early_lick"][:])
```

iii. There is no explicit note defending this repetition. The trajectory mainly discusses correctness and the spike-binning optimization, not these repeated postprocessing passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is the expansion of scalar trial labels (`choice`, `outcome`, `early_lick`) into full-length 80-bin vectors even though they do not change within a trial and the target format would also permit per-trial scalars. The script also computes and stores many bookkeeping statistics (`choice_sources`, quantiles, fallback counts, session summaries) that are useful for notes and metadata but not for downstream decoding itself.

ii. 
```python
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8)
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8)
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8)
...
"choice_source_counts": dict(total_choice_sources),
"stim_trial_count_total": int(total_stim_trials),
"tongue_bin_fallback_count_total": int(total_tongue_fallback),
"session_summary": session_summaries,
```

iii. The trajectory justifies the time-varying output representation as validator-friendly and alignment-friendly, not as the minimal downstream representation. The extra metadata and counters were explicitly added for documentation and sanity checking.
