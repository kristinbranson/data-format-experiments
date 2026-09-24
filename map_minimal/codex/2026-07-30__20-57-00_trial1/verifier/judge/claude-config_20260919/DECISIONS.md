# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB (HDF5) file per session under `data/sub-<id>/`. The AI enumerates every subject directory and every `.nwb` file inside it with `os.listdir`, sorted at both levels, giving a deterministic, complete list of 174 session files. Each file is opened **once** with raw `h5py` (not `pynwb`), and every stream is read by explicit HDF5 path: `units/*` (classification, spike_times + spike_times_index, obs_intervals + obs_intervals_index, anno_name, is_good_trials), `intervals/trials/*` (start/stop, instruction, outcome, early_lick, photostim_*), `acquisition/BehavioralEvents/*` (go, sample, left/right lick timestamps) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Sessions are processed one at a time and accumulated in a list, then assembled into the target dictionary.

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
```

```python
def convert_session(path: str) -> dict[str, Any] | None:
    with h5py.File(path, "r") as f:
        classifications = decode_array(f["units/classification"][:]).astype(str)
        ...
        trial_group = f["intervals/trials"]
        behavioral_events = f["acquisition/BehavioralEvents"]
        go_events = np.asarray(behavioral_events["go_start_times"]["timestamps"][:], dtype=np.float64)
```

```python
all_paths = load_sorted_nwb_paths(args.data_root)
print(f"Found {len(all_paths)} NWB files under {args.data_root}")
for idx, path in enumerate(all_paths, start=1):
    result = convert_session(path)
```

iii. From the trajectory (steps 17-33) the AI first enumerated the HDF5 hierarchy of one file (`f.visititems`), then switched from `pynwb` to plain `h5py` for the production path after inspecting the ragged storage layout (`spike_times` + `spike_times_index`). Its stated reasoning at step 74: *"I'm going to process spikes in session time, not trial-by-trial... That matches the NWB storage layout much better than reconstructing thousands of per-trial spike arrays in Python loops and should keep the full conversion tractable."* `CONVERSION_NOTES.md` records that all 174 files were found and that the count matches the published dataset.

## 1-b. How are the data split into subjects?

i. The subject label is taken from the **parent directory name** of each NWB file, i.e. the string `"sub-440956"`. `subjects` is the sorted set of those 28 strings; `subject_idx` is each session's index into that list, in session order. (The reference instead reads `nwb.subject.subject_id`, which is the same identifier without the `sub-` prefix — the resulting partition of sessions into animals is identical.)

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["subject"]])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

iii. Not explicitly argued in the trajectory. The DANDI layout puts one directory per animal, so the directory name *is* the subject id; the AI's summary confirms the result (28 subjects, 3-10 sessions each) against the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. The session id is the filename stem (e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order in the output follows the sorted directory walk, which is chronological within a subject because the acquisition timestamp is embedded in the filename. Session ids are also recorded in `metadata['session_ids']` and `metadata['session_summary']`.

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

```python
"session_ids": [session["session_id"] for session in session_results],
"session_summary": session_summaries,
```

iii. Implicit: the dandiset already stores one session per file, so the file boundary is the session boundary. 173 of 174 files reach the output; the one exclusion is documented (see 2-c).

## 1-d. How are the data split into trials?

i. Trials are rows of the NWB trials table (`intervals/trials`), subset to the trials actually covered by the ephys recording (see 1-e). Each retained trial gets its `start_time`/`stop_time`, and the AI **asserts** that exactly one `go_start_times` event falls inside `[start_time, stop_time]` for every trial, using that event as the trial's go cue. The tone is taken as the last `sample_start_times` event before the go cue, with an assertion that it is not earlier than the trial start (this handles the fact that an early lick replays the sample epoch, giving several sample events per trial).

ii.
```python
def assert_one_event_per_trial(event_times, trial_start, trial_stop, name):
    left = np.searchsorted(event_times, trial_start, side="left")
    right = np.searchsorted(event_times, trial_stop, side="right")
    counts = right - left
    if not np.all(counts == 1):
        bad = np.where(counts != 1)[0][:10]
        raise ValueError(f"{name}: expected exactly one event per trial, ...")
    return event_times[right - 1]
```

```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
```

iii. The AI verified the trials table contents at step 21 (`intervals/trials` columns) and checked the event-per-trial mapping empirically before committing. `CONVERSION_NOTES.md`: *"Verified that the NWB files expose the same behavior/task variables used by the reference preprocessing code: go cue, sample/tone timing, photostim fields, early-lick status, outcome, and lick times."*

## 1-e. How are trials filtered based on quality controls?

i. **One filter only**: trials not covered by the electrophysiology. The first good unit's `obs_intervals` rows are matched back to the behavioural trials table by exact `(start_time, stop_time)` pairs rounded to 4 decimals; only matched trials are kept. The AI also asserts the match is unique, strictly ordered, and that the interval count equals `units/is_good_trials.shape[1]`. **No behavioural quality filter is applied** — early-lick, `ignore`, photostim and `free_water` trials are all retained deliberately. This leaves 93,310 trials over 173 sessions (the expert's solution additionally drops the 2,450 `free_water` trials and keeps 90,860).

ii.
```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
...
first_unit_obs = obs_intervals[obs_start:obs_stop]
if first_unit_obs.shape[0] != n_ephys_trials:
    raise ValueError(f"Unexpected obs_intervals length in {path}")

# Match the ephys-covered trials back to the behavioral trial table by exact interval.
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    if key not in trial_lookup:
        raise ValueError(f"Could not match obs_interval {key} to a behavioral trial in {path}")
    trial_indices.append(trial_lookup[key])
```

```python
"inclusion_rules": [
    "Include NWB sessions with at least one unit whose classification is 'good'.",
    "Keep all trials after session inclusion so outcome=ignore, early-lick, and photostim conditions remain available for decoding.",
    "Use go-cue alignment and 50 ms non-overlapping bins.",
],
```

iii. The `obs_intervals` filter was discovered by debugging, not assumed. Step 137: *"The zero-neural trials are explained now: some NWB files contain more behavioral trials than recorded ephys trials. In the problematic session, every good unit's `obs_intervals` stops at trial 160 even though the behavior table continues to trial 480."* The AI first used `is_good_trials.shape[1]` as a leading-block count, found a session where that assumption fails (`sub-455219_ses-20190807T134913`), and generalised to exact interval matching (step 166).

For behavioural filtering, step 39: *"the reference code excludes stimulation for some analyses, but this decoder explicitly needs photostim as an input, so I'm treating stimulation, early-lick, and ignore trials as required retained data."* On the 2,452 remaining all-zero-neural trials reported by the validator, step 169: *"There are many warnings about entirely silent trials, which reflects real sparsity in some session/trial windows rather than a format error"* — i.e. the AI saw the symptom that led the expert to the `free_water` exclusion and attributed it to sparsity instead of investigating.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (the flat ragged buffer of session-absolute spike times) together with `units/spike_times_index` (per-unit end offsets), restricted to units whose `units/classification == 'good'`. `intervals/trials/start_time` and `stop_time` and the per-trial go-cue times are the other inputs, used to assign spikes to trials and to place the bins. Region labels come from `units/anno_name`.

ii.
```python
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    trial_start=trial_start,
    trial_stop=trial_stop,
    go_times=go_times,
)
```

iii. `spike_times` is the only neural representation in the file. The AI inspected the ragged layout explicitly (step 75-76: printing `spike_times`, `spike_times_index`, `obs_intervals`, `obs_intervals_index` shapes and first values) before writing the binning routine.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, divided by the bin width to give firing rates in Hz, stored as `float16` with shape `(n_neurons, 80)` per trial. The implementation walks the flat spike buffer unit by unit; for each good unit every spike is (a) assigned to a trial via `searchsorted` on `trial_start`, (b) **discarded unless it also falls at or before that trial's `stop_time`**, (c) expressed relative to that trial's go cue, (d) kept if `-2.5 <= rel < 1.5`, and (e) scatter-added into its bin with `np.add.at`. No smoothing, normalisation or baseline subtraction.

ii.
```python
counts = np.zeros((n_trials, n_bins, n_good_units), dtype=np.uint8)
...
    trial_idx = np.searchsorted(trial_start, spikes, side="right") - 1
    valid = (trial_idx >= 0) & (trial_idx < n_trials)
    if np.any(valid):
        valid_trial_idx = trial_idx[valid]
        valid &= spikes <= trial_stop[valid_trial_idx]

    if np.any(valid):
        valid_trial_idx = trial_idx[valid]
        rel_spikes = spikes[valid] - go_times[valid_trial_idx]
        in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
        valid_trial_idx = valid_trial_idx[in_window]
        rel_spikes = rel_spikes[in_window]

        if rel_spikes.size:
            bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
            np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```

```python
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. `CONVERSION_NOTES.md`: *"Neural data: spike counts per 50 ms bin, converted to firing rates by dividing by `0.05 s`."* The trial-gating step is not mentioned anywhere in the notes or the trajectory; it is a side effect of the "assign each spike to a trial once, in session time" strategy adopted at step 74 for speed.

Measured consequence (my verification): the `[go-2.5, go+1.5)` window extends past `trial_stop` on **15.6 % of trials** across the dataset, so **2.8 % of all (trial, bin) cells are forced to exactly zero across every simultaneously recorded neuron**. In `sub-440956_ses-20190207T120657` (459 good units) 36 % of trials have all 459 neurons silent for the last ~0.6 s of the window, and 10 % of that session's (trial, bin) cells are all-zero.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept iff `units/classification == 'good'` — the verdict of the spike-sorting QC classifier. No individual metric thresholds, and `unit_quality` is not used. A session with zero good units is dropped entirely. Result: 69,453 units over 173 sessions, with `sub-440958_ses-20190216T162508` the single dropped file. This is identical to the expert solution.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. `CONVERSION_NOTES.md`: *"Units are filtered with `classification == "good"`, matching the published classifier-based QC workflow"*; *"This removes exactly one NWB file with zero good units, leaving 173 sessions, which matches the paper-level session count"*; *"Verified that the full NWB collection contains 69,453 `classification == "good"` units, close to the published total of 69,943 and consistent with the same classifier-based QC source."* The AI compared `classification`, `anno_name` and `unit_quality` value counts at step 29 before choosing `classification`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**, taken from `acquisition/BehavioralEvents/go_start_times` — the single event inside each trial's `[start_time, stop_time]`. Spike times and event timestamps share one session-absolute clock, so alignment is a subtraction: `rel_spikes = spikes - go_times[trial]`, and bins are indexed by `floor((rel - (-2.5)) / 0.05)`. No resampling or interpolation. `metadata['temporal_alignment_event'] = 'Go cue onset'`, `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```

```python
"temporal_alignment_event": "Go cue onset",
"off_start": WINDOW_START_S,
"off_end": WINDOW_END_S,
```

iii. Set by the Decoder Task section of the instructions. Step 44: *"The alignment code in the repo uses two slightly different conventions: neural firing rates are built from explicit bin centers, while video markers are sampled on a fixed frame grid... I'm reconciling those conventions into one shared 50 ms grid so neural, photostim, and tongue labels stay exactly aligned per trial."* The go-cue-relative grid `BIN_EDGES_S`/`BIN_CENTERS_S` is defined once at module level and reused by the neural, photostim and tongue paths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 non-overlapping bins spanning `[-2.5 s, +1.5 s)` relative to the go cue, identical for every trial and session. Spikes are binned once from raw spike times — there is no rebinning of an already-binned product. `metadata['time_bin_size'] = 50.0` (ms), and the edges/centres/`n_timepoints` are also written to metadata. (I verified `np.arange(-2.5, 1.55, 0.05)` yields exactly 81 edges → 80 bins; the last edge carries ~1.4e-14 of float drift, which is harmless because the window test uses the exact literal `1.5`.)

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

```python
"time_bin_size": BIN_SIZE_S * 1000.0,
"bin_edges_s": BIN_EDGES_S.tolist(),
"bin_centers_s": BIN_CENTERS_S.tolist(),
"n_timepoints": int(len(BIN_CENTERS_S)),
```

iii. Directly from the instructions (*"Use 50-ms-width bins"*, *"2.5 s before to 1.5 s after"*). `CONVERSION_NOTES.md`: *"Trials are aligned to go cue onset and restricted to the interval `[-2.5 s, +1.5 s)` using 50 ms bins."*

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times` (the instruction-tone onsets) and the per-trial go-cue time. The tone used for a trial is the **last** sample event strictly before that trial's go cue, with a guard that it is not earlier than the trial start.

ii.
```python
def last_event_before_per_trial(event_times, lower_bound, upper_bound, name):
    idx = np.searchsorted(event_times, upper_bound, side="left") - 1
    if np.any(idx < 0):
        raise ValueError(f"{name}: missing prior event for trial indices=...")
    values = event_times[idx]
    valid = values >= lower_bound
    if not np.all(valid):
        raise ValueError(f"{name}: event before lower bound for trial indices=...")
    return values
```

```python
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. `CONVERSION_NOTES.md`: *"Input `time_from_tone_onset_s`: bin centers expressed relative to the final sample/tone onset before the aligned go cue."* Taking the *last* sample event handles the fact that a lick during the sample/delay epoch replays that epoch, so a trial can contain several sample onsets — this is exactly the reference's reasoning.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: each bin's value is its centre (relative to the go cue) minus the tone's offset relative to the go cue, i.e. `centre + (go - tone)`. Stored as `float32`, row 0 of the `(2, 80)` per-trial input array, named `time_from_tone_onset_s`. No clipping, no binarisation.

ii.
```python
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

```python
input_trials.append(
    np.vstack([
        time_from_tone[trial_idx],
        photostim[trial_idx],
    ]).astype(np.float32)
)
```

iii. The instruction specifies this input as *"continuous, time-varying"*, so the AI emitted a per-bin real value rather than a one-hot onset. Algebraically identical to the expert's `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed **on the same go-cue-relative grid** that defines the neural bins: the module-level `BIN_CENTERS_S` array is the single source of truth for both, so input bin *k* covers exactly the same interval as neural bin *k*. Nothing else is needed, because both the tone event and the go cue live on the same session clock.

ii.
```python
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```
```python
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. Step 44 (quoted above): the AI deliberately reconciled the reference repo's two grid conventions into "one shared 50 ms grid so neural, photostim, and tongue labels stay exactly aligned per trial."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` — both stored as **strings**, with `'N/A'` on unstimulated trials, and with the onset measured from the trial's `start_time`. `start_time` and the go cue are used to re-express them on the go-cue axis.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```
```python
def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)
```

iii. The AI inspected the stimulated rows of the trials table directly (step 62, printing `photostim_onset`, `photostim_duration`, `photostim_power` for stim trials) and found the string/`'N/A'` encoding, hence the explicit parsing helper.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series (`float32` 0/1), row 1 of the per-trial input array, named `photostim_on`. For each stimulated trial the onset/offset are converted to go-cue-relative seconds, and a bin is set to 1 if the stimulation interval **overlaps** the bin at all (`bin_left < rel_off and bin_right > rel_on`). Unstimulated trials (onset or duration `'N/A'`) are left as all zeros. 18,173 of 93,310 trials are stimulated. (The expert instead marks a bin when its *centre* lies in `[on, off)`; the two rules differ by at most one bin at each edge.)

ii.
```python
photostim = np.zeros((n_trials, n_bins), dtype=np.float32)
bin_left = BIN_EDGES_S[:-1][None, :]
bin_right = BIN_EDGES_S[1:][None, :]

for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
    duration = as_float_or_none(duration_values[trial_idx])
    if onset is None or duration is None:
        continue
    stim_trial_count += 1
    rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
    rel_off = rel_on + duration
    overlap = (bin_left < rel_off) & (bin_right > rel_on)
    photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. `CONVERSION_NOTES.md`: *"Input `photostim_on`: binary per bin, 1 if the photostim interval overlaps the bin."* The instruction asks for *"Whether photostimulation is on at every time point (discrete, time-varying)"*, so a per-bin binary series rather than a per-trial flag. The AI kept stimulated trials on purpose (step 39: the decoder "explicitly needs photostim as an input").

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset is converted from trial-relative to go-cue-relative (`trial_start + onset - go`), then compared against the same `BIN_EDGES_S` array used for the neural bins. Both streams therefore share one grid, and no interpolation or offset correction is involved.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
```

iii. Same shared-grid rationale as 3-c. I verified on `sub-440956_ses-20190207T120657` that stimulation onsets sit at ≈ -1.2 s relative to the go cue with 0.5 s duration, i.e. comfortably inside the extracted window.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the **actual lick timestamps**: `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`, restricted to `[trial_start, trial_stop]`. Three sources, in priority order:
1. first lick at or after the go cue → that side (79,585 trials),
2. otherwise first lick anywhere in the trial, including pre-go/early licks (1,063 trials),
3. otherwise `intervals/trials/trial_instruction` — the *instructed* side is used as a stand-in (12,662 trials, 13.6 %).

The trials table has no choice column, so it has to be derived; the expert instead derives it from `trial_instruction` × `outcome`.

ii.
```python
left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
left_post = left_all[left_all >= go_time]
right_post = right_all[right_all >= go_time]

if left_post.size or right_post.size:
    left_first = left_post[0] if left_post.size else np.inf
    right_first = right_post[0] if right_post.size else np.inf
    choice[trial_idx] = 0 if left_first < right_first else 1
    source_counter["post_go_lick"] += 1
    continue

if left_all.size or right_all.size:
    ...
    source_counter["any_trial_lick"] += 1
    continue

choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
source_counter["instruction_fallback"] += 1
```

iii. Step 68: *"I've confirmed the trial table does not contain behavioral choice directly; it only has the instructed side. The NWB events do preserve left/right lick timestamps, so I'm deriving choice from the first post-go lick within each trial."* Step 80: *"`ignore` trials with no post-go lick are genuinely no-choice trials. For the required left/right decoder label, I'm using first post-go lick when available, otherwise first lick anywhere in the trial, and only falling back to instructed side for the rare trials with no lick at all."* The AI logged the three source counts to `metadata['choice_source_counts']`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right` as `int8`, **two classes only**; `OUTPUT_VALUES[0] = ["left", "right"]`. The per-trial scalar is broadcast across all 80 bins into row 0 of the `(4, 80)` output array. No "no lick" class exists: the AI recognised the no-lick trials (its own counter calls them `instruction_fallback`) but assigned them the instructed side, so 12,662 trials carry a fabricated left/right label. The Decoder Task section of the instructions specifies three values for this output (*"left, right, no lick"*), and the expert emits `2 = 'no lick'` for `outcome == 'ignore'`.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
```
```python
choice = np.zeros(len(trial_start), dtype=np.int8)
...
choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
source_counter["instruction_fallback"] += 1
```
```python
np.full(len(BIN_CENTERS_S), choice[trial_idx], dtype=np.int8),
```
```python
"choice_definition": (
    "First post-go lick if available; otherwise first lick anywhere in the trial; "
    "otherwise instructed side fallback for no-lick trials."
),
```

iii. The AI's stated justification (step 80) is that the label is "required left/right", i.e. it read the output as binary and treated no-lick trials as an edge case to be imputed rather than as a third category. It disclosed the imputation in `metadata['choice_source_counts']` and `CONVERSION_NOTES.md` (*"12,662 no-lick trials requiring instructed-side fallback"*) rather than hiding it.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`, subset to the retained trials.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
```

iii. No derivation needed — the column matches the requested categories one-for-one. The AI confirmed the value set across sessions before relying on it (step 79, tabulating outcome/early/instruction distributions over all sessions).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore→0`, `miss→1`, `hit→2` (`int8`), matching the order given in the instructions and `OUTPUT_VALUES[1] = ["ignore", "miss", "hit"]`. The per-trial value is repeated across all 80 bins into row 1 of the output array. Counts are tallied into `metadata['outcome_value_counts']`.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```
```python
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
```

iii. `CONVERSION_NOTES.md`: *"Output `outcome`: `ignore=0`, `miss=1`, `hit=2`."* Coding order follows the instructions' listing. The dict lookup raises `KeyError` on any unexpected string, so an unknown category would fail loudly rather than be silently mislabelled.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, a string column holding `'no early'` / `'early'`, subset to the retained trials.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
```

iii. The trials table flags early licking explicitly; the AI verified the column's value set across sessions (step 79) before mapping it.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped `'no early'→0`, `'early'→1` (`int8`), `OUTPUT_VALUES[2] = ["no", "yes"]`, repeated across all 80 bins into row 2 of the output array. Counts tallied into `metadata['early_lick_value_counts']`.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```
```python
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
```

iii. `CONVERSION_NOTES.md`: *"Output `early_lick`: `no=0`, `yes=1`."* Coding order follows the instructions (*"no, yes"*). Identical to the expert.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` is `(n_frames, 3)` and `timestamps` is the matching frame-time vector. The AI takes **column 1 only** (`tongue_y`). **Column 2, `tongue_likelihood`, is read into `tongue_data` but never used** — every frame is treated as a valid tongue measurement. The series' own `description` attribute states the channel layout `('tongue_x', 'tongue_y', 'tongue_likelihood')`. In `sub-440956_ses-20190207T120657` only 10.6 % of frames have likelihood ≥ 0.5; the other 89 % still carry a `tongue_y` value spanning the full range (1st-99th pct ≈ -5 to 364), i.e. tracker output for a tongue that is not protruding.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. This is the only tongue measurement in the file; the AI located it by walking the HDF5 tree for `tongue`/`jaw`/`marker` keys (step 20) and printed the first rows and timestamps (step 27). On the likelihood channel, the AI searched the reference repo for any use of it (step 65: `rg -n "likelihood|tongue_likelihood|..." /app/code` → **no matches**) and, finding none, did not apply one. The reference repo's marker pipeline indeed carries only x/y, but the NWB export exposes the likelihood explicitly and the expert used it.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps, both on the **raw, unfiltered** `tongue_y`:
1. **Per-session thresholds**: `q40`/`q60` are the 40th/60th percentiles of `tongue_y` over every frame of the session, including the ~89 % of frames where the tongue is not visible.
2. **Per-bin value**: one representative frame per bin — the last frame strictly before the bin's right edge (`searchsorted(..., 'left') - 1`), *not* an average of the frames in the bin. If a bin contains no frame at all, the index is not reset: the previous frame's value is carried forward, and only a counter is incremented (149,144 bins dataset-wide, ~2 %). No NaN handling exists because no value is ever marked invalid.

ii.
```python
tongue_y_raw = tongue_data[:, 1]
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```
```python
    trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
    left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
    right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")

    # Match the marker alignment style in the reference repo by using the last sample
    # available inside each bin. If a bin has no sample, fall back to the latest sample
    # before the bin end.
    fallback_missing = int(np.sum(right_idx <= left_idx))
    sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
    y_binned = tongue_y[sample_idx]
```

iii. The last-frame-in-bin rule is explicitly grounded in the reference repo, which the AI read at step 41 — `align_markers_between_lims` in `code/Sherlock/align_markers.py` does exactly this: *"Use the embedding at the last time point within the time range"*, `marker_vecs[i, trial, :] = _this_embed_array[:, np.where(mask)[0][-1]]`. `CONVERSION_NOTES.md`: *"Binned values use the last frame inside each 50 ms bin, matching the reference marker-alignment style"*, and *"Tongue binning required previous-frame fallback in 149,144 bins... this comes from the marker stream's sampling gaps relative to fixed 50 ms bins and preserves alignment without introducing NaNs."* No justification is given for computing the percentiles over untracked frames — the likelihood channel simply never enters the decision.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. **Three** classes: `0` if the binned value `< q40`, `2` if `> q60`, `1` otherwise (the `np.ones` initialiser supplies the middle class, so values exactly equal to a threshold land in class 1). `OUTPUT_VALUES[3] = ["lt_40pct", "p40_to_p60", "gt_60pct"]`. The instructions specify a **fourth** class, `3: not visible`, which is absent; `train_decoder.py` consequently reports chance = 1/3 for this output rather than 1/4. Because the percentiles are taken over all frames and every bin is assigned a class, the class balance is ≈ 40/20/40 by construction and is driven mostly by untracked frames rather than by tongue protrusion.

ii.
```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
return tongue_disc, fallback_missing
```
```python
OUTPUT_VALUES = [
    ...
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
```

iii. `CONVERSION_NOTES.md`: *"Output `tongue_y_position`: side-camera tongue `y` position, discretized per session using the 40th and 60th percentiles of the raw session-wide tongue `y` trace."* The per-session scope and the 40/60 split follow the instructions; the "not visible" class is never discussed in the notes or anywhere in the trajectory, and the previous-frame fallback was chosen specifically to avoid ever needing a missing-data class ("preserves alignment without introducing NaNs").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. On the same go-cue-relative grid: `trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]` builds each trial's absolute bin edges from the identical `BIN_EDGES_S` used for the spikes, and frames are located by `searchsorted` on the camera timestamps. Camera timestamps share the session-absolute clock with spikes and events, so no interpolation or offset correction is applied, and tongue bin *k* covers the same interval as neural bin *k*. The one deviation is the carry-forward on empty bins, which can import a frame from before the bin (the camera runs at ~294 Hz with only 0.05 % of inter-frame gaps exceeding 50 ms, so this affects ~2 % of bins).

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
```
```python
tongue_disc, tongue_fallback_missing = bin_tongue_y(
    timestamps=tongue_timestamps,
    tongue_y=tongue_y_raw,
    go_times=go_times,
    q40=q40, q60=q60,
)
```

iii. Step 44 again: the AI's stated goal was one shared 50 ms grid across neural, photostim and tongue so the streams "stay exactly aligned per trial". This is the only genuinely time-varying output, so it is the only one where alignment does work.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Mixed: some cases fail loudly, some are imputed, one class of missing data is not recognised at all.

- **Byte/str and `'N/A'` encodings**: decoded centrally; `'N/A'`, `''` and `None` map to `None` and mean "no photostimulation".
- **Session never quality-controlled** (`classification` not a usable string): produces zero `'good'` units, and the session is dropped (`return None`), logged as `skipped_zero_good_sessions`.
- **Behavioural trials with no ephys**: excluded via `obs_intervals` matching (1-e).
- **Structural surprises fail loudly**: missing `BehavioralTimeSeries`, an `obs_intervals` row that matches no behavioural trial, duplicate or non-monotonic matches, an `obs_intervals` length that disagrees with `is_good_trials`, or a trial without exactly one go cue all raise `ValueError` and abort the run rather than being silently absorbed.
- **Camera bins with no frame**: value carried forward from the previous frame; counted but not marked.
- **Not handled**: frames where the tongue is not tracked (no likelihood filter, no "not visible" class), no-lick trials (imputed from the instructed side), and trials/bins with no spike data (`free_water` trials, and the post-`trial_stop` part of the window) — these are emitted as genuine zeros/labels rather than flagged as missing.

ii.
```python
def as_float_or_none(value: Any) -> float | None:
    value = decode_scalar(value)
    if value in ("N/A", "", None):
        return None
    return float(value)
```
```python
if n_good_units == 0:
    return None
```
```python
if key not in trial_lookup:
    raise ValueError(f"Could not match obs_interval {key} to a behavioral trial in {path}")
if np.unique(trial_indices).size != trial_indices.size:
    raise ValueError(f"obs_intervals map to duplicate behavioral trials in {path}")
if np.any(np.diff(trial_indices) <= 0):
    raise ValueError(f"obs_intervals are not strictly ordered in {path}")
```
```python
fallback_missing = int(np.sum(right_idx <= left_idx))
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The loud-failure style is deliberate: the AI added each assertion after an empirical check (step 137 → `obs_intervals`; step 166 → generalised matching after `sub-455219` broke the leading-block assumption). For the imputations, the stated principle is to avoid NaNs/holes in a fixed-shape array ("preserves alignment without introducing NaNs"; the choice fallback "for the rare trials with no lick at all"), and every imputation is counted and surfaced in `metadata` (`choice_source_counts`, `tongue_bin_fallback_count_total`). The one case the AI misread is the all-zero-neural trials, attributed to "real sparsity" (step 169) rather than to missing spike data.

## 10-a. What are the most time-consuming steps of the code?

i. The AI never profiled or documented this. From the trajectory, the full run took ≈ 8 min 40 s wall clock for 174 sessions (started 01:13:33, `converted_data.pkl` present at 01:22:14), ≈ 3 s/session, producing a 5.8 GB pickle. The dominant costs are, in order: (1) HDF5 reads — the entire `units/spike_times` buffer (up to ~10^7 doubles) and the `(n_frames, 3)` tongue array (~680 k × 3) are pulled into memory per session; (2) the per-good-unit `np.add.at` scatter-add in `bin_spike_counts_for_good_units`, which is numpy's slow unbuffered path and runs over every spike of every good unit; (3) `compute_choice_labels`, a Python loop over trials that re-masks the *entire* session lick arrays four times per trial (O(n_trials × n_licks)); (4) pickling 5.8 GB at the end, plus a second 113 MB pickle. `decode_array` is also a Python-level list comprehension over every element of several full-length string columns (including all non-good units).

ii.
```python
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```
```python
for trial_idx in range(len(trial_start)):
    left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
    right_all = right_lick_times[(right_lick_times >= start) & (right_lick_times <= stop)]
```
```python
def decode_array(values: np.ndarray) -> np.ndarray:
    return np.array([decode_scalar(v) for v in values], dtype=object)
```

iii. The AI chose the session-time spike assignment for speed (step 74, quoted in 2-a) and monitored the run with `ps` rather than instrumenting it (steps 147-159: *"the process is actively using CPU... still healthy, so I'm continuing to wait"*). Runtime was well inside the task budget, so no optimisation was attempted.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops are avoidable:
- `compute_choice_labels` — one full-array boolean mask over both lick vectors per trial. Replaceable by two `np.searchsorted` calls over the sorted lick times for all trials at once.
- `bin_photostim_series` — one iteration per trial to build an `(1, 80)` overlap mask. The whole thing is a single broadcast comparison of `(n_trials, 1)` onsets/offsets against `(1, 80)` edges (the expert does exactly this in two lines).
- the `trial_lookup` dict-building loop plus the per-`obs_interval` lookup loop — replaceable by `np.isin`/`searchsorted` on the rounded start times.
- the final per-trial list comprehension that transposes and rescales `spike_counts[trial_idx]`, and the `[outcome_map[x] for x in ...]` label loops.

The per-unit loop in `bin_spike_counts_for_good_units` cannot be collapsed (units are ragged), but its inner `np.add.at` could be replaced by a single `np.bincount` on a flattened `(trial, bin)` index, which is substantially faster.

ii.
```python
for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
    duration = as_float_or_none(duration_values[trial_idx])
    if onset is None or duration is None:
        continue
    rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
    rel_off = rel_on + duration
    overlap = (bin_left < rel_off) & (bin_right > rel_on)
    photostim[trial_idx] = overlap.astype(np.float32)[0]
```
```python
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
for start, stop in first_unit_obs:
    ...
```

iii. Not discussed. The tongue path *was* vectorised across trials (`trial_edges` is computed as a full `(n_trials, 81)` matrix), showing the AI knew the technique; it simply did not apply it to the choice and photostim paths, where per-trial `if` logic made a loop the easier expression.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and converted exactly once, so no per-session work is duplicated. What *is* repeated is the assembly stage: `build_dataset` is called a second time on the first five already-converted sessions to produce `sample_data.pkl`, re-walking those sessions' trial lists and re-building the brain-region index, and writing a second 113 MB pickle. Within a session, `np.any(valid)` is evaluated twice in a row on the same array, and the boolean mask `valid_trial_idx = trial_idx[valid]` is recomputed in both branches. `decode_array` is applied to the full-length `classification` and `anno_name` columns, i.e. every unit is decoded even though only the ~25 % `'good'` ones are used.

ii.
```python
full_data, full_summary = build_dataset(session_results)
...
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
...
write_pickle(args.sample_out, sample_data)
```
```python
valid = (trial_idx >= 0) & (trial_idx < n_trials)
if np.any(valid):
    valid_trial_idx = trial_idx[valid]
    valid &= spikes <= trial_stop[valid_trial_idx]

if np.any(valid):
    valid_trial_idx = trial_idx[valid]
```

iii. The sample dataset is intentional: `README.md` describes `sample_data.pkl` as the *"small subset used for quick verification/training"*, and the AI used it to run `train_decoder.py` end-to-end before committing to the full run (steps 130-137). Reusing already-converted sessions rather than re-reading the files is the cheaper of the two ways to produce it.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly small, one item not small:
- **`sample_data.pkl` (113 MB)** is rebuilt and rewritten on **every** run of the script, including the full production run that only needs `converted_data.pkl`. It is not part of the requested deliverable.
- Per-session bookkeeping that never reaches the decoder: `Counter`s for choice/outcome/early/choice-source, `n_behavior_trials`, `n_ephys_trials`, `stim_trial_count`, `tongue_q40`/`q60`, `tongue_bin_fallback_count`, all aggregated into `metadata` and a printed JSON summary.
- `bin_edges_s` and `bin_centers_s` are serialised into metadata as full Python lists although `time_bin_size`, `off_start` and `off_end` already determine them.
- `tongue_data` column 0 (`tongue_x`) and column 2 (`tongue_likelihood`) are read and then never used; `left_idx` in `bin_tongue_y` is computed only to count fallbacks.
- `decode_array` decodes all units' `classification`/`anno_name`, not just the good ones.

ii.
```python
sample_count = min(args.sample_sessions, len(session_results))
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
sample_data["metadata"]["sample_session_count"] = sample_count
write_pickle(args.sample_out, sample_data)
```
```python
"bin_edges_s": BIN_EDGES_S.tolist(),
"bin_centers_s": BIN_CENTERS_S.tolist(),
"choice_source_counts": dict(total_choice_sources),
"outcome_value_counts": dict(total_outcomes),
"early_lick_value_counts": dict(total_early),
"choice_value_counts": dict(total_choice),
"stim_trial_count_total": int(total_stim_trials),
"tongue_bin_fallback_count_total": int(total_tongue_fallback),
```
```python
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
fallback_missing = int(np.sum(right_idx <= left_idx))
```

iii. The extra metadata is deliberate provenance: the AI used `choice_source_counts` and `tongue_bin_fallback_count_total` to quantify and disclose its own imputations in `CONVERSION_NOTES.md`, which is a reasonable trade of a few kB against auditability. The always-on sample pickle is the one genuinely wasteful item — it is a development convenience that was never gated behind a flag before the production run.
