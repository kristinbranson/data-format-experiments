# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `data/sub-<subject_id>/*.nwb`. The AI enumerates every session with a single sorted glob over that layout and processes each file once. Unlike the reference, it does **not** use `pynwb`; it opens each file as raw HDF5 with `h5py` and addresses the NWB groups by hard-coded internal paths (`units`, `intervals/trials`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`). Byte-string columns are decoded manually, and the ragged `spike_times` / `spike_times_index` pair is indexed by hand. All 174 files are opened; 173 are retained.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    classification = decode_bytes_array(units["classification"][()])
    anno_name = decode_bytes_array(units["anno_name"][()])
    ...
    trials = f["intervals/trials"]
    ...
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
    sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
    left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
    right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
```

```python
def ragged_row(data_ds, index_ds, row_idx: int) -> np.ndarray:
    stop = int(index_ds[row_idx])
    start = 0 if row_idx == 0 else int(index_ds[row_idx - 1])
    return np.asarray(data_ds[start:stop], dtype=np.float64)
```

iii. The AI first explored the data with `pynwb` and then deliberately switched to direct HDF5 reads for speed: *"The `pynwb` scan is workable for spot checks but too slow for all-session summaries. I'm switching the bulk statistics pass to direct HDF5 reads so I can iterate quickly on session and unit filters without spending minutes on every tweak."* (trajectory step 38). It verified the field names against the actual NWB schema rather than guessing: *"I'm now inspecting the NWB schema directly to map those same variables and the tongue trace without guessing field names"* (step 18).

## 1-b. How are the data split into subjects?

i. The subject identifier is taken from the **containing directory name**, with the `sub-` prefix stripped (e.g. `sub-440956` -> `'440956'`), rather than from the NWB `subject/subject_id` field. At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list, stored as `int16`. This yields the same 28 subjects with 3-10 sessions each as the reference.

ii.
```python
stats = {
    "session_id": path.stem,
    "subject": path.parent.name.replace("sub-", ""),
    ...
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
```

iii. `CONVERSION_NOTES.md`: *"`subjects` come from the NWB subject directory names."* No further justification is given; the DANDI layout guarantees the folder name is derived from `subject_id`, so the result is identical to reading the field.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. The session identifier is the **filename stem** (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than the NWB `identifier` (`SC015_20190207_120657_s1`). Session order in the output follows the sorted file list, which is chronological within each subject because the filename embeds the acquisition timestamp. 173 of 174 sessions reach the output; the one dropped session is the one with no quality-controlled units (see 2-c). A `--session-limit` option exists for building the 5-session sample file but is not used for the full conversion.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
    session_stats.append(stats)
    if session is not None:
        sessions.append(session)
        if session_limit is not None and len(sessions) >= session_limit:
            break
```

```python
session = {
    "session_id": path.stem,
    ...
```

iii. From the trajectory (step 48) the AI considered and rejected the *data* paper's session-pruning rule: *"only 145 of 174 files pass that strict `>65%` and `>=50` correct left/right control-trial rule. That tells me the decoder conversion should probably follow the analysis code's trial masking and unit QC, but not blindly reuse the original paper's session-pruning rule."* It then checked the surviving count against the methods text: *"Included session count matches the paper-reported 173 behavioral sessions"* (`CONVERSION_NOTES.md`). Note the session identifiers are the DANDI filenames, not the lab session names used in the papers; the AI did not comment on this.

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial, with `go_start_times` taken as one go-cue event per trial row by positional correspondence. The trial-level arrays and the go-cue timestamp array are all truncated to the same length, `units/is_good_trials.shape[1]`, before any further work (see 1-e). No assertion is made that `len(go_times) == len(trials)`; the 1:1 correspondence is assumed by construction of the parallel slices.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])

trials = f["intervals/trials"]
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
auto_water = np.asarray(trials["auto_water"][()], dtype=np.int8)[:ephys_trial_count]
free_water = np.asarray(trials["free_water"][()], dtype=np.int8)[:ephys_trial_count]
...
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. The AI treated the exported trial table as authoritative: *"The NWB files already carry the paper's derived trial table and QC/unit metadata, which is useful because I can avoid re-deriving behavior labels from raw lick streams when the reference analysis also used session-level exported arrays"* (step 24). It explicitly rejected using trial start/stop as hard data boundaries for the neural window: *"I was treating the trial table as the hard data boundary, but the recordings are continuous across the session. I'm removing that artificial trial-start/trial-stop gate"* (step 91).

## 1-e. How are trials filtered based on quality controls?

i. Five filters, applied in order:

1. **Ephys coverage.** All trial-level arrays are truncated to the **first** `units/is_good_trials.shape[1]` rows, on the assumption that the ephys-covered trials are always a leading prefix of the behavioural trials.
2. **Water trials.** Trials with `auto_water != 0` **or** `free_water != 0` are dropped (3,764 trials). The reference drops only `free_water`.
3. **Video coverage.** A trial is dropped if its `[go - 2.5, go + 1.5)` window is not fully inside the side-camera timestamp range (765 trials).
4. **All-zero neural.** After binning, any trial with zero spikes across every good unit and every bin is dropped (127 trials).
5. **Session minimum.** A session is dropped if fewer than 2 trials survive (either check).

Early-lick, `ignore`, and photostimulation trials are deliberately kept. Final total: 88,654 trials over 173 sessions (reference: 90,860).

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
```

```python
keep_mask = (auto_water == 0) & (free_water == 0)
stats["n_trials_dropped_auto_free"] = int(np.sum(~keep_mask))
keep_idx = np.flatnonzero(keep_mask)
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
    ...
if n_trials < 2:
    return None, stats
```

iii. The AI documented the deviation from the papers' "regular trial" mask as task-driven: *"the reference analyses often drop early-lick and ignore trials because they are nuisance cases for motor decoding, but this decoder explicitly wants `early_lick` and `outcome` as targets. I'm treating that as an allowed task-driven deviation"* (step 63). `CONVERSION_NOTES.md` states the same and keeps `auto_water`/`free_water` exclusion because the reference code's regular-trial mask excludes them.

For ephys coverage it diagnosed an all-zero-trial problem and chose `is_good_trials` as the coverage indicator: *"some NWB files contain full-session behavior, but the ephys spike trains only cover an initial block of trials... the units table encodes an `is_good_trials` matrix whose width is the actual ephys-covered trial count for that session. Some sessions cover all behavior trials; others only the first block. I'm slicing the behavior/video trial list to that exact count so spikes and labels live on the same trial set"* (steps 101, 105). It then added the all-zero check as a backstop: *"if an entire trial is zero across every good unit and every 50 ms bin, I'll drop it"* (step 116). It explicitly considered `obs_intervals` (*"I'm checking the recorded observation interval next"*, step 111) but did not use it.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` (with `units/spike_times_index` for the ragged offsets), in session-absolute seconds, restricted to units whose `units/classification == 'good'`. The go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` supply the bin edges. Identical sources to the reference.

ii.
```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
flat_abs_edges = abs_edge_matrix.reshape(-1)
...
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
```

```python
good_unit_idx = np.flatnonzero(classification == "good")
```

iii. Spike times are the only neural representation in the file. The AI verified the scale of the spike times against per-trial validity metadata when it saw all-zero trials (step 97), confirming spike times are session-absolute rather than trial-relative.

## 2-b. How is the `neural` data processed?

i. Per-bin spike counts are obtained with one `np.searchsorted` per unit against the flattened array of absolute bin edges for all trials, then differenced. Counts are divided by the 50 ms bin width to give firing rate in Hz. No smoothing, normalisation, or baseline subtraction. The result is cast to **`float16`** (the reference uses `float32`) and scattered into one `(n_units, 80)` array per trial by a Python loop over trials.

ii.
```python
neural_trials = [np.empty((good_unit_idx.size, n_bins), dtype=np.float16) for _ in range(n_trials)]

for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. `CONVERSION_NOTES.md`: *"Neural activity is stored as firing rate in spikes/s: `rate = spike_count_in_bin / 0.05`"*, matching the reference code's `sliding_histogram(..., rate=True)`. No rationale is given for `float16`; it halves the payload (5.5 GB vs the reference's 11.9 GB) and is lossless here because rates are multiples of 20 Hz and integers below 2048 are exact in `float16`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; no thresholds on any individual quality metric are applied, and `units/unit_quality` is not used. A session with no `'good'` units is dropped outright. This retains 69,453 units over 173 sessions, a mean of 401.46 per session - **numerically identical to the reference** (69,453 units; `nneurons_mean` 401.46). The single dropped session is `sub-440958_ses-20190216T162508`, the same one the reference drops.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
anno_name = decode_bytes_array(units["anno_name"][()])
good_unit_idx = np.flatnonzero(classification == "good")
stats["n_units_good"] = int(good_unit_idx.size)
if good_unit_idx.size == 0:
    return None, stats
```

iii. *"Used the NWB-provided quality-control label `classification == "good"`. This matches the papers' and white paper's description that downstream analyses use units labeled `good` by the QC classifier. No extra firing-rate or waveform filtering was added on top of the NWB `good` label."* (`CONVERSION_NOTES.md`). The AI investigated the 490-unit gap against the methods text and concluded it is a release discrepancy, not a conversion bug: *"the raw NWBs contain 69,453 `classification == "good"` units, while the methods excerpt reports 69,943... the notes will treat the unit-count gap as a release discrepancy rather than a conversion bug"* (step 197). It also considered, and did not adopt, restricting to units with histology assignments (step 56).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes, go cues and camera frames are all on the same session-absolute clock, so no alignment correction is needed. A fixed set of 81 bin edges relative to the go cue is added to each trial's go-cue time to produce an absolute `(n_trials, 81)` edge matrix, which is flattened and used directly as the `searchsorted` query. Identical in substance to the reference.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
...
flat_abs_edges = abs_edge_matrix.reshape(-1)
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
```

iii. *"Alignment event: go cue onset"* with `temporal_alignment_event: "go cue onset"` in the metadata. The AI confirmed the reference code's alignment logic before writing the converter: *"Match the reference alignment logic at go cue"* (step 76).

## 2-e. How is the `neural` data temporally binned/resampled?

i. 80 non-overlapping 50 ms bins spanning `[-2.5, +1.5)` s relative to the go cue. The grid is built once at module level as 81 edges and reused for every trial and session, so every trial has exactly 80 timepoints. No resampling or rebinning of an intermediate representation; spikes are binned once, directly at the target resolution. `metadata['time_bin_size'] = 50.0` ms, `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05

def fixed_bin_edges(start_time: float, end_time: float, width: float) -> np.ndarray:
    """Create exact-width bins covering [start_time, end_time)."""
    n_bins = int(round((end_time - start_time) / width))
    return start_time + np.arange(n_bins + 1, dtype=np.float64) * width
```

iii. The AI initially implemented the reference code's convention of bin *centers* spanning the requested endpoints and abandoned it: *"The sample export exposed a real issue: the inclusive end-center convention forces an extra 25 ms of coverage and is dropping valid trials. I'm switching the converter to literal 50 ms bins over `[-2.5, 1.5)` so the window matches the task specification and doesn't discard trials just because they end at the requested boundary"* (step 86). `BIN_STRIDE` is declared but never used - a vestige of the earlier sliding-window implementation.

## 3-a. What variables in the raw data is `input` *time_from_tone_onset* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone onsets) together with the trial's go-cue time and `trials/start_time`. The tone for a trial is the **last** `sample_start_times` entry falling in `[trial_start, go]`. If no such entry exists, a fallback of `go - 1.85` s is used (the nominal 0.65 s sample + 1.2 s delay); the fallback was never triggered on the full dataset (0 trials).

ii.
```python
def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

```python
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
if tone_used_fallback:
    stats["tone_fallback_trials"] += 1
```

iii. *"The tone onset was taken as the last `sample_start_times` timestamp between trial start and go cue. This follows the task structure in which multiple sample starts can occur due to replay after early licking."* (`CONVERSION_NOTES.md`). Same reasoning as the reference; the AI additionally bounds the search below by trial start and instruments the fallback so it is auditable.

## 3-b. What processing is involved in computing `input` *time_from_tone_onset*?

i. The tone time is expressed relative to the go cue, and each bin's value is its center minus that offset - i.e. `bin_center + (go - tone)`, seconds since tone onset, so zero corresponds to tone onset. Stored as `float32`, row 0 of the `(2, 80)` input array. Algebraically identical to the reference. The realised range is `[-1.5, 11.9]` vs the reference's `[-1.525, 11.894]` (the small offset is the differing bin-center convention).

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
...
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. *"For each neural bin, the value is `bin_center_relative_to_go - tone_onset_relative_to_go`. That makes zero correspond to tone onset."* (`CONVERSION_NOTES.md`). No further processing was judged necessary.

## 3-c. How is `input` *time_from_tone_onset* aligned with the neural data?

i. It is computed on exactly the same grid that defines the neural bins: `BIN_CENTERS` is derived from `BIN_EDGES`, and `BIN_EDGES` is what is added to each go cue to bin the spikes. Bin `k` of the input therefore covers the same interval as bin `k` of the firing rates by construction.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]   # neural
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)   # input
```

iii. Not separately justified; it follows from using a single module-level grid for every stream.

## 4-a. What variables in the raw data is `input` *photostim* derived from?

i. From `trials/photostim_onset` and `trials/photostim_duration` (both stored as strings, with `'N/A'` on unstimulated trials), plus `trials/photostim_power` as an extra guard, with `trials/start_time` and the go-cue time used to move them onto the go-cue axis. The reference uses onset and duration only. Empirically the power guard is a no-op: across all 174 sessions, all 18,588 trials with a non-`'N/A'` onset also have `photostim_power > 0`, and no trial has power without an onset.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
    stats["stim_trials_kept"] += 1
else:
    on_rel = np.nan
    off_rel = np.nan
```

iii. *"Derived from `photostim_onset`, `photostim_duration`, and `photostim_power`. Stimulation times are converted from trial-start reference into go-cue reference, matching the reference code pattern."* (`CONVERSION_NOTES.md`). The power condition is a defensive guard against zero-power/sham entries. The AI sanity-checked the resulting cohort size against the methods text: *"Kept photostimulation trials represent about 20% of retained full-dataset trials, broadly consistent with the methods text stating that photoinhibition was deployed on a subset of about 25% of trials"*, and at step 202 chose the photostim cohort size as a sanity check precisely because *"that is directly comparable to the methods excerpt and is less ambiguous than trial counts"*.

## 4-b. What processing is involved in computing `input` *photostim*?

i. A binary, time-varying series: a bin is 1 when its center falls in `[onset, offset)` in go-cue-relative time, 0 otherwise. Unstimulated trials take an explicit all-zeros branch rather than relying on NaN comparison. Stored as `float32` in row 1 of the input array. Substantively identical to the reference.

ii.
```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. *"Binary, time-varying input... Bins are marked 1 when the bin center lies within the stimulation interval."* (`CONVERSION_NOTES.md`). This satisfies the instruction that photostimulation be represented as "whether photostimulation is on at every time point".

## 4-c. How is `input` *photostim* aligned with the neural data?

i. The stored onset is relative to trial start, so it is re-expressed relative to the go cue by subtracting `go - trial_start`; the offset is onset plus duration. Both are then compared against `BIN_CENTERS`, the same grid used for the firing rates, so no interpolation or offset correction is needed.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
```

iii. Not separately justified beyond *"Stimulation times are converted from trial-start reference into go-cue reference, matching the reference code pattern."*

## 5-a. What variables in the raw data is `output` *choice* derived from?

i. Choice is **measured from the lick event streams**, not derived from the trial table: `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`. For each trial the first lick of either side in `[go, trial_stop]` determines the side (`left = 0`, `right = 1`). When there is no post-go lick - i.e. `ignore` trials - the code falls back to `trials/trial_instruction`, the *instructed* side. The fallback fired on 13,064 of 88,654 retained trials (14.7%).

The reference instead derives choice from `trial_instruction x outcome`. I verified on session `sub-440956_ses-20190207T120657` that the two derivations agree on **every** non-ignore trial (296/296, 0 disagreements) and that no `ignore` trial has a post-go lick, so the lick-stream source is a valid and equivalent measurement; the divergence is entirely in the ignore-trial handling (5-b).

ii.
```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")

    left_time = left_licks[left_idx] if left_idx < len(left_licks) and left_licks[left_idx] <= stop_time else np.nan
    right_time = right_licks[right_idx] if right_idx < len(right_licks) and right_licks[right_idx] <= stop_time else np.nan

    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    if np.isfinite(left_time):
        return 0, False
    if np.isfinite(right_time):
        return 1, False
    return (0 if instruction == "left" else 1), True
```

iii. The AI chose the lick streams explicitly to avoid substituting the instructed side: *"Choice is the one ambiguous target in the task: the NWB trial table has `trial_instruction` and `outcome`, but not an explicit per-trial reported side. I'm checking the lick-event streams now so I can label choice from behavior itself rather than silently substituting the instructed side"* (step 69). `CONVERSION_NOTES.md`: *"Determined from the first post-go lick side within the trial."*

## 5-b. What processing is involved in computing `output` *choice*?

i. The per-trial code (0 = left, 1 = right) is repeated across all 80 bins and written to row 0 of the `(4, 80)` `int8` output array. `output_values[0] = ['left', 'right']` - **only two classes**. There is no `'no lick'` category: the 13,064 `ignore` trials are relabelled as the side the tone instructed. The realised distribution is `{left 0.496, right 0.504}` against the reference's `{left 0.429, right 0.422, no lick 0.149}`.

ii.
```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

```python
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks, right_licks, go_abs, float(trial_stop[trial_idx]), trial_instruction[trial_idx],
)
if choice_used_fallback:
    stats["choice_fallback_trials"] += 1
```

```python
output_trials.append(
    np.vstack(
        [
            np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
            ...
```

iii. The AI believed a binary label was mandated by the output format: *"Ignore trials have no post-go lick by definition, so they need a fallback to satisfy the required binary label format. For those trials: fallback choice = instructed side from `trial_instruction`"*, and in the metadata: *"ignore trials fall back to the instructed side because the target format requires a binary choice label"*. It counted and reported the fallback (13,064 trials) so the choice is auditable. The instruction sheet, however, specifies three values - *"Lick direction choice (left, right, no lick, per-trial)"* - so the premise that a binary label is required is not supported by the task specification.

## 6-a. What variables in the raw data is `output` *outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already stores the strings `'hit'`, `'miss'`, `'ignore'` - exactly the three categories the instructions ask for. Same source as the reference. I confirmed across all sessions that the column contains only these three values.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

iii. No derivation is needed; the AI verified the field contents against the trial table when mapping the NWB schema (step 18-24) and raises on an unexpected value.

## 6-b. What processing is involved in computing `output` *outcome*?

i. Mapped to `ignore = 0`, `miss = 1`, `hit = 2` by an explicit if/elif chain that raises `ValueError` on any unexpected string, then repeated across all 80 bins into row 1 of the output array. Identical mapping and realised distribution to the reference (`{ignore 0.149, miss 0.167, hit 0.683}` vs reference `{0.149, 0.166, 0.684}`).

ii.
```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
    stats["ignore_trials_kept"] += 1
elif outcome[trial_idx] == "miss":
    outcome_code = 1
    stats["miss_trials_kept"] += 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
    stats["hit_trials_kept"] += 1
else:
    raise ValueError(f"Unexpected outcome {outcome[trial_idx]!r} in {path.name}")
```

iii. *"Mapping: `ignore = 0`, `miss = 1`, `hit = 2`"* (`CONVERSION_NOTES.md`), following the order given in the instructions. Outcome is one value per trial, so it is broadcast across bins like the other per-trial outputs.

## 7-a. What variables in the raw data is `output` *early_lick* derived from?

i. Directly from the `early_lick` column of the trials table. I confirmed across all 174 sessions that the column takes exactly two values, `'no early'` (84,185) and `'early'` (10,805). Same source as the reference.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

iii. The flag is stored explicitly per trial, so no derivation from the lick streams is needed. The AI identified `early_lick` early as part of the reference code's regular-trial mask (step 18) and kept it as a decoder target rather than a filter.

## 7-b. What processing is involved in computing `output` *early_lick*?

i. Mapped to `1` if the string is exactly `'early'`, otherwise `0`, then repeated across all 80 bins into row 2 of the output array. Realised distribution `{no 0.884, yes 0.116}`, matching the reference's `{0.8846, 0.1154}`. The reference uses an explicit dictionary that would raise on an unknown value; the AI's `else 0` silently absorbs one - benign here since the vocabulary is verified binary, but less safe as a general pattern.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
early_codes.append(early_code)
stats["early_trials_kept"] += early_code
```

iii. *"Mapping: `no = 0`, `yes = 1`"* (`CONVERSION_NOTES.md`), following the instruction's ordering ("no, yes").

## 8-a. What variables in the raw data is `output` *tongue_y_position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using **column 1 (`tongue_y`) only**, with its `timestamps`. Column 0 (`tongue_x`) and **column 2 (`tongue_likelihood`) are loaded but never used**. The reference uses column 2 to decide whether the tongue is visible in each frame.

This matters: on `sub-440956_ses-20190207T120657`, 89.4% of frames have `tongue_likelihood < 0.5`, and the tracker still emits a y-coordinate on those frames. Using every frame therefore mixes real protrusions with retracted-tongue tracker output, and shifts the percentile edges (raw-frame 40th/60th = 282.5/296.5 vs visible-only 273.7/288.5).

ii.
```python
tongue_data = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

iii. *"Source: side-camera tongue tracking `Camera0_side_TongueTracking`. Used the `tongue_y` coordinate only."* (`CONVERSION_NOTES.md`). No justification is offered for discarding the likelihood channel, and the trajectory contains no discussion of tongue visibility at all - the only tongue-related planning statement is *"build 50 ms neural bins and photostim/tongue labels on the same grid"* (step 76). The series' own `description` attribute names the three channels (`('tongue_x', 'tongue_y', 'tongue_likelihood')`), so the likelihood channel was discoverable.

## 8-b. What processing is involved in computing `output` *tongue_y_position*?

i. Three steps, no visibility masking:
1. Session-level percentiles: the 40th and 60th percentiles of the **raw, unfiltered** `tongue_y` over the whole session.
2. Per bin, the value is the **last camera frame whose timestamp falls in that bin** (not the mean of the frames in the bin, as in the reference).
3. If a bin contains no frame, the most recent frame at or before the bin end is carried forward (index-clamped at the array ends).

The carry-forward branch is rarely exercised in practice - on `sub-440956_ses-20190207T120657` zero of 29,440 in-window bins were empty - because the AI separately drops trials whose window is not inside the camera span. The dominant effect is step 1/2 operating on unfiltered frames: ~79% of in-window bins contain no visible-tongue frame in that session, and all of them still receive a real 0/1/2 class.

ii.
```python
def trial_tongue_categories(video_timestamps, tongue_y, abs_edges, p40, p60):
    starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
    ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
    values = np.empty(len(starts), dtype=np.float64)

    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]
```

```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

iii. *"For each neural bin, took the last video frame within that bin. If a bin had no new frame, used the most recent available frame at or before the bin end... Percentiles were computed over the full session's raw `tongue_y` values."* (`CONVERSION_NOTES.md`). The per-session scope follows the instruction ("percentile of y-position over the session"); the "last frame in bin" rule is a nearest-sample resampling choice and is not further justified. There is no discussion of why the raw rather than visibility-filtered distribution was used.

## 8-c. How is `output` *tongue_y_position* thresholded into categories?

i. Three classes only: `0` for `y < p40`, `1` for `p40 <= y <= p60`, `2` for `y > p60`, with `p40`/`p60` the per-session percentiles. The boundary convention matches the instructions for the three visible classes. However, the instruction's **fourth class, `3: not visible`, is not implemented at all** - `output_values[3] = ['lt_40pct', 'p40_to_p60', 'gt_60pct']` has three entries and the realised range of the output is `[0, 2]`.

Realised distribution: `{lt_40pct 0.384, p40_to_p60 0.190, gt_60pct 0.426}` against the reference's `{0.097, 0.051, 0.103, not visible 0.750}`. The AI's distribution is essentially the 40/20/40 split you get by construction from percentiles of an unfiltered distribution; the reference's shows that three quarters of bins have no visible tongue.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
return cats
```

```python
"output_values": [
    ...
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

iii. `CONVERSION_NOTES.md` lists only the three percentile classes: *"Discretization is session-specific: `0`: `< 40th percentile`; `1`: `40th to 60th percentile`; `2`: `> 60th percentile`."* The instruction's `3: not visible` class is not mentioned anywhere in the notes, the code, or the trajectory - it appears to have been overlooked rather than deliberately rejected.

## 8-d. How is `output` *tongue_y_position* aligned with the neural data?

i. Using the same absolute bin-edge matrix as the firing rates: `abs_edge_matrix[row_idx]` is passed straight into `trial_tongue_categories`, and frames are assigned to bins by `searchsorted` on the camera timestamps at those edges. Camera timestamps share the session-absolute clock with spikes and events, so no interpolation or offset correction is applied. Bin `k` of the tongue output therefore covers the same interval as bin `k` of the firing rates - with the caveat that the empty-bin fallback can import a value from *outside* bin `k`.

Separately, trials whose window is not fully covered by the camera's global timestamp span are dropped entirely (765 trials), rather than having their uncovered bins labelled as missing.

ii.
```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
...
tongue_cat = trial_tongue_categories(
    video_timestamps=video_timestamps,
    tongue_y=tongue_y,
    abs_edges=abs_edge_matrix[row_idx],
    p40=tongue_p40,
    p60=tongue_p60,
)
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

iii. The AI's stated goal was to *"build 50 ms neural bins and photostim/tongue labels on the same grid"* (step 76), and it removed an earlier trial-start/trial-stop gate once it established that the streams are continuous session-wide: *"the recordings are continuous across the session. I'm removing that artificial trial-start/trial-stop gate and only checking whether the continuous video stream actually covers the requested aligned window"* (step 91).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, split between *exclusion*, *explicit reporting*, and *imputation*:

- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `decode_bytes_array` renders non-string entries via `str(value)` -> `'nan'`, so no unit matches `'good'` and the session is returned as `None` and skipped. Same outcome as the reference.
- **Trials outside ephys coverage**: truncated away by `is_good_trials.shape[1]` (see 1-e).
- **Trials with no spikes at all**: dropped by the post-binning all-zero check (127 trials).
- **Trials outside camera coverage**: dropped (765 trials).
- **Missing tone onset**: imputed as `go - 1.85` s and counted (`tone_fallback_trials`; 0 on the full dataset).
- **No post-go lick (ignore trials)**: imputed as the instructed side and counted (`choice_fallback_trials`; 13,064 trials).
- **Bin with no camera frame**: imputed by carrying the last preceding frame's y-value forward.

Every exclusion and every fallback is counted into a per-session `stats` dict and aggregated into the printed summary, so the amount of imputation is auditable from the conversion log.

ii.
```python
def decode_bytes_array(arr) -> np.ndarray:
    out = []
    for value in arr:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode())
        elif hasattr(value, "decode"):
            out.append(value.decode())
        else:
            out.append(str(value))
    return np.asarray(out)
```

```python
def parse_optional_float_array(arr) -> np.ndarray:
    ...
        if value in {"N/A", "nan", "NaN", ""}:
            out[i] = np.nan
```

```python
if good_unit_idx.size == 0:
    return None, stats
...
if len(valid_trials) < 2:
    return None, stats
...
if n_trials < 2:
    return None, stats
```

iii. The AI's stated principle was that every fallback should be visible rather than silent: *"it will explicitly count and report every fallback decision like `choice` on ignore trials or missing last-sample events so those choices are auditable in the notes"* (step 77). For all-zero trials: *"That matches the spirit of the reference data export much better than keeping obviously non-informative trials just because they survived the metadata filter"* (step 116). The imputing fallbacks (choice, empty tongue bin) are the opposite of the reference's approach, which represents legitimately-absent measurements as an explicit category rather than filling them in.

## 10-a. What are the most time-consuming steps of the code?

i. The code is not profiled or timed anywhere, and `CONVERSION_NOTES.md` contains no efficiency discussion. From the structure, the dominant costs per session are:

1. **Per-unit ragged HDF5 reads.** `ragged_row` issues one HDF5 slice read per good unit - ~69,453 separate reads over the dataset - plus the `searchsorted` of each unit's spikes against all 81 x n_trials edges.
2. **The unit x trial scatter loop.** For each unit, an inner Python loop copies that unit's rates into each trial's array: ~69,453 x ~500 = ~35 million Python-level slice assignments over the full dataset. This is pure overhead the reference does not pay.
3. **Bulk array reads.** `Camera0_side_TongueTracking/data` (~680k x 3 doubles) and the lick/event timestamp arrays are read whole per session.
4. **The per-trial tongue loop**, with an inner 80-iteration Python loop: ~7.1 million iterations over 88,654 trials.
5. **Pickling** the ~5.5 GB result (smaller than the reference's 11.9 GB because of `float16`).

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. No justification is recorded. The AI's only performance-related decision in the trajectory is the choice of `h5py` over `pynwb` for speed (step 38), and its runtime concerns were about the *decoder* rather than the converter (*"a literal all-trials, all-good-units conversion across 173 sessions is large enough that storage and decoder runtime matter"*, step 72), which motivated the `float16` storage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five, of which two are on the hot path:

1. **`for trial_row in range(n_trials)` inside the per-unit loop** (~35M iterations dataset-wide). Fully vectorizable: build one `(n_units, n_trials, n_bins)` array and slice per trial at the end, exactly as the reference does.
2. **`for i, (start_idx, end_idx) in enumerate(zip(starts, ends))` in `trial_tongue_categories`** (~7.1M iterations). Vectorizable with a per-frame bin index and `np.bincount`/`np.add.reduceat`.
3. **`for row_idx, trial_idx in enumerate(valid_trials)`** building the inputs and outputs one trial at a time. The reference computes `time_from_tone`, `photostim`, and all per-trial outputs as whole `(n_trials, n_bins)` arrays with no Python loop.
4. **`for trial_idx in keep_idx`**, the per-trial labelling pass (`first_post_go_choice`, `last_sample_before_go`, the outcome if/elif chain). All are `searchsorted`/`np.where` operations that vectorize across trials.
5. **`decode_bytes_array` / `parse_optional_float_array`**, element-by-element Python loops over every string column of every session; `np.char.decode` handles the common case.

The genuinely irreducible loop - one `searchsorted` per unit, because the ragged spike storage gives each unit a different sorted array - is the same one the reference keeps and justifies.

ii.
```python
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

```python
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]
```

```python
for trial_idx in keep_idx:
    go_abs = float(go_times[trial_idx])
    abs_edges = go_abs + BIN_EDGES
    ...
```

iii. Not justified anywhere; the AI never discussed vectorization. The per-trial loops do buy readability and make the per-trial fallback counters straightforward, which is consistent with its stated goal of making every fallback auditable (step 77), but loop 1 has no such benefit - it is a pure transpose done element-wise.

## 10-c. What processing does the code repeat multiple times?

i. Within a single conversion run, the per-session work is a single pass: each file is opened once, each array read once, and the tongue percentiles - being per-session - are computed inside that same pass with no second pass over the data. Three smaller repetitions exist:

1. **`summarize_conversion` re-walks every trial of every session** (88,654 trials) after assembly purely to rebuild the choice/outcome/early histograms, which were already tallied into `stats` during conversion.
2. **`np.searchsorted(video_timestamps, ...)` is re-issued per trial** against the full ~680k-element timestamp array (cheap - `O(log n)` per edge - but repeated 88,654 times rather than done once on a global bin index).
3. **The all-zero check `[np.any(trial != 0) for trial in neural_trials]`** re-reads the whole neural payload of the session after it has just been written.

At the workflow level the full conversion was executed twice - once interactively and once with stdout redirected to produce `conversion_full_out.txt` (steps 130, 168) - and the decoder likewise; that is an artefact of the deliverable list, not of the code.

ii.
```python
    for stats in included_stats:
        stim_trials += int(stats["stim_trials_kept"])
    ...
    for session_outputs in data["output"]:
        for trial_output in session_outputs:
            choice_hist[int(trial_output[0, 0])] += 1
            outcome_hist[int(trial_output[1, 0])] += 1
            early_hist[int(trial_output[2, 0])] += 1
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. Not discussed. The AI's emphasis was on producing an auditable summary - *"I want one stronger sanity check before I freeze the notes"* (step 195) - which is what the second histogram pass serves; it double-counts work already done but guarantees the reported numbers describe the object actually pickled rather than the intermediate state.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none expensive but all dead weight:

- **`session_output_counts` is dead code**: a `Counter` is created and incremented on every trial, then never read.
- **`tongue_data[:, 0]` (`tongue_x`) and `tongue_data[:, 2]` (`tongue_likelihood`) are loaded and discarded.** The likelihood channel in particular is read into memory and thrown away, when using it is exactly what the reference (and the instruction's `not visible` class) requires - this is the one discard that is also a correctness problem.
- **`stim_power` is parsed for every trial** to evaluate a condition that never changes the result (verified: all 18,588 onset-bearing trials have power > 0).
- **Per-session dict keys `trial_indices_source`, `tongue_percentiles`, `n_units_good`, `n_trials_kept`** are computed and stored, then dropped at assembly - they never reach the pickle.
- **The `stats` counter `n_trials_dropped_tone`** is initialised and never written to or read.
- **`BIN_STRIDE` is defined and never used** (vestige of the abandoned sliding-window binning).
- **`metadata['bin_centers_sec']`** is stored in the pickle but is not part of the target format and is unused by the decoder.
- **Full `anno_name` strings are kept** rather than collapsed to coarse areas, giving 293 `brain_regions` entries against the reference's coarse labels; this is a deliberate choice, not waste, but it does mean the region grouping used in the papers is not reproduced.

ii.
```python
session_output_counts = Counter()
...
            session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1
```

```python
BIN_STRIDE = 0.05
```

```python
session = {
    ...
    "trial_indices_source": valid_trials,
    "tongue_percentiles": (float(tongue_p40), float(tongue_p60)),
}
```

iii. The region choice is the only one justified: *"I kept the exact annotation names instead of collapsing them to coarse areas, because the NWB annotations already encode the per-neuron region identity used in the released dataset"* (`CONVERSION_NOTES.md`). The remaining items are unremarked residue from the AI's iterative development - it patched the converter ten times (steps 78-119) and did not do a cleanup pass before finalising.
