# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all `sub-*/*.nwb` files under `/app/data`, prefilters them to sessions with at least one `classification == b"good"` unit, and then reopens each kept NWB file with `h5py` for conversion. Within each file it reads raw HDF5 datasets directly from `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries`.

ii.
```python
def discover_usable_files() -> tuple[list[str], list[dict]]:
    files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
    usable, excluded = [], []
    for path in files:
        with h5py.File(path, "r") as nwb:
            classification = nwb["units/classification"][()]
            n_good = int(np.count_nonzero(classification == b"good"))
            if n_good:
                usable.append(path)
```

```python
with h5py.File(path, "r") as nwb:
    classification = nwb["units/classification"][()]
    trials = nwb["intervals/trials"]
    go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
    tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI justified direct `h5py` access as a speed/memory decision: it wanted to stream one session at a time, avoid materializing irrelevant NWB objects, and only load the fields needed for conversion.

## 1-b. How are the data split into subjects?

i. The AI uses the numeric subject ID encoded in the filename prefix, e.g. `sub-440956`, rather than reading `nwb.subject.subject_id`. It builds `subjects` as the sorted unique subject strings from the kept file paths and assigns each session with `subject_idx`.

ii.
```python
def subject_from_path(path: str) -> str:
    return Path(path).name.split("_")[0].removeprefix("sub-")
```

```python
subjects = sorted({subject_from_path(path) for path in files})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_lookup[result.subject])
```

iii. In Step 2 and Step 5 notes, the AI treated the path subject ID as authoritative because the dataset is organized one subject per `sub-<id>/` directory and those IDs matched the NWB metadata it inspected during exploration.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session ordering follows the sorted file list, and the session identifier stored in metadata is derived from the filename stem before `_behavior`.

ii.
```python
def session_id_from_path(path: str) -> str:
    return Path(path).name.split("_behavior")[0]
```

```python
all_files, excluded = discover_usable_files()
files = all_files[:2] if args.sample else all_files
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

iii. In Step 2 and Step 4 notes, the AI documented that the dandiset ships one NWB per recording session, so it used file boundaries as session boundaries and did not attempt any further grouping.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trial table, but it does not use every table row directly. It maps the ephys-backed observed trial block via `units/obs_intervals`, checks those rows are contiguous in the trial table, and then indexes trial-level arrays by those mapped rows.

ii.
```python
trials = nwb["intervals/trials"]
trial_starts_table = np.asarray(trials["start_time"], dtype=np.float64)
obs_intervals = np.asarray(nwb["units/obs_intervals"][obs_lo:obs_hi], dtype=np.float64)
ephys_trial_idx = np.searchsorted(trial_starts_table, obs_intervals[:, 0])
if np.any(np.diff(ephys_trial_idx) != 1):
    raise ValueError(f"{session_id}: ephys-backed trial rows are not contiguous")
```

```python
candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
...
go_times = go_all[selected_trial_idx]
trial_starts = trial_starts_all[selected_trial_idx]
```

iii. In Step 4 and Step 5 notes, the AI justified this as necessary because some NWB sessions have behavioral trial-table rows outside the ephys recording period; it therefore mapped trial rows through `obs_intervals` before doing any downstream alignment.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials in three stages: it removes trial-table rows outside the ephys-backed `obs_intervals` block, removes any ephys-backed trial where at least one retained good unit is marked false in `units/is_good_trials`, removes `auto_water` or `free_water` trials, and finally removes any retained trial whose neural array is all zero over the full 4 s window. Sessions with fewer than 2 surviving trials raise an error.

ii.
```python
good_trial_matrix = np.asarray(nwb["units/is_good_trials"][unit_indices, :], dtype=bool)
valid_prefix = np.all(good_trial_matrix, axis=0)
candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
```

```python
auto_water = np.asarray(trials["auto_water"])[candidate_trial_idx] != 0
free_water = np.asarray(trials["free_water"])[candidate_trial_idx] != 0
water_trial = auto_water | free_water
selected_trial_idx = candidate_trial_idx[~water_trial]
if selected_trial_idx.size < 2:
    raise ValueError(f"{session_id}: fewer than two valid non-water trials")
```

```python
zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
if n_zero_spike_trials:
    keep = ~zero_spike_trials
    neural_rates = neural_rates[keep]
    inputs = inputs[keep]
    outputs = outputs[keep]
```

iii. In Step 4 and Step 5 notes, the AI explicitly argued for keeping early/ignore/stimulation trials because they are required decoder targets/inputs, but for dropping `auto_water`/`free_water` and all-unit-invalid trials because it viewed those as altered or unusable trials rather than task conditions to decode.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` and `units/spike_times_index`, restricted to units with `classification == b"good"`, with `go_start_times` used to place the per-trial analysis window.

ii.
```python
classification = nwb["units/classification"][()]
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
...
go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
...
neural_rates = bin_selected_units(
    nwb["units/spike_times"], np.asarray(nwb["units/spike_times_index"]), unit_indices, go_times
)
```

iii. In Step 1, Step 4, and Step 5 notes, the AI described this dataset as extracellular ephys with spike times as the native neural signal, and it chose the classifier-good subset as the curated unit population.

## 2-b. How is the `neural` data processed?

i. The AI bins each good unit’s spike times into non-overlapping 50 ms bins spanning `[-2.5, 1.5)` around the go cue and converts counts to firing rates in Hz. It uses `searchsorted` to assign spikes to trials, `floor` to assign spikes to bins, `np.bincount` to accumulate counts, and divides by `BIN_SIZE_S`.

ii.
```python
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
rel = spikes - go_times[trial_idx]
inside = (rel >= OFF_START) & (rel < OFF_END)
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
```

iii. In Step 5 and Step 6 notes, the AI justified this as preserving the reference code’s half-open binning and Hz scaling while swapping in the decoder task’s required 50 ms non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only `classification == b"good"` units and additionally requires their `anno_name` fields to be nonempty and not `"nan"`. If a session has no good units, or if any retained unit lacks a usable region annotation, the session is excluded/raises.

ii.
```python
classification = nwb["units/classification"][()]
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
region_names = [decode_text(x).strip() for x in nwb["units/anno_name"][unit_indices]]
if any((not x) or x.lower() == "nan" for x in region_names):
    raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")
```

iii. In Step 1 and Step 4 notes, the AI said it wanted to follow the QC classifier used in the paper and also ensure that retained curated units have valid CCF annotations so `brain_region_idx` can be filled without heuristics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. For each retained trial it computes absolute bin centers and windows from `go_start_times`, then bins spikes relative to that trial’s go cue.

ii.
```python
go_times = go_all[selected_trial_idx]
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
```

```python
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
...
rel = spikes - go_times[trial_idx]
inside = (rel >= OFF_START) & (rel < OFF_END)
```

iii. In Step 3 and Step 5 notes, the AI explicitly adopted go-cue alignment because the instructions require it and because all NWB timestamps share the same session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping bins of width 50 ms over the interval `[-2.5, 1.5)` relative to go cue. There is no additional smoothing or rebinning after that.

ii.
```python
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = BIN_CENTERS.size
```

iii. In Step 3 and Step 5 notes, the AI explicitly treated the decoder instructions as overriding the paper’s original 40 ms / 3.4 ms movement-analysis grid.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times/timestamps` and the selected trial starts/go cues. For each trial it picks the first `sample_start_times` event that falls within that behavioral trial, before the go cue.

ii.
```python
sample_starts = np.asarray(nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"])
first_sample_idx = np.searchsorted(sample_starts, trial_starts, side="left")
tone_onsets = sample_starts[first_sample_idx]
if np.any(tone_onsets >= go_times):
    raise ValueError(f"{session_id}: first sample-start not before go")
```

iii. In Step 4 and Step 5 notes, the AI justified “first sample start within the trial” as the literal “tone onset,” arguing that repeated sample/delay replays after early licking should not redefine the initial tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes the global time of every decoder bin center, subtracts the selected tone onset for that trial, and stores the resulting continuous elapsed time in seconds as a `(n_trials, n_timepoints)` input stream.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

```python
inputs = np.stack((elapsed_from_tone, stim_on), axis=1).astype(np.float32)
```

iii. In Step 5 notes, the AI described this as a continuous, time-varying decoder input rather than a binary onset indicator, because the user instructions explicitly asked for “Time from tone onset in seconds.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone-time input is evaluated on exactly the same 80 bin centers used to represent neural activity. The AI first computes bin centers from each trial’s go cue, then subtracts tone onset from those same center timestamps.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

iii. In Step 5 notes, the AI said all streams should share the single go-cue-centered decoder grid so that neural, task, and tongue variables are aligned sample-for-sample.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, not from the trial-table `photostim_onset` and `photostim_duration` columns.

ii.
```python
stim_starts = np.asarray(nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"])
stim_stops = np.asarray(nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"])
stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

iii. In Step 4 and Step 5 notes, the AI explicitly chose the global event streams because they already live on the same clock as spikes and video, and because spot checks showed the start/stop events matched the intended late-delay stimulation period.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts photostimulation into a binary time series sampled at the decoder bin centers. Each bin is 1 if its global center time lies in a half-open interval `[stim_start, stim_stop)`, else 0.

ii.
```python
def photostim_state(centers_global, starts, stops):
    flat = centers_global.ravel()
    interval_idx = np.searchsorted(starts, flat, side="right") - 1
    safe_idx = np.clip(interval_idx, 0, starts.size - 1)
    on = (interval_idx >= 0) & (flat >= starts[safe_idx]) & (flat < stops[safe_idx])
    return on.reshape(centers_global.shape).astype(np.float32)
```

iii. In Step 5 notes, the AI justified a time-varying binary state rather than a per-trial flag because the decoder input specification says photostimulation should indicate whether the light is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by evaluating the start/stop intervals on the same absolute bin-center times used for the neural bins. No separate resampling step is applied.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

iii. In Step 5 notes, the AI’s stated rationale was that spikes, behavioral events, and video all share the NWB session clock, so alignment reduces to comparing timestamps on the same axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not use lick event timestamps to define choice. It derives choice from the trial-table `trial_instruction` and `outcome` fields: hit means the instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
instructions = np.asarray(trials["trial_instruction"])[selected_trial_idx]
```

```python
def map_choice(instructions, outcomes):
    ...
    if outcome == "ignore":
        result[i] = 2
    elif outcome == "hit":
        result[i] = 0 if instruction == "left" else 1
    elif outcome == "miss":
        result[i] = 1 if instruction == "left" else 0
```

iii. In Step 4 notes, the AI justified this as the task-authoritative mapping and noted that it largely matched first response-period lick direction in its spot checks.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as `0 = left`, `1 = right`, `2 = no lick`, and then repeated across all 80 time bins for each trial.

ii.
```python
outputs = np.empty((selected_trial_idx.size, 4, N_TIME), dtype=np.int8)
outputs[:, 0, :] = map_choice(instructions, outcomes)[:, None]
```

```python
"output_values": [
    ["left", "right", "no lick"], ["ignore", "miss", "hit"], ["no", "yes"],
    ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
],
```

iii. In Step 5 notes, the AI said it used a dense `(4, 80)` output tensor per trial, so per-trial categorical outputs are broadcast across time to share the same representation as the time-varying tongue output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trial-table `outcome` column for the retained trial rows.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
```

iii. In Step 2 and Step 5 notes, the AI treated this field as already carrying the exact categories requested by the decoder task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the three outcome strings to integer codes `ignore -> 0`, `miss -> 1`, `hit -> 2`, and repeats the resulting label across time bins.

ii.
```python
def map_outcome(outcomes):
    code = {"ignore": 0, "miss": 1, "hit": 2}
    return np.asarray([code[decode_text(x)] for x in outcomes], dtype=np.int8)
```

```python
outputs[:, 1, :] = map_outcome(outcomes)[:, None]
```

iii. In Step 5 notes, the AI justified this as a direct categorical mapping that preserves all three task outcomes, including ignore trials.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the NWB trial-table `early_lick` column for the retained trial rows.

ii.
```python
early = np.asarray(trials["early_lick"])[selected_trial_idx]
```

iii. In Step 4 and Step 5 notes, the AI explicitly decided to preserve early-lick trials rather than applying the paper’s regular-trial exclusion, because early lick is itself a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `"no early"` to `0` and `"early"` to `1`, then repeats that categorical label across all time bins.

ii.
```python
def map_early(early):
    code = {"no early": 0, "early": 1}
    return np.asarray([code[decode_text(x)] for x in early], dtype=np.int8)
```

```python
outputs[:, 2, :] = map_early(early)[:, None]
```

iii. In Step 5 notes, the AI justified this the same way as choice and outcome: per-trial outputs are tiled across time so all outputs share one array shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The tongue output comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the `timestamps` array, the second data column as tongue `y`, and the third column as DeepLabCut likelihood/visibility.

ii.
```python
tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tracking_ts = np.asarray(tracking["timestamps"], dtype=np.float64)
tracking_data = np.asarray(tracking["data"], dtype=np.float64)
```

```python
xy = np.asarray(data[:, :2], dtype=np.float64).copy()
likelihood = np.asarray(data[:, 2], dtype=np.float64)
...
clean_y = xy[:, 1]
```

iii. In Step 2 and Step 5 notes, the AI identified this as the relevant side-camera tongue tracking stream and emphasized that the likelihood column, not NaNs in the coordinates, determines visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first masks frames by a high-confidence visibility rule (`likelihood >= 0.9`), then computes frame-to-frame velocities, marks visible velocity outliers above mean + 5 SD, linearly interpolates only those outlier frames, and finally computes per-session 40th/60th percentiles from the cleaned visible `y` values. It does not average within 50 ms bins before setting the percentile thresholds.

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9
```

```python
visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
...
velocity_threshold = float(velocity_values.mean() + 5.0 * velocity_values.std())
outlier = visible & (speed > velocity_threshold)
...
xy[outlier, dim] = np.interp(
    timestamps[outlier], timestamps[base_valid], xy[base_valid, dim]
)
clean_y = xy[:, 1]
q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
```

iii. In Step 4 and Step 5 notes, the AI justified this as importing the method paper’s five-sigma video-cleaning idea while preserving low-confidence frames as “not visible” instead of mean-imputing them.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses per-session thresholds `q40` and `q60` computed from cleaned visible tongue-y frames. At aligned sample times, visible values below `q40` become class 0, visible values between `q40` and `q60` become class 1, visible values above `q60` become class 2, and nonvisible/no-nearby-frame samples become class 3.

ii.
```python
q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
...
categories = np.full(flat.shape, 3, dtype=np.int8)
y = clean_y[safe_idx]
categories[is_visible & (y < q40)] = 0
categories[is_visible & (y >= q40) & (y <= q60)] = 1
categories[is_visible & (y > q60)] = 2
```

iii. In Step 5 notes, the AI justified class 3 as the explicit occluded/low-confidence state required by the decoder instructions and justified the 40th/60th percentile thresholds as session-specific discretization boundaries.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by evaluating the last raw tongue frame at or before each decoder bin center, provided it is within about one camera frame (`<= 5.1 ms`) of that center. It does not compute a mean over all frames inside each 50 ms bin.

ii.
```python
flat = centers_global.ravel()
idx = np.searchsorted(timestamps, flat, side="right") - 1
safe_idx = np.clip(idx, 0, timestamps.size - 1)
age = flat - timestamps[safe_idx]
near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
is_visible = near & visible[safe_idx]
return categories.reshape(centers_global.shape), safe_idx.reshape(centers_global.shape)
```

```python
tongue_category, tongue_frame_idx = sample_tongue_categories(
    centers_global, tracking_ts, clean_y, visible, q40, q60
)
outputs[:, 3, :] = tongue_category
```

iii. In Step 5 notes, the AI explicitly said it was following the reference marker-alignment code’s sample-and-hold convention: use the last frame at or before the requested time rather than bin-averaging visible and invisible frames together.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or suspect data by exclusion or explicit missing-state encoding. Sessions with no classifier-good units are excluded during file discovery; sessions where retained good units lack region annotations raise errors; trials outside the ephys block, trials with any retained-unit invalidity, water trials, and all-zero/no-coverage trials are dropped; low-confidence or absent tongue frames are mapped to output class 3. The tongue cleaner interpolates only high-confidence velocity outliers, not low-confidence frames.

ii.
```python
if n_good:
    usable.append(path)
else:
    excluded.append({...})
```

```python
if any((not x) or x.lower() == "nan" for x in region_names):
    raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")
...
selected_trial_idx = candidate_trial_idx[~water_trial]
...
zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
```

```python
visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
...
categories = np.full(flat.shape, 3, dtype=np.int8)
```

iii. In Step 4 and Step 5 notes, the AI’s rationale was that truly unrecorded or invalid neural trials should be removed rather than fabricated as zero activity, whereas tongue invisibility is a meaningful output state and should remain explicit as class 3.

## 10-a. What are the most time-consuming steps of the code?

i. The code’s dominant costs are the repeated full-file HDF5 reads and the per-session neural binning loop over all retained units. Within `convert_session`, loading spike arrays and tongue-tracking arrays and then executing `bin_selected_units` is the heaviest processing. At the end, serializing the large pickle is another major cost.

ii.
```python
for path in files:
    with h5py.File(path, "r") as nwb:
        ...
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

```python
for out_unit, raw_unit in enumerate(unit_indices):
    lo = 0 if raw_unit == 0 else int(spike_index[raw_unit - 1])
    hi = int(spike_index[raw_unit])
    spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
    ...
    counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
```

iii. In Step 6 and Step 7 notes, the AI explicitly described neural conversion as the main workload and noted separate pickle-write time; it justified its low-level HDF5 approach as a response to those expected bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious remaining Python loop is the per-unit loop in `bin_selected_units`. The code also loops over x/y dimensions when interpolating tongue outliers, and it loops over files more than once at program level. The per-unit loop is partially vectorized internally, but not eliminated.

ii.
```python
for out_unit, raw_unit in enumerate(unit_indices):
    ...
    trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
    ...
    counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
```

```python
for dim in range(2):
    xy[outlier, dim] = np.interp(
        timestamps[outlier], timestamps[base_valid], xy[base_valid, dim]
    )
```

iii. In Step 6 notes, the AI justified leaving the neural binning at one loop per unit because the source spike-time storage is ragged; it emphasized vectorization inside that loop rather than full elimination of the loop.

## 10-c. What processing does the code repeat multiple times?

i. The code reopens and rescans the NWB files multiple times. It makes one pass in `discover_usable_files()` to count good units, another pass in `main()` to build the global region vocabulary, and then a third pass in `convert_session()` to do the actual conversion. Some per-session metadata such as `stim_on`, `tongue_frame_idx`, and `plot_payload` fields are also computed even when only a subset is later needed.

ii.
```python
all_files, excluded = discover_usable_files()
...
for path in files:
    with h5py.File(path, "r") as nwb:
        good = nwb["units/classification"][()] == b"good"
        region_set.update(decode_text(x).strip() for x in nwb["units/anno_name"][good])
...
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

iii. The AI did not present this as repetition in its notes; instead Step 6 emphasized one-session-at-a-time streaming. The repeated file passes appear to be an implementation tradeoff made to precompute exclusions and region vocabularies before full conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code constructs substantial diagnostic/provenance data that is not used by the decoder itself: `plot_payload` is built for every session even when `--show-processing` is off, `tongue_frame_idx` is computed mainly for diagnostics, and many `session_info` bookkeeping fields are written for provenance rather than downstream model input. It also computes `stim_on` separately before storing the full input tensor.

ii.
```python
plot_payload = {
    "session_id": session_id,
    ...
    "tracking_frame_idx": tongue_frame_idx,
    "go_times": go_times,
    "q40": q40,
    "q60": q60,
}
```

```python
if args.show_processing and number <= 2:
    plot_path = Path("/app") / f"processing_{result.info['session_id']}.png"
    plot_processing(result.plot_payload, plot_path)
```

```python
"metadata": {
    ...
    "excluded_sessions": excluded if not args.sample else [],
    "session_info": session_info,
},
```

iii. In Step 6 and Step 7 notes, the AI justified extra plotting/provenance work as sanity-check support for validating the conversion, not as data needed by the final decoder representation.
