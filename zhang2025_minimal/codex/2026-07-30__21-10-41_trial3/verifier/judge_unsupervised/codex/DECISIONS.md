# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads the BWM release CSV, groups rows by `eid`, and treats each `eid` as one session. For each session it loads the ALF `trials` object, wheel object, left-camera motion-energy object, and each probe's spike-sorting files through ONE. It only keeps sessions that survive later wheel, whisker, trial, and neuron checks.

ii. 
```python
release_df = pd.read_csv(args.release_csv)
grouped = list(release_df.groupby("eid", sort=False))

for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
wheel = one.load_object(eid, "wheel", collection="alf")
cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
metrics_path = one.load_dataset(eid, "clusters.metrics.pqt", collection=collection, revision=SPIKE_SORTING_REVISION, download_only=True)
spikes_times_path = one.load_dataset(eid, "spikes.times.npy", collection=probe.collection, revision=SPIKE_SORTING_REVISION, download_only=True)
```

iii. The notes say the agent used `bwm_release.csv` as the session/probe inventory and matched the release totals of 459 sessions and 699 insertions. In the trajectory, the agent explicitly mirrored the reference `prepare_data(...)` pattern from `0_data_caching.py`, which also starts from the release table and loads trials plus merged probe data through ONE.

## 1-b. How are the data split into subjects (mice)?

i. Subjects come from the `subject` column in the release CSV. After sessions are converted, the script builds a unique `subjects` list and a per-session `subject_idx` array.

ii. 
```python
subject = str(session_rows["subject"].iloc[0])
```

```python
subjects = []
subject_to_idx = {}

for sess in session_results:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. This follows directly from the release metadata rather than inferring subjects from paths. The notes also summarize converted subject counts from the resulting dataset.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release CSV. All probes listed for the same `eid` are merged into one session-level example.

ii. 
```python
grouped = list(release_df.groupby("eid", sort=False))
```

```python
for _, row in session_rows.sort_values("probe_name").iterrows():
    probe = load_probe_info(one, eid, str(row["probe_name"]), id_to_acronym, grey_ids)
    if probe is not None:
        probes.append(probe)
```

iii. The notes say "Probe data are merged within session." The trajectory shows the agent copied this from the reference `prepare_data(...)` path, which calls `one.eid2pid(eid)` and merges probes for a session.

## 1-d. How are the data split into trials?

i. Trials come from the ALF trials table. The script iterates trial rows that pass the base trial mask, then further keeps only trials whose wheel and whisker signals cover the whole analysis window and whose categorical outputs can be encoded. Neural data are then binned separately for the retained trial starts.

ii. 
```python
trials = load_trials(one, eid)
base_mask = make_base_trial_mask(trials)

for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
    start_time = stim_on + WINDOW_START
    end_time = stim_on + WINDOW_END
    wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
    whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
    if wheel_interp is None or whisker_interp is None:
        continue
    ...
    kept_trial_indices.append(trial_idx)
```

iii. This matches the reference code structure: `load_trials_and_mask(...)` creates a trial mask, `bin_behaviors(...)` can return missing trials as `None`, and `align_spike_behavior(...)` removes trials failing either the trial mask or behavior availability.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed if any required event is missing, if reaction time is outside 0.08 to 2.0 s, if `choice == 0`, if `feedback_times - goCue_times > 10.0`, or if wheel/whisker interpolation fails over the full analysis window. Sessions are dropped if fewer than 2 trials remain.

ii. 
```python
TRIAL_NAN_EXCLUDE = (
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
)

for col in TRIAL_NAN_EXCLUDE:
    mask &= trials[col].notna().to_numpy()
reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= trials["choice"].to_numpy() != 0
...
trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
```

```python
if wheel_interp is None or whisker_interp is None:
    continue
...
if len(kept_trial_indices) < 2:
    return None
```

iii. The notes say this was meant to match both the papers and `load_trials_and_mask(...)` from `ibl_data_utils.py`. The trajectory contains the exact reference query: non-NaN checks for the same six columns, the same 0.08 to 2.0 s reaction-time bounds, optional `choice == 0` exclusion, and `max_trial_len=10.0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived from per-probe `spikes.times.npy` and `spikes.clusters.npy`, with cluster inclusion and region labels coming from `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii. 
```python
metrics = pd.read_parquet(metrics_path)
cluster_channels = np.asarray(one.load_dataset(eid, "clusters.channels.npy", ...), dtype=np.int64)
channel_region_ids = np.asarray(one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", ...), dtype=np.int64)
```

```python
spikes_times_path = one.load_dataset(eid, "spikes.times.npy", collection=probe.collection, revision=SPIKE_SORTING_REVISION, download_only=True)
spikes_clusters_path = one.load_dataset(eid, "spikes.clusters.npy", collection=probe.collection, revision=SPIKE_SORTING_REVISION, download_only=True)
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. In the trajectory, the agent inspected the reference `prepare_data(...)` utility, where the neural dictionary is built from `spike_times`, `spike_clusters`, and `cluster_regions`. The agent changed the exact source of region labels from `clusters['acronym']` to a channel-based lookup plus atlas metadata.

## 2-b. How is the `neural` data processed?

i. Within each retained session, probes are merged, spikes are restricted to the kept clusters, then spike counts are accumulated into non-overlapping 20 ms bins over a 2 s stimulus-aligned window for each retained trial. The stored trial matrices are `(n_neurons, 100)` and cast to `float16`.

ii. 
```python
probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)

for trial_idx, start_time in enumerate(trial_starts):
    end_time = start_time + (WINDOW_END - WINDOW_START)
    start_idx = np.searchsorted(kept_times, start_time, side="left")
    end_idx = np.searchsorted(kept_times, end_time, side="left")
    ...
    bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

```python
session_counts[:, offset:end_offset, :] = probe_counts
...
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The notes explicitly say "Probe data are merged within session" and "Spike counts are binned into non-overlapping 20 ms bins." This is also the same high-level processing as the reference `merge_probes(...)` plus `bin_spiking_data(...)` utilities.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only clusters with `label >= 1`, whose assigned region is in grey matter, and whose acronym is not `void`, `root`, or `grey`. After conversion it remaps surviving acronyms to Beryl and keeps only Beryl regions with at least 5 neurons in that session-region and at least 2 sessions overall. Trials with all-zero neural activity are dropped.

ii. 
```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
good_cluster_ids = np.flatnonzero(good).astype(np.int32)
```

```python
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
unique_names, counts = np.unique(valid_names, return_counts=True)
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
...
globally_valid_regions = {
    region_name for region_name, session_ids in sessions_with_region.items()
    if len(session_ids) >= MIN_SESSIONS_PER_REGION
}
```

```python
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The notes justify this with the data paper's inclusion criteria for well-isolated neurons and grey-matter regions. In the trajectory, however, the reference `load_spiking_data(...)` helper allowed `qc=None` in `prepare_data(...)`, so the agent intentionally chose a stricter interpretation of the paper than the inspected code path.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural trial is aligned to stimulus onset, with trial start defined as `stimOn_times - 0.5` and trial end as `stimOn_times + 1.5`.

ii. 
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
...
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
```

```python
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
```

iii. The notes call this the "Common alignment event: stimulus onset." The trajectory shows the reference `0_data_caching.py` parameters were also `align_time='stimOn_times'` and `time_window=(-.5, 1.5)` for the inspected dataset-construction path.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins, giving 100 bins over a 2 s window. No extra temporal rebinning is applied after this fixed binning.

ii. 
```python
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
```

```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The notes justify this by the methods excerpt stating 2 s trials with 20 ms bins. The trajectory also shows the reference code set `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` plus fixed constants for the analysis window and bin size. The stored values are relative times, not a raw data column copied from disk.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
...
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
```

iii. The agent's notes describe a common stimulus-onset alignment. The trajectory shows the agent chose to represent this decoder input as an explicit time axis shared with the neural bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a 100-element vector at the right edge of each 20 ms bin: `[-0.48, -0.46, ..., 1.50]` seconds relative to stimulus onset. It repeats this same vector for every retained trial in the session.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
input_trial = np.vstack(
    [
        relative_time,
        np.full(NBINS, trial_num, dtype=np.float32),
    ]
).astype(np.float32, copy=False)
```

iii. There is no paper- or code-level reference implementation for this exact decoder input. The choice is justified implicitly by the instruction that time since stimulus onset should be continuous and time-varying, and by the agent's decision to use the same bin grid for all streams.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100 stimulus-aligned bin positions as the neural data and is stored trial-by-trial in the same order as the neural matrices.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
...
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
input_trials.append(input_trial)
```

iii. The notes emphasize one common alignment event and one common bin size. That is also how the behavior interpolation helper in the reference code constructs its sample grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived entirely from the raw `probabilityLeft` sequence in the trials table.

ii. 
```python
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
```

```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    ...
    if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
        current += 1
    else:
        current = 1
```

iii. This input is not present in the reference paper/code, so the agent had to invent a mapping from available task variables. The trajectory does not show a stronger justification than using the block prior field already present in the trials table.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script computes run length within consecutive equal `probabilityLeft` values across the full session, then repeats the resulting scalar across all 100 time bins for each kept trial.

ii. 
```python
trial_num = float(block_trial_num[trial_idx])
input_trial = np.vstack(
    [
        relative_time,
        np.full(NBINS, trial_num, dtype=np.float32),
    ]
)
```

iii. The decision is only loosely justified by the task description. There is no evidence in the reference pipeline that filtered-out trials should or should not count toward the within-block index, so this was an agent choice.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw trials-table `choice` column.

ii. 
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
```

iii. The notes say this follows the IBL sign convention, and the trajectory shows the agent checked easy-trial behavior to resolve which sign meant left versus right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `choice == 1` is remapped to left `0`, `choice == -1` is remapped to right `1`, and that categorical value is repeated across all 100 time bins for the trial.

ii. 
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

```python
np.full(NBINS, choice_out, dtype=np.int8)
```

iii. The notes explicitly document the sign conversion. The trajectory also contains the agent's justification that easy left trials mostly had `choice == 1` and easy right trials mostly had `choice == -1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials-table `probabilityLeft` column.

ii. 
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. The notes and the reference utilities both use `probabilityLeft` as the block or prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw prior values are mapped categorically as `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeated across all 100 time bins for the trial.

ii. 
```python
if prob_left == 0.2:
    prior_out = 0
elif prob_left == 0.5:
    prior_out = 1
elif prob_left == 0.8:
    prior_out = 2
else:
    continue
```

```python
np.full(NBINS, prior_out, dtype=np.int8)
```

iii. The notes document exactly this three-way mapping, which is also the mapping requested by the task instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the ALF wheel object's `timestamps` and `position`.

ii. 
```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The notes say wheel is loaded from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The trajectory also shows the agent read the reference wheel utility and the reference `SessionLoader.load_wheel()` implementation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1000 Hz, filtered with `velocity_filtered(...)`, converted to absolute velocity, then linearly interpolated onto the 20 ms trial bin right edges in the stimulus-aligned window.

ii. 
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

iii. The notes explicitly claim this matches `brainbox.behavior.wheel.velocity_filtered`. The trajectory confirms the reference `SessionLoader.load_wheel()` uses the same interpolation and filtered velocity pattern.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, all continuous wheel-speed samples are pooled across the converted dataset and discretized into three bins using the global 1/3 and 2/3 quantiles.

ii. 
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

```python
def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The notes justify this only as a practical way to satisfy the task requirement of 3 categories with names `low`, `medium`, `high`. The reference paper/code inspected in the trajectory did not specify these category thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset window as the neural data and sampled at the right edge of each 20 ms bin.

ii. 
```python
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
```

iii. The notes say all streams use a common stimulus-onset alignment and right-edge interpolation. The trajectory shows the reference `get_behavior_per_interval(...)` helper also interpolated behaviors at `interval_begin + binsize, ..., interval_end`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. In the final code, whisker motion energy is derived only from the left camera's `times` and `ROIMotionEnergy` arrays. If the left camera cannot be loaded, the session is skipped.

ii. 
```python
cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
times = np.asarray(cam["times"], dtype=np.float64)
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
if len(times) and len(times) == len(values):
    return "left", times, values
...
return None, None, None
```

iii. The justification changed during the run. Earlier notes in the trajectory said "use left camera if available, otherwise use right camera," matching the reference utility, but the final code and final notes were changed to "left camera only" after the agent observed that the resulting session count landed close to 433 anyway.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the left-camera motion-energy trace, checks for finite coverage, linearly interpolates it onto the stimulus-aligned 20 ms trial bins, and later discretizes the pooled continuous values into three categories.

ii. 
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
if wheel_interp is None or whisker_interp is None:
    continue
```

```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

iii. The notes say motion energy is loaded from `ROIMotionEnergy` and camera `times`, then interpolated to the common bin grid. The trajectory shows the reference utility uses `whiskerMotionEnergy` from `SessionLoader.load_motion_energy(...)` and falls back from left to right, so the final left-only choice was deliberate rather than accidental.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, the continuous whisker trace is pooled across the converted dataset and cut into three categories at the global 1/3 and 2/3 quantiles.

ii. 
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
...
discretize(whisker_vals, whisker_edges)
```

iii. The notes justify this only as a way to satisfy the required 3-bin decoder output. No stronger thresholding rule appears in the inspected reference materials.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered 2 s window as the neural data and interpolated at the right edge of each 20 ms bin.

ii. 
```python
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
```

iii. The notes say continuous traces are interpolated onto the right edge of each 20 ms trial bin. The trajectory shows the reference interpolation helper uses the same sample grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is drop-based rather than imputation-based. Missing required trial events, missing wheel loads, missing left whisker loads, interpolation gaps, unexpected prior values, non-response choices, all-zero neural trials, and sessions with fewer than 2 retained trials are all discarded.

ii. 
```python
try:
    wheel_times, wheel_speed = load_wheel_speed(one, eid)
except Exception as exc:
    print(f"Skip session {eid}: wheel load failed: {exc}")
    return None
```

```python
camera_view, whisker_times, whisker_me = load_whisker_motion_energy(one, eid)
if whisker_times is None or whisker_me is None:
    print(f"Skip session {eid}: no whisker motion energy trace available.")
    return None
```

```python
if wheel_interp is None or whisker_interp is None:
    continue
...
if nonzero_trial_mask.sum() < 2:
    return None
```

iii. The notes present this as matching the trial-mask logic and the reference behavior-alignment logic. The trajectory confirms the reference helper also drops bad intervals, but it handled whisker missingness less aggressively by falling back to the right camera.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are the per-session ONE loads, reading large spike arrays for each probe, looping over every retained trial to bin spikes, and interpolating wheel and whisker traces for every retained trial. Loading and reserializing the 5.4 GB pickle is also expensive during reprocessing.

ii. 
```python
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

```python
for trial_idx, start_time in enumerate(trial_starts):
    ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

```python
for trial_idx in np.flatnonzero(base_mask):
    wheel_interp = interpolate_trial_signal(...)
    whisker_interp = interpolate_trial_signal(...)
```

iii. The agent did not document explicit performance profiling in the notes. This conclusion follows directly from the nested session, probe, and trial loops and from the size of the input/output artifacts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial spike-binning loop in `bin_probe_spikes(...)`, the per-trial wheel/whisker interpolation loop in `process_session(...)`, the run-length computation in `compute_trial_number_in_block(...)`, and the repeated Python loops that build subject and region indices.

ii. 
```python
for trial_idx, start_time in enumerate(trial_starts):
    ...
```

```python
for trial_idx in np.flatnonzero(base_mask):
    ...
```

```python
for idx in range(1, len(prob_left)):
    ...
```

iii. No explicit rationale is recorded in the notes; this is an evaluation of the code structure itself.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly interpolates continuous behavior traces trial by trial, repeatedly remaps or rebuilds region indices first at fine acronym level and then again after Beryl filtering, and repeatedly scans spike arrays once per trial for each probe.

ii. 
```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

```python
for neuron_idx, region_name in enumerate(sess["brain_regions"]):
    ...

for region_names in new_region_names_per_session:
    ...
```

```python
for probe in probes:
    probe_counts = bin_probe_spikes(one, eid, probe, trial_starts)
```

iii. The notes do not call this out. It is evident from the implementation that several indexing and interpolation passes are performed serially.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores continuous wheel and whisker traces only to later collapse them into tertile bins, repeats static per-trial outputs across all 100 time bins, rebuilds pre-Beryl region indices that are discarded by `apply_region_filters(...)`, and stores metadata fields that the downstream decoder never uses.

ii. 
```python
wheel_trials.append(wheel_interp)
whisker_trials.append(whisker_interp)
...
output_data.append(build_output_trials(sess, wheel_edges, whisker_edges))
```

```python
np.full(NBINS, choice_out, dtype=np.int8),
np.full(NBINS, prior_out, dtype=np.int8),
```

```python
data["metadata"]["session_info"] = new_session_info
```

iii. This was not justified in the notes; it is a byproduct of choosing a decoder-friendly export format and of doing region filtering only after an initial session conversion pass.
