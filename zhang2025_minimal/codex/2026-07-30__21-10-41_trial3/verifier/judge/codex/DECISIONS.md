# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `one.search()` over the Brainwidemap cache to enumerate sessions. Instead, it reads `/app/code/code_zhang2025/data/bwm_release.csv`, groups that table by `eid`, and then loads per-session objects and per-probe datasets through ONE. Trials come from `one.load_object(..., "trials")`; wheel, whisker, cluster metrics, channels, and spike arrays are loaded on demand per session/probe.

ii. ```python
release_df = pd.read_csv(args.release_csv)
grouped = list(release_df.groupby("eid", sort=False))
```

```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
```

```python
metrics_path = one.load_dataset(
    eid,
    "clusters.metrics.pqt",
    collection=collection,
    revision=SPIKE_SORTING_REVISION,
    download_only=True,
)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as using the "2025 BWM release table" and says it matches the release-level paper counts of 459 sessions, 699 insertions, 139 subjects, and 12 labs.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the release CSV. During assembly, the AI creates `subjects` in first-seen order from the converted sessions and records `subject_idx` using that mapping.

ii. ```python
release_df["subject"] = release_df["subject"].astype(str)
```

```python
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
```

iii. No separate justification is given beyond using the release CSV as the source-of-truth session table.

## 1-c. How are the data split into sessions?

i. Sessions are split by `eid`. The release CSV is grouped by `eid`, and each group is processed as one session, with all listed probes for that `eid` merged later.

ii. ```python
grouped = list(release_df.groupby("eid", sort=False))
```

```python
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The justification in the notes is implicit: the release table is probe-level, so grouping by `eid` reconstructs sessions.

## 1-d. How are the data split into trials?

i. The AI loads the trials ALF object into a DataFrame and iterates over trial rows. It first builds a base validity mask, then loops through the surviving trial indices and constructs one neural/input/output example per kept trial.

ii. ```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
```

```python
for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
    ...
    kept_trial_indices.append(trial_idx)
```

iii. No explicit justification is given; the code treats the trials table as already trial-structured.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if several conditions hold: required columns are non-NaN; reaction time `firstMovement_times - stimOn_times` is between `0.08` and `2.0` s; `choice != 0`; optional `feedback_times - goCue_times <= 10.0`; and both wheel and whisker traces cover the whole stimulus-aligned window closely enough for interpolation. Trials failing wheel/whisker interpolation are dropped later inside the per-trial loop.

ii. ```python
TRIAL_NAN_EXCLUDE = (
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
)
```

```python
reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= trials["choice"].to_numpy() != 0
```

```python
if "goCue_times" in trials.columns:
    ...
    trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
    mask &= trial_len_ok
```

```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
if wheel_interp is None or whisker_interp is None:
    continue
```

iii. `CONVERSION_NOTES.md` says this "matched the trial mask used in the reference code and the exclusions described in the papers," and specifically cites the `0.08-2.0 s` reaction-time window and no-choice exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from spike times and spike cluster assignments, but the AI also uses cluster metrics and channel brain-location IDs to decide which units survive QC and how they are labeled anatomically.

ii. ```python
spikes_times_path = one.load_dataset(
    eid,
    "spikes.times.npy",
    collection=probe.collection,
    revision=SPIKE_SORTING_REVISION,
    download_only=True,
)
spikes_clusters_path = one.load_dataset(
    eid,
    "spikes.clusters.npy",
    collection=probe.collection,
    revision=SPIKE_SORTING_REVISION,
    download_only=True,
)
```

```python
metrics = pd.read_parquet(metrics_path)
cluster_channels = np.asarray(one.load_dataset(... "clusters.channels.npy" ...))
channel_region_ids = np.asarray(one.load_dataset(... "channels.brainLocationIds_ccf_2017.npy" ...))
```

iii. `CONVERSION_NOTES.md` says units are filtered with `clusters.metrics.label >= 1`, restricted to grey matter, and remapped to Beryl regions.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into non-overlapping 20 ms bins over a `[-0.5, 1.5]` s window around stimulus onset, merges probes within a session, and stores the result as per-trial neuron-by-time spike-count matrices. It does not divide counts by bin width, so the exported neural data are counts rather than firing rates in Hz.

ii. ```python
probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
```

```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
...
np.add.at(
    probe_counts[trial_idx],
    (trial_clusters[in_bounds], bins[in_bounds]),
    1,
)
```

```python
session_counts[:, offset:end_offset, :] = probe_counts
...
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The notes justify the window and bin size as matching the methods paper, but they do not mention the change from firing rates to raw counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `label >= 1`, further restricts them to grey-matter regions, excludes acronyms `void`, `root`, and `grey`, then remaps surviving acronyms to Beryl and applies an additional post hoc region filter: at least 5 neurons in a session-region and at least 2 sessions per region globally.

ii. ```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
```

```python
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
...
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
...
globally_valid_regions = {
    region_name
    for region_name, session_ids in sessions_with_region.items()
    if len(session_ids) >= MIN_SESSIONS_PER_REGION
}
```

iii. `CONVERSION_NOTES.md` claims these extra region filters match the data-paper region criteria and says they were added to get closer to the reference analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset. For each kept trial the code uses `stimOn_times + WINDOW_START` as the binning start, so the 100 bins span `[-0.5, 1.5]` s relative to stimulus onset.

ii. ```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
```

```python
end_time = start_time + (WINDOW_END - WINDOW_START)
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The notes explicitly say the common alignment event is stimulus onset and the common window is `[-0.5, 1.5]` s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms, giving 100 bins over the 2 s window. The neural data are directly binned at that resolution; no further temporal rebinning is applied.

ii. ```python
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
```

```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The notes justify this as matching the methods-paper description of 2 s trials split into 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time input is not read from a raw array directly. It is generated from the chosen stimulus-onset-aligned bin grid, which itself is anchored by each trial's `stimOn_times`.

ii. ```python
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
```

```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The notes justify it only indirectly by stating that stimulus onset is the common alignment event and the bin size is 20 ms.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a deterministic 100-element vector from the stimulus-aligned window, but uses the right edge of each 20 ms bin (`-0.48, -0.46, ..., 1.50`) rather than the bin center.

ii. ```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

```python
input_trial = np.vstack(
    [
        relative_time,
        np.full(NBINS, trial_num, dtype=np.float32),
    ]
).astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` says continuous traces are interpolated onto the "right edge" of each 20 ms trial bin, which is the clearest justification given for this choice.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI uses the same 20 ms trial grid for neural, wheel, whisker, and the time input, but represents the time input by bin right edges instead of centers. So it is aligned to the same trial window, but with a specific within-bin timestamp convention.

ii. ```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The notes say the behavior traces are interpolated onto the right edge of each 20 ms bin; that same convention is used for the time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`, treated as constant within a block, with a block change inferred whenever that value changes or becomes non-finite.

ii. ```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    ...
    for idx in range(1, len(prob_left)):
        prev = prob_left[idx - 1]
        curr = prob_left[idx]
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
```

iii. No standalone justification is given, but this follows the common assumption that block identity can be reconstructed from constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the count before later trial filtering, so dropped trials still advance the counter, but it counts from `1` rather than `0`.

ii. ```python
current = 1
trial_num[0] = current
...
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
        trial_num[idx] = current
```

```python
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
```

iii. There is no explicit justification for starting from 1. The notes only name `trial_number_in_block` as a decoder input.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trial-table `choice` column.

ii. ```python
choice_val = float(trials.iloc[trial_idx]["choice"])
```

iii. The notes explicitly state the IBL sign convention: `choice == 1` means left and `choice == -1` means right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes IBL choice values to the requested decoder labels: left `0`, right `1`. Trials with any other value are skipped.

ii. ```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. `CONVERSION_NOTES.md` gives this exact mapping as the output-coding rule.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii. ```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. The notes say the output is the trial's `prior_probability_left`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps the three allowed values onto category codes `0, 1, 2` for `0.2, 0.5, 0.8`.

ii. ```python
if prob_left == 0.2:
    prior_out = 0
elif prob_left == 0.5:
    prior_out = 1
elif prob_left == 0.8:
    prior_out = 2
else:
    continue
```

iii. `CONVERSION_NOTES.md` lists the same `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` coding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from wheel timestamps and wheel position.

ii. ```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The notes state that wheel is loaded from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz, computes a filtered velocity using `velocity_filtered`, takes absolute value to get speed, and then linearly interpolates that speed trace onto each trial's 20 ms stimulus-aligned bin right edges.

ii. ```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as using the same Butterworth-based wheel processing as `brainbox.behavior.wheel.velocity_filtered`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes two global tertile cut points from all wheel-speed values in the converted dataset, then uses `np.digitize` to assign `low`, `medium`, and `high`.

ii. ```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

```python
def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The notes explicitly say wheel speed is "discretized into 3 bins using global tertiles over the converted dataset."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset and interpolated onto the same 100 trial bins used for the neural data, using the bin right edge timestamps.

ii. ```python
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
```

iii. The notes say all streams share the stimulus-onset-aligned 20 ms grid and that dynamic traces are interpolated to bin right edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derives whisker motion energy only from the left-camera `times` and `ROIMotionEnergy` arrays. It does not fall back to the right camera.

ii. ```python
cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
times = np.asarray(cam["times"], dtype=np.float64)
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
```

iii. The notes and trajectory explicitly justify this as "left camera only"; the trajectory says this was changed because the paper's `60 Hz` whisker signal description pointed to the left camera specifically.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace directly, applies no extra filtering or normalization, and linearly interpolates it onto each trial's 20 ms stimulus-aligned bin right edges.

ii. ```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. `CONVERSION_NOTES.md` says whisker motion energy is loaded from `ROIMotionEnergy` and camera `times`, with continuous traces linearly interpolated onto the right edge of each bin.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI computes two global tertile cut points from all whisker-motion-energy values in the converted dataset, then digitizes values into `low`, `medium`, and `high`.

ii. ```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

```python
discretize(whisker_vals, whisker_edges)
```

iii. The notes explicitly say whisker motion energy is "discretized into 3 bins using global tertiles over the converted dataset."

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to stimulus onset and interpolated onto the same 100 trial bins as the neural data, again using the bin right edge timestamps.

ii. ```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
```

iii. The notes use the same justification as for wheel speed: a shared stimulus-aligned 20 ms grid with right-edge interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops unusable data. It skips sessions if wheel loading fails, if no left-camera whisker trace exists, if fewer than 2 trials survive behavior/alignment checks, if no good units remain, or if fewer than 2 trials have nonzero neural activity after binning. At the trial level it drops trials that fail interpolation coverage or category mapping.

ii. ```python
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
if len(kept_trial_indices) < 2:
    print(f"Skip session {eid}: only {len(kept_trial_indices)} trials after alignment and behavior checks.")
    return None
```

```python
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
if nonzero_trial_mask.sum() < 2:
    print(f"Skip session {eid}: fewer than 2 nonzero neural trials after binning.")
    return None
```

iii. The notes justify these as sanity/quality checks, especially requiring common time dimensions and usable aligned dynamic traces.

## 10-a. What are the most time-consuming steps of the code?

i. The code strongly suggests that the slowest work is loading large per-probe spike arrays and then binning them trial-by-trial. The AI did not document a separate performance analysis, but those are the dominant heavy steps in the implementation.

ii. ```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

```python
for trial_idx, start_time in enumerate(trial_starts):
    ...
    np.add.at(
        probe_counts[trial_idx],
        (trial_clusters[in_bounds], bins[in_bounds]),
        1,
    )
```

iii. No explicit justification was found in the notes; this is inferred from the code structure.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several obvious Python loops in place: the per-trial loop that interpolates wheel/whisker and builds inputs/outputs, the per-trial spike-binning loop in `bin_probe_spikes`, and the per-neuron/session loops used while rebuilding region indices after post hoc filtering.

ii. ```python
for trial_idx in np.flatnonzero(base_mask):
    ...
    wheel_interp = interpolate_trial_signal(...)
    whisker_interp = interpolate_trial_signal(...)
```

```python
for trial_idx, start_time in enumerate(trial_starts):
    ...
```

```python
for neuron_idx, region_name in enumerate(sess["brain_regions"]):
    ...
```

iii. No explicit justification was given for keeping these loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI does a second region-processing pass after assembling the full dataset: it first converts sessions with original per-unit region labels, then remaps/filter regions in `apply_region_filters`, rebuilds `brain_regions` and `brain_region_idx`, and rewrites session-level neural arrays. It also interpolates wheel and whisker separately in the same per-trial loop for every kept trial.

ii. ```python
data = build_data_from_session_results(session_results, release_df, args)
data, filter_stats = apply_region_filters(data)
```

```python
for session_idx, keep_mask in enumerate(session_region_keep):
    final_keep = keep_mask & np.isin(beryl_names_per_session[session_idx], list(globally_valid_regions))
    ...
    filtered_trials = [
        trial[final_keep].astype(np.float16, copy=False)
        for trial in data["neural"][session_idx]
    ]
```

iii. The trajectory says this region-filtering pass was added later to fix a mismatch in brain-region counts without rerunning the full raw conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does several extra steps that downstream decoder training does not need: it bins and stores neural trials before later dropping all-zero trials; it constructs and then post-filters region assignments after full assembly; and it retains extra metadata such as `lab`, `date`, `camera_view`, `probe_names`, and pre-filter trial counts that the decoder itself does not consume.

ii. ```python
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
if not np.all(nonzero_trial_mask):
    session_counts = session_counts[nonzero_trial_mask]
    ...
```

```python
session_info.append(
    {
        "eid": sess["eid"],
        "subject": sess["subject"],
        "lab": sess["lab"],
        "date": sess["date"],
        "camera_view": sess["camera_view"],
        "probe_names": sess["probe_names"],
        ...
    }
)
```

iii. The trajectory explains the post hoc region filter specifically as a pragmatic correction step after the first full export produced the wrong region count.
