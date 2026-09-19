# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI used `/app/code/code_zhang2025/data/bwm_release.csv` as the release index, grouped it by session `eid`, and then loaded session-level and probe-level data through ONE on demand. Trials came from `one.load_object(..., "trials")`; wheel and whisker data came from `one.load_object(..., "wheel"/"leftCamera")`; spike-sorting files came from `one.load_dataset(...)` per probe. It also queried Alyx once to build a region-id-to-acronym map.

ii. 
```python
release_df = pd.read_csv(args.release_csv)
...
grouped = list(release_df.groupby("eid", sort=False))
...
result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
wheel = one.load_object(eid, "wheel", collection="alf")
cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
...
spikes_times_path = one.load_dataset(...)
spikes_clusters_path = one.load_dataset(...)
```

iii. In the trajectory, the agent first identified the 2025 BWM release table as the right freeze to target (step 23), then explicitly moved away from broad probe/session loaders toward minimal per-dataset ONE loads so it would only fetch the specific ALF files it needed (steps 59, 67, 107).

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the `subject` column of the release CSV. During assembly, the AI builds `subjects` in first-seen order and a `subject_to_idx` map, then records one `subject_idx` per converted session.

ii. 
```python
release_df["subject"] = release_df["subject"].astype(str)
```

```python
subjects = []
subject_to_idx = {}
...
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
...
subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The trajectory shows the agent deciding to rely on release metadata rather than path parsing, first via the release inventory and then via per-session assembly keyed by the subject recorded there (steps 95, 107).

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release CSV. The converter groups the release table by `eid`, processes one grouped set of probe rows at a time, and emits one session entry in `neural`, `input`, and `output` per successful `eid`.

ii. 
```python
grouped = list(release_df.groupby("eid", sort=False))
...
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. In the trajectory, the agent treated the release table as already session-indexed through `eid` and focused its design work on per-session processing rather than deriving sessions some other way (steps 23, 107).

## 1-d. How are the data split into trials?

i. Trials are taken from the rows of the session’s ALF trials table. The AI loads the full table, builds a boolean trial mask, and then iterates over the surviving row indices. Each kept row becomes one trial in the session lists.

ii. 
```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
...
base_mask = make_base_trial_mask(trials)
...
for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
    ...
    kept_trial_indices.append(trial_idx)
```

iii. The trajectory shows that the agent read the trials table as the authoritative source of trial structure, then concentrated on reproducing the filtering and alignment logic rather than redefining trial boundaries (steps 18, 36, 107).

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only trials with non-NaN values in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time in `[0.08, 2.0]` s; nonzero choice; optional `feedback_times - goCue_times <= 10 s`; valid wheel and whisker coverage over the whole trial window; and, after neural binning, nonzero total neural activity. Sessions with fewer than two remaining trials were dropped.

ii. 
```python
for col in TRIAL_NAN_EXCLUDE:
    mask &= trials[col].notna().to_numpy()
reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= trials["choice"].to_numpy() != 0
...
trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
mask &= trial_len_ok
```

```python
wheel_interp = interpolate_trial_signal(...)
whisker_interp = interpolate_trial_signal(...)
if wheel_interp is None or whisker_interp is None:
    continue
...
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The agent’s trajectory says it was trying to “extract the exact trial filtering” from the reference path (step 18), but after validation it added an extra rule to drop all-zero neural trials because the verifier warned about silent trials (step 151).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are built from `spikes.times.npy` and `spikes.clusters.npy` per probe, after selecting “good” cluster IDs using `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii. 
```python
metrics_path = one.load_dataset(..., "clusters.metrics.pqt", ...)
cluster_channels = np.asarray(one.load_dataset(..., "clusters.channels.npy", ...), dtype=np.int64)
channel_region_ids = np.asarray(
    one.load_dataset(..., "channels.brainLocationIds_ccf_2017.npy", ...),
    dtype=np.int64,
)
```

```python
spikes_times_path = one.load_dataset(..., "spikes.times.npy", ...)
spikes_clusters_path = one.load_dataset(..., "spikes.clusters.npy", ...)
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. In the trajectory, the agent explicitly narrowed probe loading to exactly these spike and cluster metadata files to avoid pulling large unnecessary arrays, and it used `clusters.metrics.label >= 1` plus region metadata as the gating information for which spikes to keep (steps 59, 72, 80).

## 2-b. How is the `neural` data processed?

i. The AI bins kept spikes into non-overlapping 20 ms bins over a `[-0.5, 1.5]` s stimulus-aligned window, merges probe populations by concatenating neurons across probes, and stores the resulting per-trial arrays as `float16`. It does not divide by bin width, so the saved neural data are spike counts rather than firing rates.

ii. 
```python
probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
...
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

iii. The trajectory shows that the agent committed early to a common 2 s, 20 ms stimulus-aligned representation (steps 36, 107). It discussed probe merging and explicit per-session auditability, but there is no trajectory evidence that it intended to convert counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters with `metrics["label"] >= 1`, requires the cluster’s channel region ID to descend from grey matter, excludes acronyms in `{"void", "root", "grey"}`, then applies a second pass that remaps surviving acronyms to Beryl and keeps only regions with at least 5 neurons within a session and at least 2 sessions overall.

ii. 
```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
good_cluster_ids = np.flatnonzero(good).astype(np.int32)
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

iii. The trajectory first shows the agent deciding to keep only well-isolated units for feasibility (step 103). Later, after a first full run produced 472 regions instead of the paper’s cited 270, it added Beryl remapping and the `>=5 neurons/session-region` and `>=2 sessions/region` filters in a postprocessing pass (steps 504, 508, 512, 529).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset by defining each trial’s start as `stimOn_times + WINDOW_START`, then binning spikes from that start through the next 2 s. This makes time 0 the stimulus onset and the trial window `[-0.5, 1.5]` s relative to it.

ii. 
```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
...
for trial_idx, start_time in enumerate(trial_starts):
    end_time = start_time + (WINDOW_END - WINDOW_START)
    ...
    bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. In the trajectory, the agent called this the main design decision and repeatedly justified using a single stimulus-aligned 2 s representation for all streams (steps 36, 107, 109).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins across a 2 s window, giving 100 bins per trial. No later temporal rebinning is applied.

ii. 
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
```

iii. The trajectory explicitly locks onto the 2 s / 20 ms representation as the common target format (steps 36, 107), and the code carries that through everywhere.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not loaded from a raw time-series variable. Instead, it is constructed from the stimulus-onset alignment convention, using `stimOn_times` only implicitly to define each trial window and a fixed relative time grid shared by all trials.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
...
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
```

iii. The trajectory shows the agent choosing a common stimulus-aligned representation rather than trying to recover a separate raw “time since stimulus” stream (steps 36, 107).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a fixed 100-element vector running from `-0.48` to `1.50` s in 20 ms steps. This is the right edge of each 20 ms bin, not the bin center, and the same vector is copied into every trial.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
...
input_trial = np.vstack(
    [
        relative_time,
        np.full(NBINS, trial_num, dtype=np.float32),
    ]
).astype(np.float32, copy=False)
```

iii. The trajectory indicates the agent wanted one common grid for all trial-aligned variables (steps 36, 107). The code’s metadata later describes the dynamic interpolation as occurring at “20 ms bin right edges,” which matches this implementation.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by using the same 20 ms trial grid as the wheel, whisker, and spike binning logic, but the AI uses bin right edges rather than bin centers. So it is synchronized to the same bins, with a half-bin offset relative to a center-based representation.

ii. 
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
...
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The trajectory repeatedly says the goal was a shared stimulus-aligned grid for neural and behavioral streams (steps 36, 107), and the code reflects that shared grid choice.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials["probabilityLeft"]`, treating contiguous runs of equal prior probability as blocks.

ii. 
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    ...
    if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
        current += 1
    else:
        current = 1
```

iii. The trajectory shows the agent using `probabilityLeft` as the block-defining variable because the trials table did not supply an explicit block counter (step 18 and the later implementation).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI walks through the full session’s `probabilityLeft` sequence before trial filtering, increments the counter when the prior stays the same, resets it when the prior changes or becomes non-finite, and assigns block positions starting at `1`.

ii. 
```python
trial_num = np.zeros(len(prob_left), dtype=np.float32)
...
current = 1
trial_num[0] = current
for idx in range(1, len(prob_left)):
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

iii. The trajectory indicates the agent wanted this value computed on the original trial order, before downstream trial dropping, so that the count reflected the animal’s real block position rather than the filtered subset (steps 18, 107).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `choice` is derived directly from `trials["choice"]`.

ii. 
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. In the trajectory, the agent says it resolved the sign convention empirically from easy trials and concluded `choice == 1` is left and `choice == -1` is right (step 95).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes `choice == 1` to output class `0` and `choice == -1` to output class `1`, and discards any other value.

ii. 
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. The trajectory says the agent explicitly checked and resolved the sign convention before implementation (step 95).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from `trials["probabilityLeft"]`.

ii. 
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
if prob_left == 0.2:
    prior_out = 0
elif prob_left == 0.5:
    prior_out = 1
elif prob_left == 0.8:
    prior_out = 2
else:
    continue
```

iii. The trajectory does not show a separate debate here; the agent treated the instruction-specified mapping as direct and implemented it in the per-trial loop.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and drops any other value.

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

iii. The trajectory indicates that once the session/trial representation was chosen, this mapping was treated as a straightforward implementation detail from the task instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel timestamps and wheel position arrays loaded from the session’s ALF wheel object.

ii. 
```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The trajectory shows the agent tracing the exact upstream wheel computation so it could reuse the bundled IBL wheel preprocessing instead of approximating it (steps 43, 67).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz with `interpolate_position`, computes low-pass-filtered velocity with `velocity_filtered`, takes absolute value to get speed, and then linearly interpolates that speed trace onto the stimulus-aligned trial grid.

ii. 
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

iii. In the trajectory, the agent explicitly decided to import and reuse the bundled `ibllib` wheel functions so the preprocessing would track the IBL implementation closely (step 67).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI concatenates wheel-speed values across all converted sessions and all trials, takes the global 1/3 and 2/3 quantiles of that pooled distribution, and digitizes each per-trial wheel-speed trace into classes `0`, `1`, and `2`.

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

iii. The trajectory does not show a separate justification for using global rather than per-session tertiles. The decision appears in the implementation path after the session results are pooled for final assembly.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by extracting the wheel trace over `[stimOn - 0.5, stimOn + 1.5]` and interpolating it onto the same 100 trial bins used for the neural data. The sample points are the right edges of those bins.

ii. 
```python
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The trajectory shows the agent insisting on one common stimulus-aligned grid for dynamic variables and neural activity (steps 36, 107), and this interpolation helper is how it enforced that.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the left camera’s `times` and `ROIMotionEnergy` arrays only. If the left camera trace is absent or malformed, the session is skipped rather than falling back to another camera.

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

iii. The trajectory shows that the agent originally considered a left-then-right fallback, but later deliberately switched to left-camera-only because it judged that choice more consistent with the paper text and more likely to recover the paper’s 433-session cohort (steps 224, 244, 247, 260).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the raw released `ROIMotionEnergy` values, does no explicit filtering or normalization, and linearly interpolates them onto the stimulus-aligned 20 ms trial grid.

ii. 
```python
times = np.asarray(cam["times"], dtype=np.float64)
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
```

```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. The trajectory justifies the camera-side choice, but not extra signal processing; once the left-camera trace was accepted, the agent carried it forward as a raw continuous signal on the shared grid (steps 244, 260).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, the AI pools whisker-motion-energy samples across all converted sessions and trials, computes global 1/3 and 2/3 quantiles, and digitizes each trial trace into `0`, `1`, and `2`.

ii. 
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
```

```python
output_trial = np.vstack(
    [
        np.full(NBINS, choice_out, dtype=np.int8),
        np.full(NBINS, prior_out, dtype=np.int8),
        discretize(wheel_vals, wheel_edges),
        discretize(whisker_vals, whisker_edges),
    ]
)
```

iii. The trajectory does not record a separate rationale for global whisker quantiles. As with wheel speed, the choice is implicit in the pooled final-assembly code.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned by extracting the left-camera motion-energy trace over `[stimOn - 0.5, stimOn + 1.5]` and interpolating it onto the same 100 trial bins used for the neural data, again at bin right edges.

ii. 
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The trajectory consistently frames wheel, whisker, and neural data as sharing one stimulus-aligned grid (steps 36, 107, 244).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data are mostly dropped. Trials are skipped when required trial fields are missing, choice/prior values are unsupported, wheel or whisker traces do not cover the window well enough for interpolation, or neural activity is all zero after binning. Sessions are skipped when wheel loading fails, no left-camera whisker trace is available, no good units survive, or fewer than two trials remain. The script also includes a reprocessing path that reuses an existing pickle for post-hoc region filtering instead of rerunning the raw conversion.

ii. 
```python
except Exception as exc:
    print(f"Skip session {eid}: wheel load failed: {exc}")
    return None
...
if whisker_times is None or whisker_me is None:
    print(f"Skip session {eid}: no whisker motion energy trace available.")
    return None
```

```python
if wheel_interp is None or whisker_interp is None:
    continue
...
if len(kept_trial_indices) < 2:
    return None
...
if nonzero_trial_mask.sum() < 2:
    return None
```

iii. The trajectory shows three explicit justifications here: use session skipping rather than brittle recovery when core streams are absent (steps 240, 285, 336), add zero-neural-trial dropping after verification exposed silent trials (step 151), and reprocess the finished pickle for region curation instead of spending hours on a second raw-data crawl (steps 508, 529, 531).

## 10-a. What are the most time-consuming steps of the code?

i. The main expensive steps are probe-level spike-data loading from disk/network cache and per-trial spike binning across all sessions. The later full-pickle reprocessing step is much cheaper and was introduced specifically to avoid repeating the raw-data crawl.

ii. 
```python
spikes_times_path = one.load_dataset(..., "spikes.times.npy", ...)
spikes_clusters_path = one.load_dataset(..., "spikes.clusters.npy", ...)
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

iii. The trajectory repeatedly describes spike-side loads as the bottleneck, including one mistaken broad download path (step 59), multi-hour full conversion monitoring (steps 188-199), and the choice to do fast post-hoc reprocessing instead of another three-hour raw rerun (steps 508, 529).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial interpolation path for wheel and whisker traces, the per-trial spike-binning loop in `bin_probe_spikes`, and some of the Python loops that rebuild region and subject indices during assembly and region filtering.

ii. 
```python
for trial_idx in np.flatnonzero(base_mask):
    ...
    wheel_interp = interpolate_trial_signal(...)
    whisker_interp = interpolate_trial_signal(...)
```

```python
for trial_idx, start_time in enumerate(trial_starts):
    ...
    np.add.at(...)
```

```python
for sess in session_results:
    ...
    for neuron_idx, region_name in enumerate(sess["brain_regions"]):
        ...
```

iii. The trajectory shows the agent noticing at least one avoidable scan and patching it (step 137), but it otherwise kept explicit session/trial loops because it wanted a converter that was easy to audit when sessions dropped out (steps 109, 114, 170).

## 10-c. What processing does the code repeat multiple times?

i. The code repeats the same trial-window search/interpolation logic separately for wheel and whisker traces, iterates over trials once to collect behavioral outputs and again to bin neural data, and performs a second whole-dataset region-remapping/filtering pass after already assembling the initial dataset.

ii. 
```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

```python
for trial_idx in np.flatnonzero(base_mask):
    ...
```

```python
data = build_data_from_session_results(session_results, release_df, args)
data, filter_stats = apply_region_filters(data)
```

iii. The trajectory makes the biggest repeated-processing decision explicit: after discovering a region-count mismatch, the agent added a second-pass reprocessing path over the completed pickle rather than integrating the final region curation into the original raw-conversion path (steps 508, 529, 531).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores continuous per-trial wheel-speed and whisker traces even though the final exported outputs keep only discretized categories; it also first assembles fine-grained region labels and then discards many neurons and even some sessions during `apply_region_filters`. The region-remap/post-filter path means part of the first-pass region assembly is temporary work.

ii. 
```python
wheel_trials.append(wheel_interp)
whisker_trials.append(whisker_interp)
...
output_data.append(build_output_trials(sess, wheel_edges, whisker_edges))
```

```python
session_regions.extend(probe.cluster_regions.tolist())
...
data, filter_stats = apply_region_filters(data)
```

iii. The trajectory directly acknowledges this for the region pipeline: the agent discovered the region mismatch only after a complete first pass, then chose to reprocess the saved pickle with Beryl remapping and region-count filters instead of redoing the raw conversion (steps 504, 508, 529).
