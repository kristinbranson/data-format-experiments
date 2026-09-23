# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the frozen release manifest `bwm_release.csv`, groups its 699 probe rows into 459 session specifications, constructs each cached session path, and locates trials, wheel, spike, cluster-channel, and channel-location files recursively. Arrays are memory-mapped where practical; sessions without usable behavior are later excluded.

ii.
```python
release = pd.read_csv(RELEASE_CSV, dtype={"date": str, "subject": str, "lab": str})
for eid, rows in release.groupby("eid", sort=False):
    session_dir = (DATA_ROOT / first["lab"] / "Subjects" / first["subject"] /
                   first["date"] / f"{number:03d}")
    ...
    trial_table=_find_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#")
```

iii. The notes justify this as using the exact data-paper freeze, reproducing 139 subjects, 459 sessions, 699 probes, and 621,733 clusters, while avoiding later additions and revision double-counting. Direct local paths and deterministic revision preferences were chosen instead of ONE.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the release CSV. Final subjects are the sorted unique subjects among retained sessions, and each session receives an integer lookup index.

ii.
```python
subjects = sorted({s["spec"].subject for s in sessions})
subject_lookup = {s: i for i, s in enumerate(subjects)}
"subject_idx": np.asarray([subject_lookup[s["spec"].subject] for s in sessions], dtype=np.int32)
```

iii. The release manifest supplies an unambiguous subject ID; the final list intentionally contains only subjects represented by retained sessions.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the frozen release is one session. All probes with that `eid` are combined into one `SessionSpec` and ultimately one decoder session.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    ...
    probes=tuple(probes)
```

iii. The notes state that simultaneous probes share behavior and should form one session, while grouping by `eid` prevents cached revisions from becoming duplicate sessions.

## 1-d. How are the data split into trials?

i. The trial parquet table already has one row per trial. Retained row indices select stimulus times and all per-trial variables; neural and behavior windows are then extracted around each selected row's stimulus onset.

ii.
```python
trials = pd.read_parquet(spec.trial_table)
raw_idx = np.flatnonzero(ref_mask)
stim = trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx]
```

iii. The agent treated source-table rows as the authoritative trials and saved the original retained row indices for auditability.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have nonmissing stimulus, choice, feedback, prior, first movement, and feedback type; reaction time must be 0.08–2 s; choice must be nonzero; feedback-minus-go-cue duration must not exceed 10 s. The entire two-second wheel and whisker windows must also have boundary coverage. Sessions with fewer than two survivors are excluded.

ii.
```python
mask = trials[required].notna().all(axis=1).to_numpy().copy()
mask &= (rt >= 0.08) & (rt <= 2.0)
mask &= ~(duration > 10.0)
mask &= trials["choice"].to_numpy() != 0
...
stream_good = wheel_good & motion_good
```

iii. The agent says the event/RT/no-choice/duration mask reproduces `load_trials_and_mask`; stream coverage is an added validity requirement needed to produce complete categorical behavior traces.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from every probe's `spikes.times.npy` and `spikes.clusters.npy`. `clusters.channels.npy` defines the cluster inventory and, with channel CCF IDs, supplies region labels.

ii.
```python
spike_times = np.load(probe.spikes_times, mmap_mode="r")
spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
cluster_channels = np.load(probe.clusters_channels, mmap_mode="r")
channel_ids = np.load(probe.channel_ids, mmap_mode="r")
```

iii. The notes identify spike time and assigned cluster as the quantities needed for binned activity, with channel anatomy used only for metadata.

## 2-b. How is the `neural` data processed?

i. For every retained trial and probe, spikes in the half-open `[-0.5, 1.5)` window are assigned to 20 ms bins and counted by flattened cluster×bin `bincount`. Probe cluster axes are concatenated. The stored float32 values remain spike counts; they are not divided by bin width or smoothed.

ii.
```python
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
flat = clusters[keep] * N_BINS + bins[keep]
counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
neural[j][offset:offset + n_clusters] = counts
```

iii. The agent argues that the supplied methods cache bins counts without smoothing and that downstream standardization belongs to decoder training, not conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by unit quality. Every Kilosort cluster in the frozen manifest is retained, including clusters mapped to `root` or `void`; quality metrics are not loaded.

ii.
```python
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
```

iii. The notes explicitly distinguish the data paper's 75,708 well-isolated units from the methods cache, which calls loading with `qc=None`; the agent chose the latter “all neurons” interpretation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. Absolute spike times are sliced from stimulus onset −0.5 s to +1.5 s and converted to bin indices relative to that beginning.

ii.
```python
stim = behavior["stim_times"]
begins, ends = stim + OFF_START, stim + OFF_END
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
```

iii. The agent selected the supplied cache's explicit stimulus-aligned two-second grid as the closest direct reference for the requested task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural window has 100 nonoverlapping 20 ms bins. Raw spike events are binned once; no smoothing or later temporal rebinning is applied. The coordinate exposed to the decoder is the right edge of each bin, −0.48 through +1.50 s.

ii.
```python
BIN_SIZE = 0.020
N_BINS = 100
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS)
```

iii. The notes cite the cache parameters and choose right-edge labels because the reference behavior interpolator labels bins that way.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured offsets and bin size around each trial's `stimOn_times`, rather than from a separate sampled raw variable.

ii.
```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

iii. The agent regards the time coordinate as the common stimulus-aligned bin grid defined by the reference cache.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed sequence of 100 bin-right-edge values from −0.48 to +1.50 seconds is generated and cast to float32 in each trial input.

ii.
```python
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. The notes say right-edge timestamps preserve the reference behavior sampler's convention.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `t` labels the right edge of neural count bin `t`; both are defined from the same stimulus onset and 20 ms interval grid.

ii.
```python
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
# paired with RELATIVE_BIN_ENDS in assemble_data
```

iii. The agent describes the coordinate as the reference bin-ending label for half-open neural intervals.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from changes in the full, unfiltered `trials.probabilityLeft` sequence.

ii.
```python
probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
block_no_all = trial_number_in_block(probs_all)
```

iii. Because no explicit block ID exists, the agent uses constant runs of prior probability as experimental blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Run boundaries are found wherever probability changes; each run is numbered from zero internally. Counting occurs before trial filtering, and the selected scalar is repeated across all 100 bins.

ii.
```python
starts = np.r_[0, np.flatnonzero(probability_left[1:] != probability_left[:-1]) + 1]
for start, end in zip(starts, ends):
    out[start:end] = np.arange(end - start, dtype=np.int32)
```

iii. The notes justify pre-filter counting so discarded trials do not compress the animal's actual position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from `trials.choice` for retained trials.

ii.
```python
choices = trials["choice"].to_numpy(dtype=np.float64)[raw_idx]
```

iii. The source choice column is treated as the authoritative per-trial response; no-choice trials have already been removed.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps source −1 to class 0 (“left”) and +1 to class 1 (“right”), casts to int8, and repeats the class over 100 time bins.

ii.
```python
"choice": (choices == 1).astype(np.int8)
...
np.full(N_BINS, s["choice"][j], dtype=np.int8)
```

iii. Its notes and metadata assert that IBL uses −1 for left and +1 for right, so the mapping was chosen to meet left=0/right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from each retained trial's `probabilityLeft` value.

ii.
```python
priors = probs_all[raw_idx]
```

iii. This field is already the task's block prior and therefore needs no inference from behavior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped in ascending order to 0, 1, and 2 using `searchsorted`, cast to int8, and repeated over time.

ii.
```python
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8)
```

iii. The mapping is explicitly required by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
wheel_raw_position = np.asarray(np.load(spec.wheel_position, mmap_mode="r"), dtype=np.float64)
```

iii. The agent follows the bundled IBL behavior functions rather than differentiating irregular samples directly.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz; velocity is computed with the bundled order-8, 20 Hz filtered method; absolute velocity gives speed; it is linearly sampled at trial bin ends (with reference-like final-bin extrapolation).

ii.
```python
wheel_position, wheel_times = interpolate_position(wheel_raw_times, wheel_raw_position, freq=1000)
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
wheel_speed = np.abs(wheel_velocity)
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. The notes identify these as the defaults used by `SessionLoader.load_wheel` and the supplied reference behavior code.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two global thresholds are the 1/3 and 2/3 quantiles over every retained aligned wheel sample in all sessions. `searchsorted(..., side="right")` produces low=0, medium=1, high=2.

ii.
```python
wheel_q = np.quantile(wheel_values, [1 / 3, 2 / 3])
np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8)
```

iii. The agent chose global thresholds to keep physical class definitions consistent across sessions while approximately balancing the full dataset.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at `stimOn_times + RELATIVE_BIN_ENDS`, one value for each corresponding neural bin.

ii.
```python
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. Both streams use the synchronized session clock and the same stimulus-relative grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses a paired `<view>Camera.ROIMotionEnergy.npy` and `_ibl_<view>Camera.times.npy`, preferring left and falling back to right.

ii.
```python
for view, revision in (("left", "#2025-05-29#"), ("right", "#2025-05-31#")):
    times = _optional_file(alf, f"_ibl_{view}Camera.times.npy")
    values = _optional_file(alf, f"{view}Camera.ROIMotionEnergy.npy", revision)
```

iii. The agent says this is the view policy used by the reference loader and avoids fabricating a stream when neither paired source exists.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are used without filtering or normalization and linearly interpolated to the 100 bin-end targets, including reference-like last-bin extrapolation.

ii.
```python
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. The notes report no additional reference transformation beyond alignment and categorical discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 1/3 and 2/3 quantiles across all retained aligned whisker samples in all sessions are global cut points; searchsorted yields classes 0, 1, and 2.

ii.
```python
whisker_q = np.quantile(whisker_values, [1 / 3, 2 / 3])
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8)
```

iii. As for wheel speed, the stated aim is a consistent dataset-wide scale with balanced categories.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is evaluated at each trial's stimulus onset plus the 100 neural bin-right-edge times.

ii.
```python
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. Camera times and spikes are assumed to be synchronized upstream on the session clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files, invalid/nonmonotonic/nonfinite behavior streams, insufficient trial counts, and processing exceptions exclude a session with a recorded reason. Trials missing required fields or complete behavior coverage are dropped. Unexpected choice/prior values and malformed spike arrays raise errors. Three retained all-zero neural windows caused by ephys ending before behavior were deliberately preserved.

ii.
```python
if motion is None:
    return None, "missing paired left and right whisker-motion streams"
...
except Exception as exc:
    behavior, reason = None, f"behavior processing error: {type(exc).__name__}: {exc}"
```

iii. The agent prioritizes complete mandatory outputs and auditable exclusions, while retaining zero neural windows because the reference defines no neural-coverage trial filter and raw checking confirmed they were not conversion errors.

## 10-a. What are the most time-consuming steps of the code?

i. Spike loading/slicing/binning across roughly 21 billion source events and writing the approximately 99 GiB dense pickle dominate. Behavior preparation and inventory construction are comparatively small.

ii.
```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    sessions = list(pool.map(bin_spikes_for_session, behavior_sessions))
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes give a full conversion time of 455.48 s and identify memory-mapped spike I/O and dense output writing as the costly work.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-session behavior loop, per-probe loop, per-trial spike loop, per-block loop, and final per-trial assembly loops could potentially be batched further. The important inner spike counting is already vectorized with `searchsorted` and `bincount`, and independent sessions are threaded.

ii.
```python
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    ...
    counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
```

iii. The agent says trial slicing avoids rescanning full spike arrays and makes the full conversion tractable; it favors bounded session parallelism over a more complex all-trial vectorization.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly locates files via recursive glob during manifest construction; loads some cluster/spike arrays again for inventory, binning, and the independent spot check; creates repeated static time, block, choice, and prior rows for every trial; and performs structurally similar wheel and whisker coverage/interpolation.

ii.
```python
candidates = sorted(p for p in root.glob(f"**/{name}") if p.is_file())
...
p0 = spec.probes[0]
st = np.load(p0.spikes_times, mmap_mode="r")
sc = np.load(p0.spikes_clusters, mmap_mode="r")
```

iii. The notes accept some repetition for independent sanity checks, straightforward stream handling, and deterministic revision resolution.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads wheel position and acceleration-related results even though only absolute velocity is retained; computes extensive inventory/session metadata and a direct spike spot check that the decoder does not use; and materializes constant per-trial variables across all 100 bins. Optional plotting also prepares raw captures used only for diagnostics.

ii.
```python
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
...
np.full(N_BINS, s["choice"][j], dtype=np.int8)
```

iii. These choices support validation, target shape consistency, and auditing, though they add work or storage that is not itself a learned decoder feature.
