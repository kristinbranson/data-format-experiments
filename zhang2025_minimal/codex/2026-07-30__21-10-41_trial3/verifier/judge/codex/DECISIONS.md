# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter reads the 2025 BWM release table, groups it by session `eid`, opens a ONE client against `/app/data`, and then loads each session’s trial table, wheel object, whisker-motion-energy trace, and each probe’s spike-sorting files. It uses the release CSV to decide which sessions/probes exist, and then uses ONE/ALF files to materialize the actual data.

ii.
```python
parser.add_argument(
    "--release-csv",
    type=str,
    default="/app/code/code_zhang2025/data/bwm_release.csv",
)

one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir=args.cache_dir,
    silent=True,
)

release_df = pd.read_csv(args.release_csv)
grouped = list(release_df.groupby("eid", sort=False))

trials = load_trials(one, eid)
wheel_times, wheel_speed = load_wheel_speed(one, eid)
camera_view, whisker_times, whisker_me = load_whisker_motion_energy(one, eid)
probe = load_probe_info(one, eid, str(row["probe_name"]), id_to_acronym, grey_ids)
```

iii. The justification in `CONVERSION_NOTES.md` is that the 2025 `bwm_release.csv` matches the public release totals in the data paper (459 sessions, 699 insertions, 139 subjects, 12 labs), so it used that file as the authoritative session/probe list and then loaded per-session ALF data through ONE.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the release CSV/session metadata. Each converted session carries a `subject` string, and the final export builds a unique `subjects` list plus a per-session `subject_idx`.

ii.
```python
eid = str(session_rows["eid"].iloc[0])
subject = str(session_rows["subject"].iloc[0])

subjects = []
subject_to_idx = {}

for sess in session_results:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The agent did not give a separate deep justification beyond following the release table; the notes say the release CSV matches paper-level subject counts and is therefore the source of subject identity.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `eid` values from the BWM release CSV. The script groups the CSV by `eid`, processes one grouped session at a time, and emits one session entry each in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
release_df = pd.read_csv(args.release_csv)
release_df["eid"] = release_df["eid"].astype(str)
grouped = list(release_df.groupby("eid", sort=False))

for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
    if result is None:
        continue
    session_results.append(result)
```

iii. The notes justify this by saying the release table is the session/probe source and that it matches the paper’s release statistics.

## 1-d. How are the data split into trials?

i. Trials are rows in the session trial table. The script first computes a base trial mask on the full table, then iterates over the valid trial indices, creates a fixed stimulus-aligned window for each trial, and keeps only trials whose wheel and whisker traces can be interpolated on that window. After neural binning, it may drop additional trials whose neural activity is all zeros.

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
    kept_trial_indices.append(trial_idx)

nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
if not np.all(nonzero_trial_mask):
    session_counts = session_counts[nonzero_trial_mask]
```

iii. The notes say this was intended to mirror the reference code’s trial mask and the reference utility’s alignment logic for dynamic behaviors.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by required non-NaN fields, reaction time in `[0.08, 2.0]` s, exclusion of no-choice trials, and `feedback_times - goCue_times <= 10.0` when available. After that, the script further excludes trials lacking usable interpolated wheel or whisker data in the alignment window, and then removes trials with all-zero neural counts.

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

if wheel_interp is None or whisker_interp is None:
    continue
...
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. `CONVERSION_NOTES.md` explicitly says this was meant to match the trial exclusions in the papers plus the reference code path, including the `<= 10 s` trial-length criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike times and spike-cluster assignments from each probe, together with cluster metrics and channel-to-brain-region mappings used for unit and region filtering.

ii.
```python
metrics_path = one.load_dataset(
    eid, "clusters.metrics.pqt", collection=collection, revision=SPIKE_SORTING_REVISION
)
cluster_channels = np.asarray(
    one.load_dataset(eid, "clusters.channels.npy", collection=collection, revision=SPIKE_SORTING_REVISION)
)
channel_region_ids = np.asarray(
    one.load_dataset(
        eid, "channels.brainLocationIds_ccf_2017.npy", collection=collection, revision=SPIKE_SORTING_REVISION
    )
)

spikes_times_path = one.load_dataset(
    eid, "spikes.times.npy", collection=probe.collection, revision=SPIKE_SORTING_REVISION
)
spikes_clusters_path = one.load_dataset(
    eid, "spikes.clusters.npy", collection=probe.collection, revision=SPIKE_SORTING_REVISION
)
```

iii. The notes justify this as using the same spike-sorting outputs and atlas metadata that the reference code and papers rely on.

## 2-b. How is the `neural` data processed?

i. The script merges probes within a session, filters clusters, bins spikes into non-overlapping 20 ms bins over a `[-0.5, 1.5]` s stimulus-aligned window for each kept trial, concatenates neurons across probes, and stores each trial as a `(n_neurons, 100)` matrix.

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

session_counts[:, offset:end_offset, :] = probe_counts
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The notes say this matches the reference idea of session-level probe merging and non-overlapping temporal spike-count bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only clusters with `metrics["label"] >= 1`, restricts them to grey-matter regions, excludes `void`, `root`, and `grey` acronyms, and then applies an additional Beryl-region filter requiring at least 5 neurons in a session-region and at least 2 sessions per region.

ii.
```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
good_cluster_ids = np.flatnonzero(good).astype(np.int32)

session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
unique_names, counts = np.unique(valid_names, return_counts=True)
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
...
globally_valid_regions = {
    region_name for region_name, session_ids in sessions_with_region.items()
    if len(session_ids) >= MIN_SESSIONS_PER_REGION
}
```

iii. `CONVERSION_NOTES.md` says this was chosen to match the paper’s well-isolated-neuron and grey-matter region criteria, plus the reference code’s Beryl atlas mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to stimulus onset. For each kept trial, the neural window begins at `stimOn_times - 0.5` s and ends at `stimOn_times + 1.5` s, and spikes are assigned to bins relative to that window start.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5

stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END

trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The notes explicitly say the common alignment event is stimulus onset, following the task’s decoder specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins. There are 100 bins per 2 s trial, and the script directly bins spikes at that resolution rather than first binning at another resolution and rebining.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))

bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
probe_counts = np.zeros((len(trial_starts), len(probe.good_cluster_ids), NBINS), dtype=np.uint16)
```

iii. The notes justify this with the reference code path and the methods summary that uses 2 s trials with 20 ms bins in the cached dataset.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw variable. It is constructed from the decision to align each trial to `stimOn_times`, together with the fixed window and bin-size constants.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The notes justify it as a decoder input defined by the task specification rather than by a pre-existing ALF field.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a single 100-element relative-time vector at the right edge of each 20 ms bin, then copies that same vector into every kept trial as the first input channel.

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

iii. The only explicit rationale is that this gives a time-varying decoder input aligned to the same bins as the neural data.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100-bin trial grid as the spike counts and the interpolated behavioral traces: same window, same bin size, same right-edge sampling convention.

ii.
```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The notes explicitly say that continuous traces are interpolated onto the right edge of each 20 ms bin, and the relative-time input follows that same convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` sequence in the trials table.

ii.
```python
block_trial_num = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=float))
...
trial_num = float(block_trial_num[trial_idx])
```

iii. The agent did not cite a paper/code source for this exact construction. It appears to be an inferred feature built from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script walks the `probabilityLeft` sequence in order, increments a counter while the probability stays the same, and resets it to 1 whenever the probability changes or becomes non-finite. The resulting scalar is then repeated across all time bins of the trial.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.zeros(len(prob_left), dtype=np.float32)
    current = 1
    trial_num[0] = current
    for idx in range(1, len(prob_left)):
        prev = prob_left[idx - 1]
        curr = prob_left[idx]
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
        trial_num[idx] = current
```

iii. There is no explicit justification in the notes beyond the need to create the requested decoder input; this is the agent’s own derived implementation.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the trial-table `choice` column.

ii.
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
```

iii. The notes explicitly identify IBL `choice` as the source variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw IBL sign convention is remapped so that `choice == 1` becomes exported class `0` (`left`) and `choice == -1` becomes exported class `1` (`right`). Any other value is skipped. The class is then repeated across all 100 bins of the trial.

ii.
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue

np.full(NBINS, choice_out, dtype=np.int8)
```

iii. The notes justify this with the decoder-task requirement `left = 0, right = 1`, plus a sanity check on easy left/right trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trial-table `probabilityLeft` column.

ii.
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. The notes explicitly identify `probabilityLeft` as the source variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw probabilities are discretized exactly as required by the task: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. Trials with any other value are skipped. The resulting class is repeated across all 100 time bins.

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

np.full(NBINS, prior_out, dtype=np.int8)
```

iii. The notes say this mapping follows the decoder-task specification exactly.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the raw wheel `timestamps` and `position` arrays loaded from the session’s wheel object.

ii.
```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The notes explicitly state that wheel speed is loaded from the wheel timestamps and position files.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is linearly interpolated to 1000 Hz, a Butterworth-filtered velocity is computed, and speed is defined as the absolute value of that velocity.

ii.
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

iii. The notes say this matches `brainbox.behavior.wheel.velocity_filtered`, which is also the wheel-processing path used by the reference `SessionLoader`.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The script concatenates all kept wheel-speed samples across the converted dataset, computes the global 1/3 and 2/3 quantiles, and then uses those two edges to assign `low`, `medium`, and `high` categories with `np.digitize`.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())

def discretize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The notes justify this only at the level of satisfying the task’s requirement for 3 bins; they describe the bins as global tertiles over the converted dataset.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each kept trial, wheel speed is interpolated onto the same 100 stimulus-aligned 20 ms bins used for the neural data. The interpolation samples the right edge of each bin in the window `stimOn - 0.5 s` to `stimOn + 1.5 s`.

ii.
```python
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
start_time = stim_on + WINDOW_START
end_time = stim_on + WINDOW_END
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)

sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The notes explicitly say the continuous traces are linearly interpolated onto the right edge of each 20 ms trial bin.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived only from the left camera’s `times` and `ROIMotionEnergy` arrays. If those are missing, the session is skipped.

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

iii. The notes explicitly say the agent decided to use the left camera only.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Beyond loading the left-camera ROI motion-energy trace, the script does not compute motion energy itself. It takes the stored `ROIMotionEnergy` values as the continuous signal and later linearly interpolates them onto the trial grid.

ii.
```python
camera_view, whisker_times, whisker_me = load_whisker_motion_energy(one, eid)
...
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. The notes justify this as reusing the already-computed ALF motion-energy output rather than recomputing from video frames.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it is globally discretized into tertiles across the converted dataset using the 1/3 and 2/3 quantiles of all kept whisker-motion-energy samples.

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
...
discretize(whisker_vals, whisker_edges)
```

iii. The notes describe these as global tertile bins chosen to satisfy the required 3-category output.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. For sessions that have a left-camera whisker trace, it is linearly interpolated onto the same 100 stimulus-aligned 20 ms bins used for the neural data. Sessions without a usable left-camera whisker trace are skipped entirely.

ii.
```python
camera_view, whisker_times, whisker_me = load_whisker_motion_energy(one, eid)
if whisker_times is None or whisker_me is None:
    print(f"Skip session {eid}: no whisker motion energy trace available.")
    return None

whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. The notes justify the interpolation convention, and separately note that the agent intentionally used the left camera only.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly handles missing or malformed data by exclusion rather than repair. It reconstructs `intervals_0`/`intervals_1` if only a combined `intervals` field exists, but otherwise it skips sessions when wheel or whisker loading fails, skips trials if interpolation coverage is inadequate or values are non-finite, and drops sessions/trials with no valid neurons or no nonzero neural activity.

ii.
```python
if "intervals_0" not in trials.columns and "intervals" in trials.columns:
    intervals = np.asarray(trials["intervals"].to_list())
    if intervals.ndim == 2 and intervals.shape[1] == 2:
        trials["intervals_0"] = intervals[:, 0]
        trials["intervals_1"] = intervals[:, 1]

except Exception as exc:
    print(f"Skip session {eid}: wheel load failed: {exc}")
    return None

finite = np.isfinite(times) & np.isfinite(values)
times = times[finite]
values = values[finite]
if len(times) < 2:
    return None
```

iii. The notes frame this as matching the reference utilities’ behavior of dropping unusable trials/sessions after alignment checks, rather than imputing missing data.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large per-probe spike arrays and then repeatedly looping over trials to bin spikes for each probe. A secondary cost is per-trial interpolation of wheel and whisker traces across every valid trial.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")

for trial_idx, start_time in enumerate(trial_starts):
    ...
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)

for trial_idx in np.flatnonzero(base_mask):
    ...
    wheel_interp = interpolate_trial_signal(...)
    whisker_interp = interpolate_trial_signal(...)
```

iii. There is no explicit self-justification from the agent here; this follows directly from the structure of the implemented code.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial spike-binning loop in `bin_probe_spikes`, the per-trial behavior-construction loop in `process_session`, the `compute_trial_number_in_block` loop over the `probabilityLeft` sequence, and the per-neuron/per-session loops used to build region and subject index arrays.

ii.
```python
for idx in range(1, len(prob_left)):
    ...

for trial_idx, start_time in enumerate(trial_starts):
    ...

for trial_idx in np.flatnonzero(base_mask):
    ...

for neuron_idx, region_name in enumerate(sess["brain_regions"]):
    ...
```

iii. The agent did not explicitly discuss vectorization in its notes; this is an evident consequence of the implementation it chose.

## 12-c. What processing does the code repeat multiple times?

i. It repeats the same search/interpolation logic for wheel and whisker on every valid trial, loops over all kept spikes separately for every probe, and makes an additional full post-processing pass to remap/filter brain regions after the initial conversion is already complete.

ii.
```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)

for probe in probes:
    probe_counts = bin_probe_spikes(one, eid, probe, trial_starts)
    ...

data = build_data_from_session_results(session_results, release_df, args)
data, filter_stats = apply_region_filters(data)
```

iii. The trajectory shows the agent later added the extra Beryl remap/filter pass to correct region curation, which made this repeated processing an intentional post hoc fix.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It keeps continuous wheel-speed and whisker-motion-energy traces for every trial even though the final dataset only stores their discretized versions; it also first builds fine-grained region labels and only later remaps them to Beryl and throws many neurons/regions away. Some trials are fully behavior-processed before being discarded for all-zero neural activity.

ii.
```python
wheel_trials.append(wheel_interp)
whisker_trials.append(whisker_interp)
...
output_data.append(build_output_trials(sess, wheel_edges, whisker_edges))

session_regions.extend(probe.cluster_regions.tolist())
...
data, filter_stats = apply_region_filters(data)

if not np.all(nonzero_trial_mask):
    session_counts = session_counts[nonzero_trial_mask]
```

iii. The trajectory and notes show this was partly deliberate: the agent first converted with fine-grained labels, then added a second-pass Beryl filter to better match the paper’s region criteria.
