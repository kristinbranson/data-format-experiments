# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the 699-insertion BWM release CSV, groups its rows by session `eid`, and uses ONE against `/app/data` to load trials, wheel, left-camera motion energy, probe metadata, and spike arrays. It processes the 459 listed sessions sequentially; sessions lacking usable required data are skipped.

ii.
```python
release_df = pd.read_csv(args.release_csv)
grouped = list(release_df.groupby("eid", sort=False))
one = ONE(base_url="https://openalyx.internationalbrainlab.org", cache_dir=args.cache_dir, silent=True)
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The trajectory says the 2025 BWM release corresponds to the paper's 459-session public release. The agent chose ONE so cached files could be resolved locally and missing ALF files fetched, and later reported 433 retained sessions after data and region filtering.

## 1-b. How are the data split into subjects?

i. Subject labels come directly from the release CSV. During assembly, first occurrence order defines `subjects`, and each retained session receives the corresponding `subject_idx`; after region filtering, unused subjects are removed and indices remapped.

ii.
```python
subject = str(session_rows["subject"].iloc[0])
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The trajectory treats the release inventory as authoritative session/subject metadata and does not infer subjects from file paths.

## 1-c. How are the data split into sessions?

i. Rows in the release CSV are grouped by `eid`; all probe rows for an `eid` are processed as one session and their neurons are concatenated.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The agent identified `eid` as the release's session identifier and designed explicit per-session processing to make dropped sessions auditable.

## 1-d. How are the data split into trials?

i. The ALF trials object is converted to a DataFrame with one row per trial. Retained row indices define trial windows and corresponding per-trial neural, input, and output arrays.

ii.
```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
for trial_idx in np.flatnonzero(base_mask):
    stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
```

iii. The trajectory relied on the existing trials table rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have nonmissing stimulus, choice, feedback, prior, first movement and feedback type; reaction time must be 0.08–2 s; choice must be nonzero; and, where available, feedback minus go cue must be at most 10 s. Both behavioral traces must cover the window with finite interpolation, and trials with no retained spikes anywhere are subsequently removed. Sessions need at least two retained trials.

ii.
```python
for col in TRIAL_NAN_EXCLUDE:
    mask &= trials[col].notna().to_numpy()
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= trials["choice"].to_numpy() != 0
trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
...
if wheel_interp is None or whisker_interp is None:
    continue
...
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The agent explicitly traced the reference trial filtering and cited the 80 ms–2 s mask and no-choice exclusion. Coverage checks prevent incomplete aligned behavior, while all-zero neural trials were added after the verifier warned about a zero-information trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `spikes.times.npy` and `spikes.clusters.npy`. Cluster metrics, cluster-to-channel assignments, and channel CCF region IDs determine which clusters are retained and their brain regions.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
metrics = pd.read_parquet(metrics_path)
cluster_channels = np.asarray(one.load_dataset(eid, "clusters.channels.npy", ...))
```

iii. The trajectory says per-dataset loading was chosen to avoid unnecessarily broad probe downloads and to request only spike times, assignments, metrics, and channel metadata.

## 2-b. How is the `neural` data processed?

i. For each probe, retained spikes are assigned to 20 ms bins in each two-second trial window using `np.add.at`. Probe matrices are concatenated along the neuron axis. The saved values are raw spike counts cast to `float16`; they are not divided by bin width and are not smoothed.

ii.
```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
np.add.at(probe_counts[trial_idx],
          (trial_clusters[in_bounds], bins[in_bounds]), 1)
...
session_counts[:, offset:end_offset, :] = probe_counts
neural_trials = [session_counts[idx].astype(np.float16, copy=True) ...]
```

iii. The agent chose explicit trial-wise binning for auditability and memory-mapped the large spike arrays. The trajectory discusses well-isolated units and practical pickle/training size, but does not justify retaining counts instead of the reference firing-rate conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters require metrics label at least 1, a region descended from grey matter, and an acronym other than `void`, `root`, or `grey`. Regions are later mapped to Beryl; neurons are retained only in session-regions with at least five neurons and regions present in at least two sessions. Sessions emptied by this filter are dropped.

ii.
```python
good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
...
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
...
if len(session_ids) >= MIN_SESSIONS_PER_REGION
```

iii. The trajectory says label ≥1 identifies well-isolated units. After an initially inflated region count, the agent interpreted the paper/reference as requiring Beryl mapping, ≥5 neurons per session-region, and presence in ≥2 sessions, and reprocessed the pickle accordingly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins 0.5 s before `stimOn_times` and ends 1.5 s after it. Absolute spike timestamps are sliced into that window and binned relative to the window start, placing stimulus onset at 0.5 s (bin boundary 25).

ii.
```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(...) + WINDOW_START
end_time = start_time + (WINDOW_END - WINDOW_START)
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The trajectory explicitly selected the reference-supported stimulus-aligned window of −0.5 to +1.5 s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 100 nonoverlapping 20 ms bins over two seconds. Spikes are counted directly into those bins; there is no smoothing or later neural rebinning.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The agent states that a 2 s, stimulus-aligned, 20 ms-binned representation is supported by the reference text and suitable for the common decoder representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a constructed relative-time grid based on `stimOn_times`, the −0.5/+1.5 s window, and 20 ms bins. The stored grid runs from −0.48 through +1.50 s at bin right edges.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
stim_on = float(trials.iloc[trial_idx]["stimOn_times"])
```

iii. The agent chose one common stimulus-aligned representation for static and dynamic decoder variables. The trajectory does not give a specific reason for using right edges rather than bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw signal is transformed. A fixed arithmetic progression is generated once per session and copied into every retained trial's first input row.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
input_trial = np.vstack([relative_time, np.full(NBINS, trial_num, dtype=np.float32)])
```

iii. It is defined from the selected window/bin parameters; no additional justification appears in the trajectory.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has 100 entries corresponding one-for-one to the 100 neural bins, but labels them by their right edges. Thus neural bin 0 covers [−0.5, −0.48) and receives time −0.48, rather than its center −0.49.

ii.
```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The agent intended all streams to share a single grid; the right-edge convention is implicit rather than explained.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full unfiltered `trials.probabilityLeft` sequence. Equal consecutive priors remain in one block; a changed or nonfinite prior begins another.

ii.
```python
block_trial_num = compute_trial_number_in_block(
    trials["probabilityLeft"].to_numpy(dtype=float))
```

iii. The agent followed the idea that blocks are identified by changes in the block prior and computes the number before trial filtering so removed trials still count.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop counts consecutive equal prior values, starting every block at 1. The scalar is then broadcast across all 100 time points of that trial.

ii.
```python
current = 1
trial_num[0] = current
...
if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
    current += 1
else:
    current = 1
...
np.full(NBINS, trial_num, dtype=np.float32)
```

iii. The trajectory does not discuss the 1-based convention. Computing before filtering preserves actual experimental position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`, where +1 is left and −1 is right; zero-choice trials are excluded.

ii.
```python
choice_val = float(trials.iloc[trial_idx]["choice"])
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
```

iii. The agent checked the sign convention from easy-trial behavior and concluded that +1 means left and −1 right, matching the required left=0/right=1 mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Values are recoded to 0/1 and broadcast as an `int8` row across all 100 time bins.

ii.
```python
np.full(NBINS, choice_out, dtype=np.int8)
```

iii. The recoding follows the decoder specification; broadcasting gives a common time-varying output matrix shape.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from each trial's `probabilityLeft` field.

ii.
```python
prob_left = float(trials.iloc[trial_idx]["probabilityLeft"])
```

iii. The agent treats the trials table as the authoritative source for the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are recoded to categories 0, 1, and 2 and broadcast across the 100 time bins.

ii.
```python
if prob_left == 0.2: prior_out = 0
elif prob_left == 0.5: prior_out = 1
elif prob_left == 0.8: prior_out = 2
...
np.full(NBINS, prior_out, dtype=np.int8)
```

iii. This is the mapping explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed derives from the ALF wheel `timestamps` and `position` arrays.

ii.
```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The trajectory says the bundled upstream IBL wheel functions were used to mirror, rather than approximate, reference preprocessing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1000 Hz, converted to an eighth-order 20 Hz Butterworth-filtered velocity, made nonnegative with absolute value, and linearly interpolated to each trial's 20 ms right-edge grid. It is then discretized using global dataset quantiles.

ii.
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The agent deliberately reused IBL's wheel utilities. A common grid was chosen for all dynamic variables; no trajectory justification is given for dataset-global rather than per-session discretization.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel samples from all sessions/trials are concatenated. Their 1/3 and 2/3 quantiles define two global thresholds; `np.digitize` yields low/medium/high categories 0/1/2.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results])
wheel_edges = tuple(np.quantile(wheel_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
return np.digitize(values, bins=np.array([low_edge, high_edge]), right=False).astype(np.int8)
```

iii. The trajectory does not explain the scope of the quantiles; it appears intended to make three comparably populated categories across the complete converted dataset.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each stimulus-aligned trial, wheel speed is interpolated at absolute times from stimulus−0.48 through stimulus+1.50 s, one sample per neural bin. These are neural-bin right edges, not centers.

ii.
```python
start_time = stim_on + WINDOW_START
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The agent intended a bin-for-bin common representation; using right edges is not separately justified.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the left camera's ALF `times` and `ROIMotionEnergy` arrays only. Sessions without a valid left-camera trace are dropped; there is no right-camera fallback.

ii.
```python
cam = one.load_object(eid, "leftCamera",
    attribute=["times", "ROIMotionEnergy"], collection="alf")
times = np.asarray(cam["times"], dtype=np.float64)
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
```

iii. The agent initially considered/followed fallback behavior but deliberately switched to left-only because the paper's 60 Hz whisker description appeared to refer specifically to the left camera and this brought the cohort close to 433 sessions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is not filtered or normalized. Finite values are linearly interpolated to the same trial right-edge grid and later discretized with global dataset quantiles.

ii.
```python
finite = np.isfinite(times) & np.isfinite(values)
times = times[finite]; values = values[finite]
interp_vals = np.interp(sample_times, times, values)
```

iii. The agent used released ROI motion energy directly and chose the shared stimulus-aligned grid. No extra motion-energy transformation was considered necessary.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. All retained whisker samples from every session are concatenated, global 1/3 and 2/3 quantiles are calculated, and `np.digitize` assigns categories 0/1/2.

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results])
whisker_edges = tuple(np.quantile(whisker_all, [1 / 3, 2 / 3]).astype(np.float32).tolist())
discretize(whisker_vals, whisker_edges)
```

iii. As for wheel speed, the trajectory provides no explicit rationale for global thresholds; the apparent aim is globally balanced low/medium/high classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated at the same 100 right-edge timestamps used for wheel and the time input, relative to the same `stimOn_times`. Each value corresponds positionally to one neural bin, though not to its center.

ii.
```python
start_time = stim_on + WINDOW_START
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. The common trial grid was an explicit design goal. The right-edge choice is implicit.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Required trial NaNs are filtered. Missing wheel or left-camera session data cause the whole session to be skipped. A trace with fewer than two finite samples, missing edge coverage, or nonfinite interpolation causes that trial to be skipped. Probes with no good units are ignored; sessions with no probe, fewer than two trials, no retained region, or fewer than two nonzero neural trials are skipped. Unexpected session errors are caught and skipped.

ii.
```python
except Exception as exc:
    print(f"Skip session {eid}: wheel load failed: {exc}")
    return None
...
if len(times) < 2: return None
if abs(start_time - times[0]) > binsize: return None
...
except Exception as exc:
    print(f"Skip session {eid}: unexpected error: {exc}")
    result = None
```

iii. The agent prioritized a clean verifier result and auditable skip messages. Missing whisker energy explains most session exclusions; zero-spike trial removal was prompted by a verifier warning.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/downloading the large per-probe spike arrays and iterating through every session/probe/trial to bin them dominate runtime. The full raw conversion took roughly three hours; loading and rewriting the roughly 6 GB pickle for region reprocessing was another substantial step.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
for trial_idx, start_time in enumerate(trial_starts):
    ...
    np.add.at(probe_counts[trial_idx], ..., 1)
```

iii. The trajectory identifies broad probe loading/downloads and avoidable spike-array scans as bottlenecks, switches to minimal per-dataset requests/memory maps, and avoids rerunning raw conversion by postprocessing the existing pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over eligible trials that interpolates behavior/builds inputs, the per-probe trial spike-binning loop, the block-number loop, and neuron-by-neuron region-index construction could all be partly vectorized. Session/probe loops are less readily eliminated because files and neuron counts differ.

ii.
```python
for idx in range(1, len(prob_left)):
    ...
for trial_idx in np.flatnonzero(base_mask):
    ...
for trial_idx, start_time in enumerate(trial_starts):
    ...
for neuron_idx, region_name in enumerate(sess["brain_regions"]):
    ...
```

iii. The trajectory says explicit per-session/per-trial processing was retained for auditability. It did optimize one forced full-array scan by using cluster metadata length instead.

## 10-c. What processing does the code repeat multiple times?

i. Trial-window lookup and interpolation are independently repeated for wheel and whisker. Output construction repeatedly fills static choice/prior arrays. Region acronym mapping/index rebuilding occurs during initial construction and again in `apply_region_filters`; the delivered pickle was explicitly reprocessed through that latter path.

ii.
```python
wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
...
np.full(NBINS, choice_out, dtype=np.int8)
np.full(NBINS, prior_out, dtype=np.int8)
```

iii. The shared interpolation helper favors consistency. The second region pass was intentional: after discovering the 472-region mismatch, the agent remapped and filtered the completed pickle rather than repeating the expensive raw crawl.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/converts interval columns that are never used, stores release/session metadata not consumed by decoding, creates full 100-point copies of static trial number/choice/prior, initially maps detailed regions that are later remapped to Beryl, and retains continuous wheel/whisker traces until global thresholds are computed even though only categorical outputs are saved.

ii.
```python
if "intervals_0" not in trials.columns and "intervals" in trials.columns:
    intervals = np.asarray(trials["intervals"].to_list())
...
np.full(NBINS, trial_num, dtype=np.float32)
...
session_counts[:, offset:end_offset, :] = probe_counts
```

iii. The trajectory emphasizes reproducibility metadata, a uniform decoder matrix shape, and the need for all continuous samples to compute global quantiles. Detailed-region work became redundant only after the late Beryl-curation correction.
