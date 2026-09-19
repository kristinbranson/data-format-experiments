# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the BWM release CSV (`bwm_release.csv`) with 699 probe insertions / 459 sessions as the session source. It connects to the ONE API pointed at the local cache under `/app/data`, then iterates through sessions grouped by `eid` from the CSV. Individual datasets (spikes, clusters, trials, wheel, camera) are loaded via `one.load_object` and `one.load_dataset` with explicit collection paths and spike sorting revision `2024-05-06`.

ii.
```python
release_df = pd.read_csv(args.release_csv)
# ...
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir=args.cache_dir,
    silent=True,
)
# ...
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The agent chose to use the BWM release CSV as the authoritative list of sessions and probes, and the ONE API for resolving local file paths. The agent reasoned: "The release metadata under `2025_Q3_IBL_et_al_BWM` lines up with the paper's 459-session public release."

## 1-b. How are the data split into subjects?

i. The subject name comes from the `subject` column of the release CSV. Sessions are grouped by `eid`, and the subject is extracted from the first row of each group. At assembly, subjects are collected into a list with an index mapping.

ii.
```python
subject = str(session_rows["subject"].iloc[0])
# ...
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
```

iii. The release CSV already contains subject information per probe insertion, so no derivation was needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the release CSV. The CSV is grouped by `eid`, and each group (potentially containing multiple probes) constitutes one session.

ii.
```python
grouped = list(release_df.groupby("eid", sort=False))
for session_idx, (eid, session_rows) in enumerate(grouped, start=1):
    result = process_session(one, session_rows, id_to_acronym, grey_ids)
```

iii. The release CSV naturally organizes data by session (eid), so no splitting decision was required.

## 1-d. How are the data split into trials?

i. Trials are loaded from the ALF trials object via `one.load_object(eid, "trials", collection="alf")`, which returns a DataFrame with one row per trial.

ii.
```python
trials = one.load_object(eid, "trials", collection="alf").to_df()
```

iii. The trials table already has one row per trial, so no splitting decision was required.

## 1-e. How are trials filtered based on quality controls?

i. Multiple filters are applied: (1) Non-NaN values required for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`. (2) Reaction time (`firstMovement_times - stimOn_times`) must be in [0.08, 2.0] seconds. (3) Exclude no-choice trials (`choice == 0`). (4) Trial length filter: `feedback_times - goCue_times <= 10.0` seconds. (5) Wheel and whisker motion energy must span the trial window (checked via `interpolate_trial_signal` returning None). (6) Trials with all-zero neural activity are dropped post-binning.

ii.
```python
TRIAL_NAN_EXCLUDE = (
    "stimOn_times", "choice", "feedback_times",
    "probabilityLeft", "firstMovement_times", "feedbackType",
)

def make_base_trial_mask(trials):
    mask = np.ones(len(trials), dtype=bool)
    for col in TRIAL_NAN_EXCLUDE:
        mask &= trials[col].notna().to_numpy()
    reaction_time = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    mask &= reaction_time >= 0.08
    mask &= reaction_time <= 2.0
    mask &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials.columns:
        go_cue = trials["goCue_times"].to_numpy()
        feedback = trials["feedback_times"].to_numpy()
        trial_len_ok = np.ones(len(trials), dtype=bool)
        good_len = np.isfinite(go_cue) & np.isfinite(feedback)
        trial_len_ok[good_len] = (feedback[good_len] - go_cue[good_len]) <= 10.0
        mask &= trial_len_ok
    return mask
```

Also, post-binning filter for all-zero neural trials:
```python
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The agent stated: "I'm extracting the exact trial filtering, behavior loading, and session metadata logic so the converter matches those decisions." The reaction time bounds and no-choice exclusion match the reference code's `ibl_data_utils.py`. The additional NaN checks on `feedback_times`/`feedbackType`, the trial length filter, and the all-zero neural trial filter are extra steps not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` and `spikes.clusters.npy`, plus `clusters.metrics.pqt` for quality labels, `clusters.channels.npy` for channel assignments, and `channels.brainLocationIds_ccf_2017.npy` for anatomical region IDs.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The agent reasoned: "I'm switching to per-dataset loads so the converter only requests `spikes.times`, `spikes.clusters`, `clusters.metrics`, and channel metadata."

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2-second trial window using `np.add.at`. The counts are stored as `uint16` and then cast to `float16`. Spikes are NOT converted to firing rates (no division by bin width). When a session has multiple probes, their units are pooled into one population with continuous indexing.

ii.
```python
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
in_bounds = (bins >= 0) & (bins < NBINS) & (trial_clusters >= 0)
np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
# ...
neural_trials = [session_counts[idx].astype(np.float16, copy=True) for idx in range(session_counts.shape[0])]
```

iii. The agent's notes state "Spike counts are binned into non-overlapping 20 ms bins" without discussion of rate conversion. No explicit reasoning was found about converting counts to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters are applied to clusters: (1) `clusters.metrics.label >= 1.0` (well-isolated units). (2) Units must be in grey matter (by tracing ancestry to region ID 8 via the Alyx API). (3) Units with acronyms in `{"void", "root", "grey"}` are excluded. Additionally, after Beryl remapping, two more filters are applied: regions must have >= 5 neurons per session-region, and regions must be present in >= 2 sessions globally.

ii.
```python
INVALID_REGION_ACRONYMS = {"void", "root", "grey"}
MIN_NEURONS_PER_SESSION_REGION = 5
MIN_SESSIONS_PER_REGION = 2

good = metrics["label"].to_numpy(dtype=float) >= GOOD_LABEL_THRESHOLD
good &= np.isin(region_ids, list(grey_ids))
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))
```

Post-processing region filter:
```python
valid_regions = set(unique_names[counts >= MIN_NEURONS_PER_SESSION_REGION].tolist())
globally_valid_regions = {
    region_name for region_name, session_ids in sessions_with_region.items()
    if len(session_ids) >= MIN_SESSIONS_PER_REGION
}
```

iii. The agent reasoned: "The reference code maps cluster acronyms to Beryl regions, and the papers add two more region curation rules: keep only regions with at least 5 well-isolated neurons within a session and present in at least 2 sessions." The agent also noted that excluding `root` follows from the region filtering logic, though the reference code keeps `root` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike times are aligned to stimulus onset by computing `trial_starts = stimOn_times + WINDOW_START`. The binning then counts spikes relative to this start time.

ii.
```python
trial_starts = trials.iloc[kept_trial_indices]["stimOn_times"].to_numpy(dtype=np.float64) + WINDOW_START
# ...
trial_times = kept_times[start_idx:end_idx]
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)
```

iii. The agent confirmed that spike times and trial event times share a common session clock, so alignment is achieved by subtracting the appropriate offset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The 2-second window is divided into 100 bins of 20 ms each. No rebinning or interpolation is applied to neural data.

ii.
```python
WINDOW_START = -0.5
WINDOW_END = 1.5
BINSIZE = 0.02
NBINS = int(round((WINDOW_END - WINDOW_START) / BINSIZE))
```

iii. The agent stated that the 20 ms bin size and 2 s window match the reference code and papers.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time values are computed as a fixed grid relative to stimulus onset.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The time grid is derived from the window parameters and bin size, not from raw data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as the right edges of each bin: `WINDOW_START + BINSIZE * arange(1, NBINS+1)`, giving values from -0.48 to 1.50 in 0.02 steps.

ii.
```python
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. The agent's notes describe this as "linear interpolation to stimulus-aligned 20 ms bin right edges." No explicit reasoning was found debating bin centers vs. bin edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values represent the right edges of each bin, while the neural data bins spikes using `np.floor((spike_time - start_time) / BINSIZE)`. This means the time input represents the end of each bin rather than the center.

ii.
```python
# Neural binning (left-edge assignment):
bins = np.floor((trial_times - start_time) / BINSIZE).astype(np.int16)

# Time input (right edges):
relative_time = WINDOW_START + BINSIZE * np.arange(1, NBINS + 1, dtype=np.float32)
```

iii. No explicit reasoning was found discussing the alignment between bin edges and neural bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From the `probabilityLeft` column of the trials table, which is constant within a block. A change in value signals a new block.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return trial_num
    current = 1
    trial_num[0] = current
    for idx in range(1, len(prob_left)):
        if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
            current += 1
        else:
            current = 1
        trial_num[idx] = current
    return trial_num
```

iii. The agent implemented this directly from the task specification, treating it as a sequential counter within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The counter starts at 1 for the first trial of each block and increments by 1 for each subsequent trial with the same `probabilityLeft` value. When `probabilityLeft` changes, the counter resets to 1. This is computed on all trials before filtering, so filtered trials still count toward the block position.

ii.
```python
current = 1
trial_num[0] = current
for idx in range(1, len(prob_left)):
    if np.isfinite(prev) and np.isfinite(curr) and curr == prev:
        current += 1
    else:
        current = 1
    trial_num[idx] = current
```

iii. Computed before trial filtering so the count reflects the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

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

iii. The agent verified the sign convention: "Verified choice sign convention from easy trials: high-contrast left trials are mostly `choice == 1`; high-contrast right trials are mostly `choice == -1`."

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values are recoded: +1 (left) -> 0, -1 (right) -> 1. Trials with choice == 0 (no response) are excluded.

ii.
```python
if choice_val == 1:
    choice_out = 0
elif choice_val == -1:
    choice_out = 1
else:
    continue
```

iii. Follows the task specification: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

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

iii. Follows the task specification: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping of the three probability values to integers 0, 1, 2. Trials with other values are excluded.

ii. Same as 6-a.

iii. No further processing.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw wheel position and timestamps, loaded via `one.load_object(eid, "wheel", collection="alf")`.

ii.
```python
wheel = one.load_object(eid, "wheel", collection="alf")
timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
position = np.asarray(wheel["position"], dtype=np.float64)
```

iii. The agent traced the wheel processing to match the reference implementation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) The raw wheel position is interpolated onto a 1000 Hz grid using `brainbox.behavior.wheel.interpolate_position`. (2) Velocity is computed using a Butterworth low-pass filter at 20 Hz via `velocity_filtered`. (3) Speed is the absolute value of velocity. The speed trace is then linearly interpolated onto the trial time grid.

ii.
```python
interp_pos, interp_time = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
speed = np.abs(np.asarray(velocity, dtype=np.float32))
```

iii. The agent noted: "The reusable upstream wheel functions import cleanly once I add the bundled `ibllib` tree to `PYTHONPATH`, so I can mirror the wheel preprocessing instead of approximating it."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using **global** tertile edges (1/3 and 2/3 quantiles) computed across all trials in the entire converted dataset, not per-session.

ii.
```python
wheel_all = np.concatenate([np.concatenate(sess["wheel_trials"]) for sess in session_results]).astype(np.float32)
wheel_edges = tuple(np.quantile(wheel_all, [1/3, 2/3]).astype(np.float32).tolist())
# ...
def discretize(values, edges):
    low_edge, high_edge = edges
    return np.digitize(values, bins=np.array([low_edge, high_edge], dtype=np.float32), right=False).astype(np.int8)
```

iii. The agent's notes describe this as "discretized into 3 bins using global tertiles over the converted dataset."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is linearly interpolated onto the same time grid as the neural data using `np.interp`. The sample times are the right edges of each bin: `start_time + binsize * arange(1, NBINS+1)`.

ii.
```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. The agent used linear interpolation to align the wheel trace to the neural binning grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the left camera only: `leftCamera.ROIMotionEnergy` and `leftCamera.times`, loaded via `one.load_object`.

ii.
```python
cam = one.load_object(eid, "leftCamera", attribute=["times", "ROIMotionEnergy"], collection="alf")
times = np.asarray(cam["times"], dtype=np.float64)
values = np.asarray(cam["ROIMotionEnergy"], dtype=np.float32)
```

iii. The agent initially used left-then-right fallback but corrected to left-only: "The paper text's `60 Hz` whisker signal description points to the left camera specifically."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no additional filtering or normalization. It is linearly interpolated onto the trial time grid (right bin edges).

ii.
```python
whisker_interp = interpolate_trial_signal(whisker_times, whisker_me, start_time, end_time)
```

iii. No additional processing beyond interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using **global** tertile edges across all trials in the entire dataset.

ii.
```python
whisker_all = np.concatenate([np.concatenate(sess["whisker_trials"]) for sess in session_results]).astype(np.float32)
whisker_edges = tuple(np.quantile(whisker_all, [1/3, 2/3]).astype(np.float32).tolist())
```

iii. Same global discretization approach as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: linearly interpolated onto the right bin edges of the neural time grid.

ii.
```python
sample_times = start_time + binsize * np.arange(1, NBINS + 1, dtype=np.float64)
interp_vals = np.interp(sample_times, times, values)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple safeguards: (1) Trials with NaN in key columns are excluded. (2) Trials where wheel or whisker interpolation fails (returns None) are dropped. (3) Probes with no good clusters are skipped. (4) Sessions with fewer than 2 valid trials are skipped. (5) Sessions with no whisker motion energy are skipped. (6) Trials with all-zero neural activity are dropped. (7) Sessions losing all neurons after region filtering are dropped.

ii.
```python
if wheel_interp is None or whisker_interp is None:
    continue
# ...
if len(kept_trial_indices) < 2:
    print(f"Skip session {eid}: only {len(kept_trial_indices)} trials...")
    return None
# ...
nonzero_trial_mask = session_counts.reshape(session_counts.shape[0], -1).sum(axis=1) > 0
```

iii. The agent stated: "A trial whose selected neurons are all silent across the full 2 s window. I'm filtering those zero-information trials out in the converter so the final exports are clean."

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk, particularly the large `spikes.times.npy` and `spikes.clusters.npy` arrays. The agent uses memory-mapped loading (`mmap_mode="r"`) to mitigate this. Additionally, building region maps via `one.alyx.rest("brain-regions", "list")` at startup is a network-dependent step.

ii.
```python
spike_times = np.load(spikes_times_path, mmap_mode="r")
spike_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

iii. The cost is mainly file I/O for the large spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop iterates over each trial individually with `np.add.at`. This could potentially be vectorized by offsetting spike indices across trials. The per-trial behavioral signal interpolation loop could also be vectorized.

ii.
```python
for trial_idx, start_time in enumerate(trial_starts):
    # ... per-trial spike binning
    np.add.at(probe_counts[trial_idx], (trial_clusters[in_bounds], bins[in_bounds]), 1)
```

```python
for trial_idx in np.flatnonzero(base_mask):
    # ... per-trial wheel/whisker interpolation
    wheel_interp = interpolate_trial_signal(wheel_times, wheel_speed, start_time, end_time)
```

iii. No explicit discussion of vectorization optimization.

## 10-c. What processing does the code repeat multiple times?

i. The region filtering with Beryl remapping is applied twice: once during initial probe loading (in `load_probe_info` where `INVALID_REGION_ACRONYMS` are excluded) and again in `apply_region_filters` (which re-applies Beryl mapping and further filters by min neurons/sessions). The spike data is also loaded separately for cluster info and for actual spike binning.

ii.
```python
# First in load_probe_info:
good &= ~np.isin(region_acronyms, list(INVALID_REGION_ACRONYMS))

# Then in apply_region_filters:
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
invalid = np.isin(session_beryl, ["root", "void"])
```

iii. The agent added the `apply_region_filters` step later when discovering the region count didn't match the reference, leading to some redundancy.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds a detailed `build_region_maps` function that queries the Alyx API for brain region ancestry to determine grey matter membership, which is then partially redundant with the `iblatlas.BrainRegions` Beryl mapping applied later. The `grey_ids` filtering is done at probe loading time but then the same regions are re-evaluated during `apply_region_filters`. The code also collects and stores extensive metadata (lab, date, probe names, trial counts) that may not be used downstream.

ii.
```python
records = one.alyx.rest("brain-regions", "list", no_cache=True)
# ... builds grey_ids via ancestry traversal
# Later, iblatlas is used independently:
brainreg = BrainRegions()
session_beryl = np.asarray(brainreg.acronym2acronym(session_old_names, mapping=REGION_MAPPING), dtype=object)
```

iii. The redundancy arose from iterative development where the agent first implemented grey-matter filtering via the Alyx API, then later added the Beryl-based region filtering as a correction.
