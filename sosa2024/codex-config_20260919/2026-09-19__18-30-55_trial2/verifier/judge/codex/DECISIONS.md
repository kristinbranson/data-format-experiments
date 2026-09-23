# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `*_behavior+ophys.nwb` file under `/app/data/sub-*`, sorted them by subject and session number parsed from the filename, and opened each file with `h5py`. Within each session it read the synchronized behavior group and the ophys fluorescence/neuropil datasets directly from the NWB HDF5 hierarchy. It also supported a `--sample` mode that truncates to the first two files.

ii. ```python
def discover_files(sample: bool) -> list[str]:
    files = sorted(
        glob.glob(str(DATA_ROOT / "sub-*" / "*_behavior+ophys.nwb")),
        key=_session_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    return files[:2] if sample else files

with h5py.File(path, "r") as nwb:
    behavior = nwb["processing/behavior/BehavioralTimeSeries"]
    neural_group = nwb["processing/ophys"]
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented that the dataset contains 152 NWB files organized as one file per subject/day session, and in Step 5 it states session order is subject then numeric day.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file and then deduplicated/sorted into the dataset-level `subjects` list. Per-session subject identity is read from `general/subject/subject_id` and converted into `subject_idx`.

ii. ```python
subject = _decode(nwb["general/subject/subject_id"])
...
subjects = sorted({x["subject"] for x in converted}, key=lambda s: int(s[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
"subject_idx": np.asarray(
    [subject_lookup[x["subject"]] for x in converted], dtype=np.int64
),
```

iii. The notes say the provided files are organized by subject subdirectories and that session order should preserve subject/day provenance.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are ordered by subject number and session/day number parsed from the filename.

ii. ```python
def _session_sort_key(path: str) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", os.path.basename(path))
    ...
    return int(match.group(1)), int(match.group(2))

with h5py.File(path, "r") as nwb:
    day = int(_decode(nwb["general/session_id"]))
    session_id = f"{subject}_ses-{day:02d}"
```

iii. Step 2 of the notes explicitly records one NWB file per subject/day session and Step 5 says session order is subject then numeric session day.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing each positive `trial_start` sample with the corresponding positive `teleport` sample, then slicing each stream on `[start:stop)`. The code validates equal counts and requires every `teleport` index to occur after its paired start.

ii. ```python
starts = np.flatnonzero(dense["trial_start"] > 0)
teleports = np.flatnonzero(dense["teleport"] > 0)
if len(starts) != len(teleports) or not np.all(teleports > starts):
    raise ValueError(f"Unpaired or reversed trial bounds in {session_id}")
...
for i, (start, stop) in enumerate(zip(starts, teleports)):
    fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
```

iii. In Step 2/4 notes the agent says paired `trial_start` to `teleport` rows define the on-track interval and that slicing `[start:stop)` excludes teleport/ITI while matching the paper after accounting for indexing conventions.

## 1-e. How are trials filtered based on quality controls?

i. The agent did not use the reference solution’s `<50`-sample cutoff. Instead it dropped entire trials if more than 30% of their frames had lick counts above 2 (`lick_bad`), matching the paper’s lick-sensor exclusion, and required at least two retained trials per session.

ii. ```python
LICK_ERROR_FRACTION = 0.30
...
lick_bad = np.array([
    np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, teleports)
])
...
if lick_bad[i]:
    continue
...
if len(kept_events) < 2:
    raise ValueError(f"Fewer than two retained trials in {session_id}")
```

iii. Step 4 and Step 5 of `CONVERSION_NOTES.md` explicitly say the repository snapshot uses 35% but the published paper uses >30%, and that 81 flagged trials are reproduced exactly at 30%; the trajectory repeats that rationale at step 47 and step 58.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved neural signal is derived from raw `Fluorescence/plane0/data` and `Neuropil/plane0/data` plus the manual ROI curation fields in `ImageSegmentation/PlaneSegmentation`. The agent explicitly did not use the NWB `Deconvolved` series.

ii. ```python
segmentation = neural_group["ImageSegmentation/PlaneSegmentation"]
plane_index = np.asarray(segmentation["planeIdx"][:])
iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
plane0_iscell = iscell[plane_index == 0]
...
fluorescence_ds = neural_group["Fluorescence/plane0/data"]
neuropil_ds = neural_group["Neuropil/plane0/data"]
```

iii. The module docstring and Step 4/5 notes say the conversion intentionally starts from raw ROI and neuropil fluorescence because NWB `Deconvolved` is a different processing stage than the one analyzed in the paper.

## 2-b. How is the `neural` data processed?

i. For each trial, the agent subtracted `0.7 * neuropil`, added back the trial mean neuropil term, built a maximin baseline with a 15-sample Gaussian plus 300-sample min/max filters, computed dF/F, smoothed dF/F with a 2-sample Gaussian, and OASIS-deconvolved it with `tau=0.7`. The important divergence is that this was done trial-by-trial, with no session-level `keep_teleports` handling.

ii. ```python
def compute_dff_and_events(fluorescence, neuropil, compute_events=True):
    corrected = fluorescence - NEUROPIL_COEF * neuropil
    corrected += NEUROPIL_COEF * np.mean(neuropil, axis=1, keepdims=True)
    baseline_seed = gaussian_filter(
        corrected, sigma=(0.0, BASELINE_SMOOTH_SIGMA), mode="reflect"
    )
    baseline = minimum_filter1d(baseline_seed, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect")
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect")
    dff_unsmoothed = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff_unsmoothed, DFF_SMOOTH_SIGMA, axis=1)
    events = dcnv.oasis(np.asarray(dff, dtype=np.float32), 2000, OASIS_TAU_SECONDS, FRAME_RATE_HZ)
```

iii. The notes repeatedly justify recomputing the paper’s own dF/F and OASIS events, but the final code simplified this to purely per-trial processing; there is no implementation of the reference `teleport_sessions` logic described in the human reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: keep only manually curated `iscell` ROIs from plane 0, then remove putative interneurons whose dF/F-speed correlation exceeds `0.5`.

ii. ```python
plane0_iscell = iscell[plane_index == 0]
roi_columns = np.flatnonzero(plane0_iscell)
...
speed_corr = _finalize_corr(corr_sums)
is_interneuron = np.isfinite(speed_corr) & (speed_corr > INTERNEURON_SPEED_R)
neuron_keep = ~is_interneuron
kept_events = [np.asarray(x[neuron_keep], dtype=np.float32) for x in kept_events]
```

iii. Step 5 of the notes says the neuron curation rule is manual `iscell` plus the published dF/F-speed correlation exclusion, and explicitly says place-cell/remapping-class filters are not applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by extracting the same `[trial_start:teleport)` interval used to define each trial. No additional temporal shifting is applied because the requested alignment event is trial start itself.

ii. ```python
for i, (start, stop) in enumerate(zip(starts, teleports)):
    fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
    ...
    kept_events.append(events)
...
"temporal_alignment_event": "entry into the 450 cm track (trial_start)",
"off_start": 0.0,
```

iii. Step 5 notes say alignment is “trial-start entry to the linear track” and that rows `[trial_start, teleport)` are included.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent used a fixed sampling rate of `15.5078125 Hz`, i.e. `64.483627 ms` per bin, and performed no temporal rebinning or interpolation.

ii. ```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
if not np.allclose(np.diff(timestamps), 1.0 / FRAME_RATE_HZ, atol=1e-9):
    raise ValueError(f"Nonuniform timestamps in {session_id}")
...
"time_bin_size": TIME_BIN_MS,
```

iii. Step 2 and Step 4 notes emphasize that dense behavior timestamps are uniformly spaced at 0.064483627204 s and that no temporal resampling is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the common dense behavior timestamps, specifically `position/timestamps`, after the agent verified that all dense behavior series share the same timestamps.

ii. ```python
timestamps = np.asarray(behavior["position/timestamps"][:common_length])
if not all(
    np.allclose(behavior[name]["timestamps"][:common_length], timestamps, rtol=0.0, atol=1e-9)
    for name in dense_names
):
    raise ValueError(f"Dense behavior timestamps differ in {session_id}")
```

iii. Step 2 notes say dense behavior timestamps are identical across streams and are the authoritative sampling axis.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp is subtracted so time starts at 0 seconds and then increases in native frame steps.

ii. ```python
input_data = np.vstack((
    timestamps[start:stop] - timestamps[start],
    np.full(T, env_values[0]),
    np.full(T, raw_trial),
    np.full(T, outcomes[i - 1] if i > 0 else 0),
)).astype(np.float32)
```

iii. The mapping table in Step 5 says “timestamp[s:e] - timestamp[s], float32 seconds; time-varying, begins at exactly 0”.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by using the same trial start/stop indices and the same common frame count as the neural data. Neural and behavior streams are first truncated to `common_length`.

ii. ```python
common_length = min(
    fluorescence_ds.shape[0],
    *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
)
...
T = stop - start
input_data = np.vstack((timestamps[start:stop] - timestamps[start], ...))
```

iii. The notes emphasize that dense behavior and neural rows are already synchronized and that the conversion should keep that row-aligned axis.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the dense behavior `environment` series.

ii. ```python
dense_names = [
    "position", "speed", "lick", "environment", "trial number",
    "reward_zone", "trial_start", "teleport",
]
...
env_values = np.unique(dense["environment"][start:stop])
```

iii. Step 5 of the notes maps dense `environment` directly to the decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent required the environment to be constant and binary within each trial, then repeated that single value across all trial timepoints.

ii. ```python
env_values = np.unique(dense["environment"][start:stop])
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
...
np.full(T, env_values[0])
```

iii. Step 5 notes describe environment as a per-trial variable: “Confirm a single 0/1 in each trial, then repeat it across trial frames.”

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The agent derived trial number from the raw dense `trial number` variable sampled at each trial start, rather than from the trial-loop index.

ii. ```python
raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
if len(np.unique(raw_trial_numbers)) != len(raw_trial_numbers):
    raise ValueError(f"Repeated raw trial numbers in {session_id}")
...
raw_trial = int(raw_trial_numbers[i])
```

iii. Step 5 notes say to “Read at start row and repeat original zero-based value across frames” and explicitly “do not renumber after exclusions.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. After reading the raw trial number at the trial start, the agent repeats that scalar across all timepoints in the trial.

ii. ```python
input_data = np.vstack((
    timestamps[start:stop] - timestamps[start],
    np.full(T, env_values[0]),
    np.full(T, raw_trial),
    np.full(T, outcomes[i - 1] if i > 0 else 0),
)).astype(np.float32)
```

iii. The notes describe trial number as a per-trial continuous input that should not be renumbered after exclusions.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome comes from per-trial reward outcomes computed from `Reward/timestamps`, dense trial timestamps, and the dense `reward_zone` signal indicating zone entry within the trial.

ii. ```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
```

iii. Step 4 and Step 5 notes say reward outcome should come from sparse reward timestamps plus a zone-entry check, matching the paper’s start-to-teleport logic.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes `outcomes` for all raw trials, then for each kept trial copies `outcomes[i-1]` across time; the first trial gets 0. Because outcomes are computed before bad-lick exclusions, the “previous trial” remains the actual previous raw trial.

ii. ```python
outcomes = _reward_outcomes(...)
...
input_data = np.vstack((
    timestamps[start:stop] - timestamps[start],
    np.full(T, env_values[0]),
    np.full(T, raw_trial),
    np.full(T, outcomes[i - 1] if i > 0 else 0),
)).astype(np.float32)
```

iii. Step 5 notes make this explicit: “Previous original trial outcome, repeated over frames; first raw trial gets 0. A dropped bad-lick trial still counts as the actual previous trial.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from dense `position` and a per-trial zone identity inferred from the session `identifier`/scene string plus the raw trial number. The dense `reward_zone` series is not used to define zone identity for this output.

ii. ```python
scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
labels = parse_scene_zones(scene)
raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
...
zone_label = zone_for_trial(labels, raw_trial)
zone_start, zone_end = ZONE_BOUNDS[zone_label]
distance = position - np.clip(position, zone_start, zone_end)
```

iii. Step 4/5 notes say scene parsing plus the known switch-at-trial-30 schedule was validated against the raw reward-zone entries and chosen over noisier direct use of the dense `reward_zone` values.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The signed continuous distance is computed as point-to-interval distance: negative before the zone, exactly zero inside it, and positive after it, via `position - clip(position, zone_start, zone_end)`.

ii. ```python
zone_start, zone_end = ZONE_BOUNDS[zone_label]
distance = position - np.clip(position, zone_start, zone_end)
```

iii. Step 5 notes define distance exactly this way and justify it as the most direct implementation of “distance to any location in the reward zone.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The agent used explicit inequality-based thresholds for the 7 requested categories: `<-50`, `[-50,-10)`, `[-10,0)`, `==0`, `(0,10]`, `(10,50]`, `>50`.

ii. ```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    out[distance > 50.0] = 6
```

iii. Step 5 notes explicitly list these boundaries and say the inequalities were chosen to obey the task wording exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance on the same trial slice and same frame count as the saved neural events.

ii. ```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
T = stop - start
...
output_data = np.vstack((
    distance_classes(distance),
    position_classes(position),
    speed_classes(speed),
    ...
)).astype(np.int8)
if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
    raise AssertionError(...)
```

iii. The notes repeatedly describe the conversion as frame-aligned and verify equal `T` across neural/input/output.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the dense behavior `position` series.

ii. ```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. Step 5 maps dense `position` directly to absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. There is no transformation beyond slicing the trial interval and discretizing position; the agent does not clip out-of-range values before binning.

ii. ```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
...
position_classes(position)
```

iii. The notes say absolute position should be the continuous on-track position sampled on the trial-start axis, then categorized into 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm track is divided with explicit inequalities into 5 bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`.

ii. ```python
def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    out[position > 360.0] = 4
```

iii. Step 5 notes say the explicit inequalities were chosen because the task wording treats 360 cm as belonging to the fourth bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position is aligned simply by taking the same `[start:stop)` frames used for neural activity and storing one position class per neural time bin.

ii. ```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
...
if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
    raise AssertionError(...)
```

iii. The notes describe all time-varying outputs as living on the same frame axis as the neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the dense behavior `lick` series.

ii. ```python
lick = np.asarray(dense["lick"][start:stop])
```

iii. Step 5 maps dense cumulative lick counts to the lick output after QC.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After dropping bad lick trials, the raw lick count is binarized framewise as `(lick > 0)`.

ii. ```python
output_data = np.vstack((
    ...
    (lick > 0).astype(np.int8),
    ...
)).astype(np.int8)
```

iii. Step 5 notes explicitly say “Binarize valid lick counts (`count > 0`).”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the same trial interval and storing one binary lick value per neural frame.

ii. ```python
lick = np.asarray(dense["lick"][start:stop])
T = stop - start
...
if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
    raise AssertionError(...)
```

iii. The notes treat lick as a frame-aligned time-varying output on the common 64.48 ms axis.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB session `identifier` scene string plus the raw trial number, not from the dense `reward_zone` activity directly.

ii. ```python
scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
labels = parse_scene_zones(scene)
raw_trial = int(raw_trial_numbers[i])
zone_label = zone_for_trial(labels, raw_trial)
```

iii. Step 4/5 notes say the scene/switch schedule was validated against raw zone-entry data and chosen as the authoritative source of per-trial reward-zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed for one or two chronological zone labels, and if there are two labels the code switches from the first to the second once `raw_trial_number >= 30`. The result is repeated across all frames of the trial as class A/B/C = 0/1/2.

ii. ```python
def parse_scene_zones(scene: str) -> list[str]:
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    ...

def zone_for_trial(labels: list[str], raw_trial_number: int) -> str:
    if len(labels) == 1 or raw_trial_number < 30:
        return labels[0]
    return labels[1]
...
np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8)
```

iii. Step 5 notes call out the switch-trial-30 rule explicitly and say it was validated against the raw reward-zone entries.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from sparse `Reward/timestamps`, dense trial timestamps, trial boundaries, and the dense `reward_zone` activity used to require a zone-entry event.

ii. ```python
outcomes = _reward_outcomes(
    timestamps,
    starts,
    teleports,
    np.asarray(behavior["Reward/timestamps"][:]),
    dense["reward_zone"],
)
```

iii. Step 4/5 notes say reward outcome should be defined as a reward event occurring within `[start, teleport)` together with zone entry, ignoring the three sparse reward events that fall outside paired on-track trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code uses `searchsorted` on reward timestamps to check whether any reward occurred between the trial’s start and teleport timestamps and also requires `reward_zone > 0` somewhere in the trial. The resulting binary outcome is repeated across the trial.

ii. ```python
def _reward_outcomes(...):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
...
np.full(T, outcomes[i], dtype=np.int8)
```

iii. The notes justify this as matching the reference start-to-teleport reward logic while ignoring three out-of-trial sparse reward events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handled several issues explicitly: it truncated neural and dense behavior streams to a shared `common_length`, verified all dense timestamps are identical and uniformly sampled, ignored sparse reward events outside any paired trial, dropped bad-lick trials, rejected sessions with malformed trial bounds or too few retained trials, and replaced nonfinite OASIS outputs with zero.

ii. ```python
common_length = min(
    fluorescence_ds.shape[0],
    *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
)
...
events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
...
if len(starts) != len(teleports) or not np.all(teleports > starts):
    raise ValueError(...)
if len(kept_events) < 2:
    raise ValueError(...)
```

iii. Step 2/4/5 notes document the one-extra-neural-frame issue in 10 sessions, the 3 out-of-trial reward timestamps, and the bad-lick trials; the code turns those findings into concrete handling rules.

## 13-a. What are the most time-consuming steps of the code?

i. The agent’s notes say the heavy steps are contiguous HDF5 trial-block reads plus per-trial dF/F/OASIS processing across all sessions; the code is structured to process one session at a time to control memory.

ii. ```python
for i, path in enumerate(files):
    print(f"Converting session {i + 1}/{len(files)}: {path}", flush=True)
    converted.append(convert_session(path, show_processing and i < 2))
...
for i, (start, stop) in enumerate(zip(starts, teleports)):
    fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
    ...
    dff, events, processing_trace = compute_dff_and_events(...)
```

iii. Step 6 and Step 7 notes say contiguous NWB reads and OASIS must happen trial-by-trial and are the main runtime drivers; runtime estimates in the trajectory also focused on full-session conversion cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate loops are the per-trial `_reward_outcomes` loop, the `lick_bad` list comprehension, and the large per-trial conversion loop. Some bookkeeping could be vectorized, but trial-wise OASIS on variable-length windows remains the hard sequential part.

ii. ```python
lick_bad = np.array([
    np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, teleports)
])
...
for i, (start, stop) in enumerate(zip(starts, teleports)):
    ...

def _reward_outcomes(...):
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        ...
```

iii. The notes explicitly discuss optimizing I/O and correlation accumulation, but they also say OASIS “must operate trial-by-trial to match the reference.”

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial boundaries: once to label bad-lick trials, again to compute outcomes, and again to build trial-level neural/input/output arrays. It also always computes and returns a full `processing_trace`, even though that trace is usually not used downstream.

ii. ```python
outcomes = _reward_outcomes(...)
lick_bad = np.array([
    np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, teleports)
])
...
for i, (start, stop) in enumerate(zip(starts, teleports)):
    dff, events, processing_trace = compute_dff_and_events(...)
```
```python
trace = {
    "corrected": corrected,
    "baseline": baseline,
    "dff_unsmoothed": dff_unsmoothed,
    "dff": dff,
}
return dff, events, trace
```

iii. This follows directly from the code structure; unlike the human reference, the repetition is intra-session rather than a separate survey/conversion pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is that `compute_dff_and_events` always builds and returns `processing_trace` (`corrected`, `baseline`, unsmoothed dF/F, smoothed dF/F) even though the saved dataset keeps only `events` and only the first plotted trial ever uses the trace. The optional plot payload and diagnostic plot generation are also discarded by downstream decoder training.

ii. ```python
trace = {
    "corrected": corrected,
    "baseline": baseline,
    "dff_unsmoothed": dff_unsmoothed,
    "dff": dff,
}
return dff, events, trace
...
if show_processing and plot_payload is None:
    plot_payload = {
        "processing": {k: v[example_neuron].copy() for k, v in processing_trace.items()},
        ...
    }
```

iii. This is visible directly in the AI code and was not highlighted in the human reference because that solution answered “N/A” here.
