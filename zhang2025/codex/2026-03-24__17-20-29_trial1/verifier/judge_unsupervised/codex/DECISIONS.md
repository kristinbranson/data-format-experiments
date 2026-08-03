# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads a session manifest parquet, resolves each manifest row into a session directory under `DATA_ROOT`, then recursively finds the needed ALF / pykilosort files inside each session with `pick_one_file()`. It processes sessions independently, optionally in parallel with a `ThreadPoolExecutor`. A notable implementation detail is that the checked-in script hardcodes `DATA_ROOT` to an external `/groups/...` path and leaves the project-local `/app/data/one_cache` path commented out.

ii. 
```python
DATA_ROOT = Path('/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/data-format/data/zhang2025/one_cache')
#DATA_ROOT = ROOT / "data" / "one_cache"
MANIFEST_FILES = [
    DATA_ROOT / "2025_Q3_IBL_et_al_BWM" / "sessions.pqt",
    DATA_ROOT / "Brainwidemap" / "sessions.pqt",
    DATA_ROOT / "2022_Q4_IBL_et_al_BWM" / "sessions.pqt",
]

def load_session_manifest() -> pd.DataFrame:
    for manifest in MANIFEST_FILES:
        if manifest.exists():
            return pd.read_parquet(manifest)

def resolve_session_specs() -> tuple[list[SessionSpec], list[str]]:
    manifest = load_session_manifest()
    ...
    for eid, row in manifest.iterrows():
        session_path = (
            DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
            / str(row["date"]) / f"{int(row['number']):03d}"
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose the `2025_Q3` manifest as the “canonical local 459-session release manifest,” and chose recursive version-robust file lookup because the cache contains versioned ALF folders such as `#2025-03-03#`.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the manifest/session metadata (`spec.subject`). After session processing, the script builds a sorted list of unique kept subjects and a per-session `subject_idx`.

ii. 
```python
return ProcessedSession(
    eid=spec.eid,
    subject=spec.subject,
    ...
)

subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. The notes say the mapping for `subjects` / `subject_idx` should come directly from release-session metadata and preserve deterministic session order.

## 1-c. How are the data split into sessions?

i. Each manifest row / `eid` is treated as one session. Sessions are only kept if the corresponding directory exists and the session survives all later preprocessing requirements.

ii. 
```python
for eid, row in manifest.iterrows():
    ...
    if not session_path.exists():
        missing.append(eid)
        continue
    specs.append(SessionSpec(eid=eid, ...))

for session in executor.map(process_session_worker, specs):
    if session is not None:
        processed.append(session)
```

iii. The agent’s notes say the “session-level units of analysis” should match the Zhang preprocessing pipeline, with probes merged within session before export.

## 1-d. How are the data split into trials?

i. Trials come from the per-session `_ibl_trials.table.pqt` file. The script first masks bad trial rows, then uses each surviving row’s `stimOn_times` as the alignment anchor. Neural and behavior are then sliced/interpolated trial-by-trial into one 2 s window per kept trial.

ii. 
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    return pd.read_parquet(trial_file)

masked_trials = trials.loc[trial_mask].reset_index(drop=False)
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_for_trials(..., align_times=align_times)
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. The notes explicitly tie this to the reference `load_trials_and_mask()` / `bin_spiking_data()` path and to the user-required stimulus-onset-aligned decoder format.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is multi-stage. First, `compute_trial_mask()` keeps only trials with required event columns present, reaction time in `[0.08, 2.0]` s, `choice != 0`, and `feedback_times - goCue_times <= 10 s`. After that, the script drops trials whose wheel or whisker stream cannot cover the full aligned window, and also drops trials whose entire binned neural matrix is zero. Sessions with fewer than 2 surviving trials are skipped.

ii. 
```python
required = [
    "stimOn_times", "choice", "feedback_times",
    "probabilityLeft", "firstMovement_times", "feedbackType",
]
rt = trials["firstMovement_times"] - trials["stimOn_times"]
mask &= rt >= TRIAL_MASK_RT[0]
mask &= rt <= TRIAL_MASK_RT[1]
mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
mask &= trials["choice"] != 0
for col in required:
    mask &= trials[col].notna().to_numpy()

combined_mask = wheel_mask & whisk_mask & neural_mask
if combined_mask.sum() < 2:
    return None
```

iii. The notes say this trial mask was chosen to match the reference trial curation, and later Step 10 says the `neural_mask = np.any(trial)` filter was added after verification exposed genuine all-zero spike windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is built from spike times and spike cluster IDs, with cluster QC labels and channel-to-brain-region metadata used for filtering and annotation.

ii. 
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

iii. The notes describe the intended mapping as “`spikes.times`, `spikes.clusters` from all probes in one session” plus `clusters.metrics.label` and region metadata.

## 2-b. How is the `neural` data processed?

i. The script merges probes within a session, filters to QC-passing clusters, remaps cluster IDs to a dense session-local index, converts channel brain-location IDs to acronyms, sorts spikes by time, and bins spike counts into per-trial `(n_neurons, 100)` matrices using 20 ms bins.

ii. 
```python
for probe_path in probe_paths:
    times, clusters, regions, total_here, good_here = load_probe_spikes_and_regions(...)
    merged_times.append(times)
    merged_clusters.append(clusters + cluster_offset)
    merged_regions.append(regions)
    cluster_offset += good_here

order = np.argsort(spike_times, kind="stable")
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]

counts, _, cluster_idx = bincount2D(
    spike_times[idx0:idx1],
    spike_clusters[idx0:idx1],
    xbin=binsize,
    xlim=[start, end],
)
trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```

iii. In the notes, the agent says this is meant to mirror the reference pattern “merge probes within session” and “bin spike counts into 20 ms bins over `[-0.5, 1.5]` s around `stimOn_times`.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered by `clusters.metrics.label >= 1`, and sessions with zero such clusters are dropped. In addition, trials with all-zero binned neural matrices are later dropped.

ii. 
```python
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)
...
if n_clusters_good == 0:
    print(f"[skip] {spec.eid}: no good clusters after QC")
    return None

neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. The notes say the agent deliberately chose `label >= 1` because it matched the data paper’s reported well-isolated-neuron count in the local release, even though the executable Zhang preprocessing path apparently kept all clusters and only stored QC metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`, with a fixed window from `-0.5` s to `+1.5` s relative to stimulus onset.

ii. 
```python
TIME_WINDOW = (-0.5, 1.5)
...
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
intervals = np.c_[align_times + window[0], align_times + window[1]]
```

iii. The notes explicitly say the agent chose stimulus-onset alignment because the user instruction said “Temporally align based on stimulus onset,” and because the Zhang caching code uses `stimOn_times` with the same 2 s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, giving 100 time bins per trial. No later temporal rebinning is applied after spike binning/interpolation.

ii. 
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
...
"time_bin_size": 20.0,
```

iii. The notes say 20 ms was chosen to match the executable Zhang code and the top-level `T = 100` description, despite prose inconsistencies elsewhere in the papers.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read directly as a raw array. It is synthesized from the chosen alignment event (`stimOn_times`) plus the fixed global trial grid defined by `TIME_WINDOW` and `BINSIZE_S`.

ii. 
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)

align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
time_input = make_time_input()
```

iii. In the mapping notes, the agent describes this variable as the “common trial time grid” repeated for every trial to match the aligned bin centers/right edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script constructs a 100-bin vector from `-0.48` to `1.5` s using `np.linspace`, then repeats that same vector for every kept trial as the first input channel.

ii. 
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)

input_trial = np.vstack(
    [
        time_input,
        np.full(N_BINS, block_num, dtype=np.float32),
    ]
).astype(np.float32)
```

iii. The notes justify the `-0.48 ... 1.5` grid as matching the behavior-interpolation grid’s right-edge sample times.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the exact same 100-bin trial grid as the binned neural data and is stored per trial in the same session/trial structure.

ii. 
```python
neural_trials = bin_spikes_for_trials(..., align_times=align_times)
...
time_input = make_time_input()
input_trial = np.vstack([time_input, np.full(N_BINS, block_num, dtype=np.float32)])
```

iii. The agent’s notes repeatedly describe a “shared stimulus-onset-aligned 20 ms grid” for neural, wheel, whisker, and time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trials table.

ii. 
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes map this input directly to the “raw `probabilityLeft` block sequence.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script counts consecutive trials with the same `probabilityLeft`, resets the counter when the probability changes, computes this on the original unfiltered trial table, and then repeats the resulting value across all 100 bins of each kept trial.

ii. 
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    ...
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
        counters[i] = count

block_vals = block_trial_number[masked_keep["index"].to_numpy()]
...
np.full(N_BINS, block_num, dtype=np.float32)
```

iii. The notes say this was a deliberate choice: “Compute trial number in block on the original unfiltered trial table. Excluded trials should not renumber the latent block progression.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials.choice`.

ii. 
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. The notes identify `choice` as coming “directly from `trials.choice`.”

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script maps the raw nonzero IBL choice codes to the requested left/right classes, using `choice == 1 -> 0 (left)` and `choice == -1 -> 1 (right)`, then repeats the class across all 100 bins in the trial.

ii. 
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right

choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. The trajectory says the agent explicitly checked the raw sign convention before implementing it, and the notes say local data confirmed `choice == 1` means left and `choice == -1` means right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials.probabilityLeft`.

ii. 
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. The notes describe this output as coming directly from the raw block probability variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats that categorical label across all 100 bins of the trial.

ii. 
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2

prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. The notes say this coding was chosen because the user instruction requested exactly this categorical mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
```

iii. The notes say wheel speed should come from the wheel position/timestamp stream using the bundled `brainbox.behavior.wheel` utilities.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. If timestamps are stored as two-column intervals, the script averages the two columns. It then interpolates wheel position to 1 kHz, computes filtered velocity, takes the absolute value to get speed, and linearly interpolates that continuous speed trace into each trial’s stimulus-onset-aligned 20 ms grid.

ii. 
```python
if ts.ndim == 2 and ts.shape[1] == 2:
    ts = ts.mean(axis=1)
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)

wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
```

iii. The notes state that wheel is loaded with the bundled wheel helper functions and then interpolated onto the shared 20 ms stimulus-onset grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The script computes two global tertile edges across all aligned wheel-speed samples in all kept sessions/trials, then applies `np.digitize` to assign each sample to `{0,1,2}`.

ii. 
```python
def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    return float(q1), float(q2)

def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)

wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont
)
```

iii. The notes justify this as a pragmatic export choice: “Global edges preserve a common categorical meaning across sessions and should keep class balance better than fixed-width bins.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated trial-by-trial onto the same stimulus-onset-aligned 100-bin grid as the neural data, and trials are rejected if the wheel source does not adequately cover the requested window.

ii. 
```python
wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
combined_mask = wheel_mask & whisk_mask & neural_mask

x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The notes say the agent intentionally used a single common stimulus-onset alignment because the task explicitly required it, despite movement-aligned dynamic behavior descriptions in the paper text.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` when available, otherwise `rightCamera.ROIMotionEnergy.npy`, together with the corresponding camera timestamps.

ii. 
```python
for camera in ("left", "right"):
    me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
    times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
    ...
    return times, values, camera
```

iii. The notes say this left-then-right fallback was chosen to match the reference code exactly.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the chosen camera’s motion-energy values and timestamps, trims them to equal length if necessary, removes nonfinite samples during interpolation, and linearly interpolates the continuous trace into each trial’s common 20 ms grid.

ii. 
```python
values = np.asarray(np.load(me_file), dtype=np.float64)
times = np.asarray(np.load(times_file), dtype=np.float64)
if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

valid_source = np.isfinite(target_times) & np.isfinite(target_values)
target_times = target_times[valid_source]
target_values = target_values[valid_source]
```

iii. The notes describe whisker motion energy using the paper’s ROI-motion-energy definition and the reference code’s left-first/right-fallback loading rule.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly like wheel speed: the script computes two global tertile edges over all aligned whisker-motion samples, then digitizes each sample into three bins.

ii. 
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont
)
...
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. The notes justify this for the same reason as wheel speed: a common cross-session categorical scale with roughly balanced classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same stimulus-onset-aligned 100-bin grid as the neural data, and trials lacking sufficient whisker coverage are dropped.

ii. 
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. The notes explicitly describe whisker as part of the “shared stimulus-onset-aligned 20 ms grid.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles minor data issues pragmatically. It resolves versioned ALF files recursively, skips sessions with missing required modalities/files, trims whisker arrays when value/time lengths disagree, removes nonfinite source samples before interpolation, rejects trials with missing required events or insufficient behavioral coverage, and raises errors if supposedly categorical variables contain unexpected values.

ii. 
```python
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
    ...

if wheel_pos_file is None or wheel_ts_file is None:
    raise FileNotFoundError(...)

if len(values) != len(times):
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]

valid_source = np.isfinite(target_times) & np.isfinite(target_values)
...
if np.any(mapped < 0):
    raise ValueError(...)
```

iii. The notes say the agent wanted robust handling for versioned caches and missing modalities, and Step 10 says all-zero neural windows were also treated as bad trials and filtered out after verification.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are the per-session spike loading/merging and especially the trial-by-trial spike binning loop; the per-trial behavior interpolation loops are another major cost. The script’s structure and the agent’s notes both identify spike binning as the main hot path.

ii. 
```python
for probe_path in probe_paths:
    times, clusters, regions, total_here, good_here = load_probe_spikes_and_regions(...)

for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
    if idx1 > idx0:
        counts, _, cluster_idx = bincount2D(...)

for i, align_time in enumerate(align_times):
    ...
    interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
```

iii. `CONVERSION_NOTES.md` says “Trial-by-trial spike binning is the main hot path,” and the full conversion logs show per-session times rising sharply with larger neuron/trial counts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-trial spike-binning loop in `bin_spikes_for_trials()`, the per-trial interpolation loop in `interpolate_behavior_trials()`, the per-trial input/output construction loop in `process_session()`, and the per-trial discretization/stacking loop in `build_data_dict()`.

ii. 
```python
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    ...

for i, align_time in enumerate(align_times):
    ...

for block_num, choice_val, prior_val in zip(block_vals, choice_vals, prior_vals, strict=True):
    ...

for choice, prior, wheel_cont, whisk_cont in zip(
    session.choice, session.prior, session.wheel_cont, session.whisker_cont, strict=True
):
    ...
```

iii. The notes do not spell out every vectorization opportunity, but they do identify the trial-by-trial spike path as the dominant bottleneck, which is consistent with these loops being the main optimization targets.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly re-runs recursive file searches (`pick_one_file`) for each asset, re-creates the same `time_input` vector once per session even though it is constant, and stores continuous wheel/whisker traces only to discretize them later in a second pass when building the final dataset.

ii. 
```python
matches = list(base.rglob(pattern))

time_input = make_time_input()

wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
...
discretize_three_bins(wheel_cont, wheel_edges),
discretize_three_bins(whisk_cont, whisker_edges),
```

iii. The notes mention that dynamic outputs are first stored as continuous aligned traces and only later discretized, and that versioned-cache file discovery required repeated robust path resolution.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded work is that `wheel_cont` and `whisker_cont` are fully stored in each `ProcessedSession` even though only their discretized versions are written to the final pickle. The script also keeps bookkeeping fields like `kept_trial_indices`, `n_clusters_total`, and `whisker_source` for logging/inspection but not for downstream decoding. There is also an unused `cluster_ids` allocation in `bin_spikes_for_trials()`.

ii. 
```python
cluster_ids = np.arange(n_clusters, dtype=np.int32)

return ProcessedSession(
    ...
    wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
    whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
    kept_trial_indices=masked_keep["index"].to_numpy(dtype=np.int32),
    whisker_source=whisker_source,
    n_clusters_total=n_clusters_total,
)

output_trial = np.vstack(
    [
        choice,
        prior,
        discretize_three_bins(wheel_cont, wheel_edges),
        discretize_three_bins(whisk_cont, whisker_edges),
    ]
)
```

iii. The notes make clear that continuous wheel/whisker traces were retained mainly to support global-threshold computation, validation, and plotting; they are not part of the final exported decoder inputs/outputs.
