# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API loaders as the primary entrypoint. It reads the frozen release table from `code/code_zhang2025/data/bwm_release.csv`, builds per-session filesystem paths under `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/`, and then opens trial, wheel, camera, spike, cluster-metric, and channel files directly from those ALF paths.

ii. <Code snippets>
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")
```

```python
def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
```

```python
@property
def session_path(self) -> Path:
    return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by treating the frozen 459-session release in `bwm_release.csv` as authoritative and by following the local ALF/ONE directory structure directly. The trajectory shows it deliberately preferred the local cache plus frozen release metadata over API-based loading.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `subject` column of the release CSV. The final output uses sorted unique subject names and a per-session `subject_idx`.

ii. <Code snippets>
```python
SessionSpec(
    eid=eid,
    subject=str(row["subject"]),
```

```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The notes state that subject IDs come directly from frozen release metadata, so nothing is inferred from file names beyond looking up the session path.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in the release CSV. The AI groups rows by `eid`, with one `SessionSpec` per session and one or more `probe_names` attached to that session.

ii. <Code snippets>
```python
grouped = bwm.groupby("eid", sort=False)
for eid, df in grouped:
    row = df.iloc[0]
    probe_names = tuple(df["probe_name"].tolist())
```

iii. The justification in the notes is that the frozen release table is the session universe, so `eid` is the session key.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of the `_ibl_trials.table.pqt` table for each session. Filtering is done with a boolean mask, but the row-wise trial structure is inherited directly from the raw trials table.

ii. <Code snippets>
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

```python
trials = load_trials_table(spec.session_path)
trial_mask = compute_trial_mask(trials)
```

iii. The AI’s notes describe the trials table as the native per-trial structure, matching the reference code’s use of the IBL trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if required task variables are non-null, reaction time is between 0.08 s and 2.0 s, trial duration from `goCue_times` to `feedback_times` is at most 10 s, and `choice != 0`. That mask is then intersected with wheel and whisker coverage masks that require a valid full interpolation window around stimulus onset.

ii. <Code snippets>
```python
rt = trials["firstMovement_times"] - trials["stimOn_times"]
mask = (
    ~trials["stimOn_times"].isnull()
    & ~trials["choice"].isnull()
    & ~trials["feedback_times"].isnull()
    & ~trials["probabilityLeft"].isnull()
    & ~trials["firstMovement_times"].isnull()
    & ~trials["feedbackType"].isnull()
    & (rt >= 0.08)
    & (rt <= 2.0)
    & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
    & (trials["choice"] != 0)
)
```

```python
keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. In the notes, the AI explicitly says it is following the shared missing-event and reaction-time filters from the paper and the provided code path, while also requiring usable wheel and whisker windows because those outputs are mandatory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Cluster metrics and channel-region arrays are additionally used to keep only `label >= 1` clusters and assign Beryl brain-region labels.

ii. <Code snippets>
```python
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
metrics = pd.read_parquet(metrics_path, columns=["label"])
```

```python
cluster_channels = np.load(clusters_channels_path, mmap_mode="r")[good_rows]
channel_region_ids = np.load(channels_region_ids_path, mmap_mode="r")
cluster_regions = brain_regions.id2acronym(region_ids, mapping="Beryl").astype(str)
```

iii. The notes justify this as an electrophysiology conversion: no calcium preprocessing is needed, and the main curation decision is which sorted clusters to keep.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, remaps surviving cluster IDs so they are contiguous across probes, sorts the merged spikes by time, and bins spikes into 20 ms bins over a `[-0.5, 1.5]` s window around `stimOn_times`. The output is stored as spike counts, not divided by bin width into firing rate.

ii. <Code snippets>
```python
remap[good_rows] = np.arange(int(good_rows.sum()), dtype=np.int32)
good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset
```

```python
order = np.argsort(merged_spike_times)
merged_spike_times = merged_spike_times[order]
merged_spike_clusters = merged_spike_clusters[order]
```

```python
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The notes describe the transform as “bin spike counts on a common stimulus-onset window.” I did not find any later justification for omitting the reference conversion from counts to Hz; the trajectory mainly frames this as producing dense per-trial tensors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only clusters whose `clusters.metrics.label >= 1`. It does not explicitly drop Beryl `void` units before constructing the final neural array.

ii. <Code snippets>
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
```

iii. The trajectory shows the AI chose `label >= 1` because it reproduced the paper’s reported 75,708 well-isolated neurons exactly. The notes frame that as the key QC rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. For each kept trial, the binning window runs from 0.5 s before to 1.5 s after stimulus onset.

ii. <Code snippets>
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
```

```python
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
```

iii. The notes repeatedly justify this by the explicit task requirement to align everything to stimulus onset, even where the methods paper used target-specific alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, for 100 bins per trial. There is no additional temporal rebinning after this.

ii. <Code snippets>
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

```python
n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
```

iii. The notes explicitly cite the reference code’s `time_window=(-0.5, 1.5)` and `binsize=0.02` as the intended temporal grid.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not computed from an extra recorded stream. It is a synthetic common time axis defined relative to the session’s `stimOn_times` alignment event.

ii. <Code snippets>
```python
ALIGN_EVENT = "stimOn_times"
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The notes describe this input as the common trial grid relative to `stimOn_times`, reused for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI constructs a fixed 100-sample vector from `-0.48` s to `1.5` s in 20 ms steps and repeats it for every trial. It does not derive trial-specific timing beyond choosing `stimOn_times` as the alignment anchor.

ii. <Code snippets>
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

```python
inp = np.vstack(
    [
        COMMON_RELATIVE_TIMES,
        np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
    ]
).astype(np.float32)
```

iii. The notes justify using a common 2D input shape for all trials and sessions. I did not find an explicit justification for using this right-shifted grid instead of the reference bin centers.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intends this input to be the common time grid used alongside the stimulus-aligned neural bins for every trial. In practice it uses the same 100-bin trial structure as the neural data, but labels those bins with `COMMON_RELATIVE_TIMES`.

ii. <Code snippets>
```python
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
```

```python
inp = np.vstack([COMMON_RELATIVE_TIMES, ...]).astype(np.float32)
```

iii. The notes justify a shared stimulus-onset-aligned grid across all modalities. There is no explicit discussion of the 10 ms offset between these values and the reference’s bin-center convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table. A change in `probabilityLeft` marks the start of a new block.

ii. <Code snippets>
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
```

```python
trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes explicitly describe block structure as being reconstructed from `probabilityLeft` because the trials table does not contain a ready-made block counter.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans the full unfiltered trial sequence, resets the counter whenever `probabilityLeft` changes, counts within block starting at `1`, and then keeps only the entries for the retained trials. It repeats that scalar across all 100 time bins of the trial’s input matrix.

ii. <Code snippets>
```python
prev = None
counter = 0
for i, val in enumerate(prob_left):
    current = None if pd.isna(val) else float(val)
    if i == 0 or current != prev:
        counter = 1
    else:
        counter += 1
    out[i] = counter
```

```python
trial_number_in_block=trial_number_in_block[keep_mask],
```

iii. The notes justify computing this on the original trial order, before filtering, so that removed trials still advance the within-block count. The trajectory and notes both show the AI intentionally used 1-based counting.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii. <Code snippets>
```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
```

```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
```

iii. The notes map raw task choices directly from the trial table, after trial filtering removes no-response trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps raw `choice == 1` to decoder value `0` and raw `choice == -1` to decoder value `1`. The resulting trial-level categorical value is repeated across all 100 time bins.

ii. <Code snippets>
```python
mapped[raw_choice == 1] = 0
mapped[raw_choice == -1] = 1
```

```python
np.full(NBINS, choice[trial_idx], dtype=np.int8),
```

iii. The notes justify this as the task-required recoding of left/right into `0/1`, with no-choice trials already excluded.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. <Code snippets>
```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
```

```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
```

iii. The notes identify `probabilityLeft` as the block-prior variable carried by the raw trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds the raw value to one decimal place and maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. As with choice, the scalar category is repeated across all 100 bins.

ii. <Code snippets>
```python
mapper = {0.2: 0, 0.5: 1, 0.8: 2}
for i, val in enumerate(raw_prior):
    key = round(float(val), 1)
```

```python
np.full(NBINS, prior[trial_idx], dtype=np.int8),
```

iii. The notes justify this as the decoder-task remapping requested in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. <Code snippets>
```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes explicitly map wheel speed back to the raw wheel position stream and use IBL wheel-processing functions to obtain velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1000 Hz with `interpolate_position`, computes filtered velocity with `velocity_filtered`, takes absolute value, interpolates the resulting speed trace into each trial’s stimulus-aligned 100-bin window, and then discretizes using globally computed tertile edges over all kept wheel samples from all sessions.

ii. <Code snippets>
```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = np.interp(x_interp, curr_times, curr_vals)
```

```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)
```

iii. The notes justify the wheel-speed computation as matching IBL’s documented interpolation/filtering, but the metadata and trajectory explicitly justify the binning rule as “global tertile bins over all kept timepoints.”

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes two global thresholds from all kept wheel-speed samples across all kept sessions and time bins, then applies `np.digitize` to produce categories `0`, `1`, and `2`.

ii. <Code snippets>
```python
def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
```

```python
wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
```

```python
"dynamic_output_binning": "global tertile bins over all kept timepoints",
```

iii. The AI explicitly documented this policy in metadata and in the trajectory. I did not find a justification tying it back to the reference code; it appears to have been chosen for cross-session consistency.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated into the same 100-bin stimulus-onset-aligned trial window used for the neural tensor, trial by trial.

ii. <Code snippets>
```python
wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
```

```python
interval_begs = align_times + time_window[0]
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes justify this as part of the common stimulus-onset-aligned grid imposed by the task.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<view>Camera.ROIMotionEnergy.npy` plus the matching `_ibl_<view>Camera.times.npy`. The AI prefers the left camera and falls back to the right camera if the left stream is missing.

ii. <Code snippets>
```python
for view in ("left", "right"):
    me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
    ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
```

```python
return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
```

iii. The notes explicitly say “prefer left camera when available, else right,” mirroring the reference behavior-loading choice.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy trace directly, trims timestamp arrays if they are longer than the motion-energy array, interpolates the trace into each trial’s stimulus-aligned 100-bin window, and later discretizes those values into three categories.

ii. <Code snippets>
```python
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
```

```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The notes say no extra filtering or normalization is applied to whisker motion energy beyond interpolation onto the common trial grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI computes global tertile edges from all kept whisker-motion samples across all sessions and uses those two edges to digitize values into bins `0`, `1`, and `2`.

ii. <Code snippets>
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
```

```python
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. The metadata again records this as “global tertile bins over all kept timepoints.” The trajectory suggests the AI preferred this for uniform categories across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is interpolated into the same stimulus-onset-aligned 100-bin trial window used for the neural data.

ii. <Code snippets>
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

```python
interval_begs = align_times + time_window[0]
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes justify this by the same task-wide requirement to use a common stimulus-onset-aligned grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled mostly by exclusion. Sessions with no good units or fewer than two valid trials are dropped. Sessions missing required streams are dropped. Trials are dropped if interpolation windows are missing or contain NaNs. Camera timestamps longer than motion-energy traces are trimmed; camera timestamps shorter than data raise an error. Invalid spike cluster indices are masked out instead of indexing past the metrics table.

ii. <Code snippets>
```python
except FileNotFoundError:
    return spec.eid, None, "missing_required_stream"
```

```python
if keep_mask.sum() < 2:
    return None
```

```python
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
```

```python
valid_spikes = spikes_clusters < n_clusters
if np.all(valid_spikes):
    spike_mask = good_rows[spikes_clusters]
else:
    spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
    spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
```

iii. The notes justify this as defensive handling needed when reading raw cache files directly rather than via higher-level loaders. The general policy is to drop unusable trials/sessions rather than impute.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading large spike and cluster files from disk and then binning spikes trial by trial. There is also a full first pass over all sessions to read trials, wheel, whisker, and cluster metrics before the second pass that actually loads spikes and builds the dataset.

ii. <Code snippets>
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```

```python
for i in range(len(align_times)):
    ...
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```

iii. The trajectory repeatedly treats spike I/O and session-wide preprocessing as the main computational constraint, especially when deciding to use QC-filtered units and multi-threading.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the explicit per-trial loops in behavior interpolation and spike binning, plus the simple per-trial loop that assembles input/output matrices and the loop that computes trial number in block.

ii. <Code snippets>
```python
for i in range(len(align_times)):
    ...
    y_interp = np.interp(x_interp, curr_times, curr_vals)
```

```python
for i in range(len(align_times)):
    ...
    counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```

```python
for trial_idx in range(len(neural_trials)):
    ...
    session_output.append(out)
```

iii. There is no explicit justification in the notes beyond clarity and practicality; the trajectory focuses more on making the conversion run end-to-end than on micro-optimizing these loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work. It reads cluster metrics once in `count_good_units` during the first pass and again in `load_good_spikes_and_regions` during the second pass. It also performs repeated `resolve_latest(...glob...)` lookups across the same session directories in both passes.

ii. <Code snippets>
```python
def count_good_units(spec: SessionSpec) -> int:
    ...
    metrics = pd.read_parquet(metrics_path, columns=["label"])
```

```python
def load_good_spikes_and_regions(spec: SessionSpec, brain_regions: BrainRegions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
    metrics = pd.read_parquet(metrics_path, columns=["label"])
```

iii. I did not find a written justification for this repeated work. It appears to come from the AI’s two-pass design: first determine which sessions are usable, then build the final dataset.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest redundant processing is the first-pass good-unit counting, because the second pass reloads the same cluster metrics and reconstructs the actual good-unit set again. The code also maintains some bookkeeping that is not materially used downstream, such as the empty `excluded_session_notes` list in metadata.

ii. <Code snippets>
```python
n_good_units = count_good_units(spec)
if n_good_units == 0:
    return None
```

```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

```python
excluded_session_notes: list[dict[str, Any]] = []
...
"excluded_session_notes": excluded_session_notes,
```

iii. No explicit justification was documented. This looks like an implementation convenience rather than a deliberate part of the final analysis pipeline.
