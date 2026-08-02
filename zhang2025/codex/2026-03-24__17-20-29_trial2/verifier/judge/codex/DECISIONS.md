# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the session universe from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, and uses the grouped rows to recover each session's subject, lab, date, session number, and probe names. It then reads raw ALF files directly from `data/one_cache/.../alf/` instead of using `ONE` / `SessionLoader`.

ii.
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    sessions: list[SessionSpec] = []
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(
            SessionSpec(
                eid=eid,
                subject=str(row["subject"]),
                lab=str(row["lab"]),
                date=str(row["date"]),
                session_number=int(row["session_number"]),
                probe_names=probe_names,
            )
        )
    return sessions
```

iii. In `CONVERSION_NOTES.md` and the trajectory, the agent says it treated `bwm_release.csv` as authoritative and bypassed `ONE` / `SessionLoader` because those paths were not workable in the offline environment. It justified direct ALF loading as a functional replacement for the reference loaders.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `SessionSpec.subject`. After preprocessing, the agent creates a sorted unique subject list and stores one `subject_idx` per retained session.

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The trajectory and notes describe this as a target-format decision: the converted dataset should expose unique subject names plus a per-session index.

## 1-c. How are the data split into sessions?

i. Each unique `eid` is treated as one session. If a session has multiple probes, those probes are merged into one session-level payload.

ii.
```python
grouped = bwm.groupby("eid", sort=False)

@dataclass(frozen=True)
class SessionSpec:
    eid: str
    subject: str
    lab: str
    date: str
    session_number: int
    probe_names: tuple[str, ...]
```

iii. The notes explicitly say the agent followed the reference release CSV at the session level and merged probes within a session, mirroring the reference code's `merge_probes` logic.

## 1-d. How are the data split into trials?

i. Trials are read from `_ibl_trials.table.pqt`; each row of that table is treated as one trial. The per-session trial list is then reduced to the subset passing the final `keep_mask`.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. The notes tie this to the reference pipeline's use of the session trials table via `SessionLoader.load_trials()`.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time must be 0.08-2.0 s; trial duration (`feedback_times - goCue_times`) must be at most 10 s; and no-choice trials (`choice == 0`) are removed. The agent then further requires successful wheel and whisker interpolation coverage on the aligned window.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
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
    return mask.to_numpy(dtype=bool)

keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. The notes say this was copied from the reference trial mask (`load_trials_and_mask`) plus an added behavior-coverage requirement so every kept trial has all mandatory outputs on the requested common grid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural counts are derived from per-probe spike timestamps and spike cluster assignments, with cluster metrics used for QC filtering and channel/region files used to attach brain-region labels to retained units.

ii.
```python
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
clusters_channels_path = resolve_latest(probe_dir, "**/clusters.channels.npy")
channels_region_ids_path = resolve_latest(probe_dir, "**/channels.brainLocationIds_ccf_2017.npy")
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
```

iii. The notes compare these files directly to the reference code's `SpikeSortingLoader` outputs and state that the direct file reads are standing in for those loader calls.

## 2-b. How is the `neural` data processed?

i. The agent loads all retained probes for a session, remaps cluster IDs to avoid collisions, concatenates spikes across probes, sorts by spike time, and bins stimulus-aligned spike counts into 20 ms bins over a 2 s window. Each trial is stored as `(n_neurons, n_bins)`.

ii.
```python
good_spike_times = np.asarray(spikes_times[spike_mask], dtype=np.float64)
good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset
...
merged_spike_times = np.concatenate(spike_times_all)
merged_spike_clusters = np.concatenate(spike_clusters_all)
order = np.argsort(merged_spike_times)
merged_spike_times = merged_spike_times[order]
merged_spike_clusters = merged_spike_clusters[order]
...
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The notes say this is meant to reproduce the reference session-level probe merge and trial-aligned spike binning, with a denser output orientation required by the decoder task.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent filters clusters to `clusters.metrics.label >= 1` before binning. Only spikes from those clusters are retained, and sessions with zero such units are dropped.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
...
spike_mask = good_rows[spikes_clusters]
...
if cluster_offset == 0:
    raise ValueError(f"No good units remained for {spec.eid}")
```

iii. The notes and trajectory justify this on two grounds: it reproduces the paper's 75,708 well-isolated-neuron count, and it makes the dense converted dataset tractable. The trajectory explicitly shows the agent changed course from the reference code's all-cluster loading to this QC filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to `stimOn_times`. For each kept trial, the binning window is `[-0.5, 1.5]` s relative to stimulus onset.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
...
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
```

iii. The trajectory quotes the task instruction "Temporally align based on stimulus onset," and the notes say the agent followed that explicit requirement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins, giving 100 bins over the 2 s trial window. No later rebinning is applied.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes cite the reference cache parameters `time_window=(-0.5, 1.5)` and `binsize=0.02`, and the agent reused those directly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a raw file. The agent computes it from the chosen alignment window and bin size.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32
)
```

iii. The notes justify this as the trial-relative time axis implied by the aligned binning scheme.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A single 100-element relative-time vector is precomputed once and reused for every trial. The values correspond to the interpolation / bin-time grid used elsewhere in the script.

ii.
```python
inp = np.vstack(
    [
        COMMON_RELATIVE_TIMES,
        np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
    ]
).astype(np.float32)
```

iii. The notes say the agent matched the reference behavior interpolation grid, which also uses `linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `TIME_WINDOW`, `BINSIZE`, and `NBINS` constants define both the neural bins and the time-since-stimulus input, so they are aligned by construction.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes describe this as a common-grid design choice required by the decoder task.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw trial-table column `probabilityLeft`.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(
    trials["probabilityLeft"].to_numpy()
)
```

iii. The notes say block boundaries were inferred from changes in `probabilityLeft`, consistent with the IBL task structure.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans trials in original session order, resets the counter to 1 whenever `probabilityLeft` changes, increments otherwise, and then applies `keep_mask` after the full-session block counter is computed. The per-trial scalar is repeated across all 100 bins.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
        else:
            counter += 1
        out[i] = counter
        prev = current
    return out
```

iii. The notes explicitly justify computing this before filtering so excluded trials do not renumber the animal's actual position within a block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trial-table `choice` column.

ii.
```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
```

iii. The notes and trajectory both describe this as the standard IBL trial choice variable after removing no-choice trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering, the script maps IBL choices `1 -> 0` and `-1 -> 1`, corresponding to `left=0`, `right=1`. The mapped value is then repeated across the full 100-bin trial for shape consistency.

ii.
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped
```

iii. The trajectory says the agent re-checked the sign convention directly against raw high-contrast trials and the decoder-task requirement `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii.
```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
```

iii. The notes identify `probabilityLeft` as the raw block-prior variable and tie it to both the paper and the reference code's `block` behavior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, rounding to one decimal place before lookup. Like choice, the per-trial category is broadcast across all time bins.

ii.
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    out = np.empty(raw_prior.shape[0], dtype=np.int8)
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        if key not in mapper:
            raise ValueError(f"Unexpected probabilityLeft value {val}")
        out[i] = mapper[key]
    return out
```

iii. The notes say this mapping was taken directly from the decoder-task specification.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes compare this to the reference `SessionLoader.load_wheel()` path and say the agent used the same underlying raw wheel stream.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The script linearly interpolates wheel position to 1000 Hz, computes filtered velocity with `velocity_filtered`, takes the absolute value to convert velocity to speed, and then trial-aligns / interpolates the resulting continuous trace.

ii.
```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. The notes state this was chosen specifically to mirror the underlying `brainbox.behavior.wheel` processing used by the IBL loader path.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After collecting all kept wheel-speed values across all prepared sessions, the agent computes global 1/3 and 2/3 quantile cut points and digitizes every per-bin wheel value into `low`, `medium`, or `high`.

ii.
```python
wheel_edges = robust_tertile_edges(all_wheel)
...
wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
```

iii. The notes justify global tertiles as a task-specific discretization choice that keeps categories comparable across sessions and roughly balanced.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. The script interpolates wheel speed onto the same per-trial 100-bin grid used for neural data, with intervals defined by `stimOn_times + TIME_WINDOW`.

ii.
```python
align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
wheel_interp, wheel_mask = interpolate_behavior_per_trial(
    wheel_times, wheel_speed, align_times
)
```

iii. The trajectory explicitly says the agent overrode the paper's movement-aligned behavior analysis because the task instruction required stimulus-onset alignment for the decoder.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` if present, otherwise `rightCamera.ROIMotionEnergy.npy`, paired with the corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
```

iii. The notes say the agent followed the reference preference order of left camera first, then right as fallback.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the precomputed ROI motion-energy trace, repairs a common timestamp/data-length mismatch by trimming timestamps from the front if necessary, and then trial-aligns / interpolates the continuous trace.

ii.
```python
def check_video_timestamps(view: str, video_timestamps: np.ndarray, video_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```

iii. The notes say the agent looked up the IBL timestamp-repair behavior and intentionally matched that raw-data irregularity handling.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The agent computes global tertile edges from all kept whisker-motion-energy values and digitizes every aligned bin into `low`, `medium`, or `high`.

ii.
```python
whisker_edges = robust_tertile_edges(all_whisker)
...
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. The notes give the same justification as wheel speed: the reference pipeline treats whisker motion energy as continuous, so the task-required discretization was implemented as global tertiles for consistency across sessions.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same stimulus-aligned 100-bin trial grid as the neural data.

ii.
```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(
    whisker_times, whisker_motion, align_times
)
```

iii. The trajectory says this was a deliberate task override relative to the methods paper's first-movement alignment for dynamic behaviors.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required files cause the session to be skipped. Missing or invalid trial events cause the trial to fail `compute_trial_mask`. Wheel or whisker intervals with no coverage, NaNs, or inadequate boundary coverage are excluded trial-wise. Camera timestamp arrays that are longer than the motion-energy array are repaired by front-trimming. Sessions with no good units or fewer than two valid trials are dropped.

ii.
```python
except FileNotFoundError:
    return spec.eid, None, "missing_required_stream"
...
if curr_vals.shape[0] == 0:
    continue
if np.isnan(curr_vals).any():
    continue
if abs(t_beg - curr_times[0]) > binsize:
    continue
if abs(t_end - curr_times[-1]) > binsize:
    continue
...
if keep_mask.sum() < 2:
    return None
```

iii. The notes repeatedly frame this as "drop invalid data, do not fabricate or pad values," and document the excluded-session reasons in the final metadata and run summary.

## 12-a. What are the most time-consuming steps of the code?

i. The agent identified two main bottlenecks: pass 1 behavior loading/interpolation over all sessions, and pass 2 spike loading plus session payload construction / trial binning. The notes report pass 2 as the dominant cost.

ii.
```python
first_pass_start = time.time()
...
print(f"[pass1] completed in {time.time() - first_pass_start:.2f}s")
...
session_start = time.time()
...
elapsed = time.time() - session_start
return prepared.spec.eid, neural_trials, session_input, session_output, cluster_regions.astype(str), elapsed
```

iii. `CONVERSION_NOTES.md` explicitly records pass 1 and pass 2 timings and discusses the spike-loading path as the main optimization target.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining obvious loop candidates are the sequential block-counter loop, the per-trial interpolation loop, the per-trial spike-binning loop, the per-element prior mapping loop, and the per-trial input/output assembly loop.

ii.
```python
for i, val in enumerate(prob_left):
    ...

for i in range(len(align_times)):
    ...

for i in range(len(align_times)):
    ...

for i, val in enumerate(raw_prior):
    ...

for trial_idx in range(len(neural_trials)):
    ...
```

iii. The notes say the agent optimized the highest-impact paths first, but several smaller loops remain because I/O and spike processing were the larger bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly resolves ALF file paths with recursive globs, instantiates `BrainRegions()` inside each session build, and uses a two-pass design that loads session behavior once in pass 1 and reloads spike data in pass 2.

ii.
```python
def resolve_latest(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    ...

brain_regions = BrainRegions()
...
prepared_sessions: list[PreparedSession] = []
...
data = build_dataset(...)
```

iii. The notes call out repeated recursive path discovery and per-session atlas construction as avoidable overhead, and describe the two-pass design as a tradeoff needed to compute global tertile thresholds before final payload assembly.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores and carries intermediate continuous wheel and whisker arrays only to derive global tertile thresholds and final bins; tracks `whisker_source` mostly for plotting/logging; and initializes `excluded_session_notes` but never populates it in the final dataset. It also counts good units in pass 1 and then reloads the actual units again in pass 2.

ii.
```python
wheel_cont = np.stack([wheel_interp[i] for i in np.where(keep_mask)[0]], axis=0)
whisker_cont = np.stack([whisker_interp[i] for i in np.where(keep_mask)[0]], axis=0)
...
excluded_session_notes: list[dict[str, Any]] = []
...
"excluded_session_notes": excluded_session_notes,
```

iii. The notes describe these as acceptable overhead from the chosen two-pass design and from optional plotting / metadata support, not as downstream-required outputs.
