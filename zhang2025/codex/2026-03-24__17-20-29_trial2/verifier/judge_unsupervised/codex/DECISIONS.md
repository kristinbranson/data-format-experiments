# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the session universe from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, and builds local ALF paths under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>`. It then reads trial tables from `_ibl_trials.table.pqt`, wheel files from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, whisker files from `*Camera.ROIMotionEnergy.npy` plus camera timestamps, and in a second pass reads spike-sorting outputs from each probe’s `pykilosort` folder.

ii.
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    ...

def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

```python
wheel_times, wheel_speed = load_wheel_speed(spec.session_path)
whisker_times, whisker_motion, whisker_source = load_whisker_motion_energy(spec.session_path)
...
spike_times, spike_clusters, cluster_regions = load_good_spikes_and_regions(prepared.spec, brain_regions)
```

iii. In the notes, the agent says the frozen 459-session BWM release from `bwm_release.csv` should define the session set, and that direct ALF loading was used because `SessionLoader`/`ONE.load_object` was not viable in this environment. It explicitly describes the converter as a two-pass pipeline: pass 1 loads trials and behaviors, pass 2 reloads spikes and bins them.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. The final `subjects` list is the sorted unique subject names among included sessions, and `subject_idx` stores the subject index for each retained session.

ii.
```python
SessionSpec(
    eid=eid,
    subject=str(row["subject"]),
    lab=str(row["lab"]),
    date=str(row["date"]),
    session_number=int(row["session_number"]),
    probe_names=probe_names,
)
```

```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
...
subject_idx_list.append(subject_to_idx[prepared.spec.subject])
```

iii. The notes say to use “subject IDs from frozen release metadata” for `subjects` and `subject_idx`, matching the session metadata in `bwm_release.csv`.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid` values from the release CSV. All trial/neural/behavior data for a given `eid` are bundled into one session payload. In multi-worker mode, completed session payloads are appended as futures finish, so final session order is completion order rather than guaranteed release order.

ii.
```python
grouped = bwm.groupby("eid", sort=False)
for eid, df in grouped:
    ...
    sessions.append(SessionSpec(...))
```

```python
eid, neural_trials, session_input, session_output, cluster_regions, elapsed = future.result()
neural_all.append(neural_trials)
input_all.append(session_input)
output_all.append(session_output)
session_ids.append(eid)
```

iii. The notes say the “459-session frozen BWM release” is authoritative. The code organizes outputs as lists of sessions, each tied to one `eid`.

## 1-d. How are the data split into trials?

i. Trials come from rows of each session’s `_ibl_trials.table.pqt`. The agent computes a `keep_mask` and retains only masked trials. Neural data are binned separately for each kept trial’s alignment time, and behavior arrays are stacked per kept trial.

ii.
```python
trials = load_trials_table(spec.session_path)
trial_mask = compute_trial_mask(trials)
...
align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
...
keep_mask = trial_mask & wheel_mask & whisker_mask
...
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_by_trial(...)
```

iii. The notes say the converted dataset should keep only trials with full valid neural/wheel/whisker coverage on the common window and that at least two trials per session are required for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies the reference-style trial mask for missing key events, reaction time 0.08 to 2.0 s, maximum trial duration 10 s, and no-choice removal. It then further requires wheel and whisker interpolation to succeed on the common window, and drops sessions with fewer than 2 kept trials.

ii.
```python
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
...
keep_mask = trial_mask & wheel_mask & whisker_mask
if keep_mask.sum() < 2:
    return None
```

iii. The notes say to keep the provided code path’s `exclude_nochoice=True` and `max_trial_len=10.0`, but also to drop trials lacking valid dynamic-behavior coverage because the requested decoder outputs require wheel and whisker signals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from probe-level spike times and cluster assignments, filtered using `clusters.metrics.label`, with region labels taken from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
clusters_channels_path = resolve_latest(probe_dir, "**/clusters.channels.npy")
channels_region_ids_path = resolve_latest(probe_dir, "**/channels.brainLocationIds_ccf_2017.npy")
spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")
```

iii. The notes map `neural` from `spikes.times`, `spikes.clusters`, and `clusters.metrics.label` across all probes, with Beryl region mapping based on cluster/channel metadata.

## 2-b. How is the `neural` data processed?

i. The agent merges probes within a session, remaps kept cluster IDs to a contiguous session-wide index, sorts merged spikes by time, bins spikes into 20 ms counts on a `[-0.5, 1.5]` s window around `stimOn_times`, and stores each trial as an `(n_neurons, 100)` matrix using `float16`.

ii.
```python
good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset
...
order = np.argsort(merged_spike_times)
merged_spike_times = merged_spike_times[order]
merged_spike_clusters = merged_spike_clusters[order]
```

```python
interval_begs = align_times + time_window[0]
interval_ends = align_times + time_window[1]
...
flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
out.append(counts.astype(np.float16))
```

iii. The notes say the goal was to mirror the reference spike-count binning and probe-merging logic while transposing into the decoder’s `(neuron, time)` trial format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent filters neurons to those with `clusters.metrics.label >= 1`, counts only those units, removes probes with no such units, and drops sessions where no good units remain.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
if not np.any(good_rows):
    continue
...
if cluster_offset == 0:
    raise ValueError(f"No good units remained for {spec.eid}")
```

iii. The notes explicitly justify this as reproducing the paper’s 75,708 well-isolated-neuron count and making the dense decoder format tractable, even though the bundled method code loaded all clusters with `qc=None`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to `stimOn_times`, using a common window from 0.5 s before stimulus onset to 1.5 s after.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
...
kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
neural_trials = bin_spikes_by_trial(..., align_times=kept_align_times, ...)
```

iii. The notes state that the task explicitly required stimulus-onset alignment, so a single stimulus-onset-aligned grid was used for the whole converted dataset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins and 100 bins over a 2 s trial window. The agent does not apply any further temporal rebinning after this binning.

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```

iii. The notes say 20 ms was chosen to match the reference choice/dynamic decoding setup and to keep a common grid across inputs and outputs, even though the method paper used a special 50 ms prior setup.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a measured time series; it is derived from the chosen alignment convention around `stimOn_times`, producing a fixed relative-time vector shared by every trial.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
...
inp = np.vstack(
    [
        COMMON_RELATIVE_TIMES,
        ...
    ]
)
```

iii. The notes describe `input[0]` as “common trial grid relative to `stimOn_times`” and say it should be a 100-length vector reused for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs a 100-point vector from `-0.48` s to `1.50` s using `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)` and repeats it for each trial.

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
...
np.vstack([COMMON_RELATIVE_TIMES, ...]).astype(np.float32)
```

iii. The notes justify this by pointing to the reference interpolation grid, which also samples at `interval_beg + binsize` through `interval_end`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is exactly on the same 100-bin stimulus-onset-aligned grid as the neural trial matrices.

ii.
```python
inp = np.vstack([COMMON_RELATIVE_TIMES, ...])
...
neural_trials = bin_spikes_by_trial(..., align_times=kept_align_times, ...)
```

iii. The notes say `time_since_stimulus_onset_s` uses “the same grid as aligned behavior/neural bins.”

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes map `trial_number_in_block` from the `probabilityLeft` block structure in the raw trial table.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full unfiltered trial sequence, resets the counter to 1 whenever `probabilityLeft` changes, increments otherwise, then subsets the resulting block counts by `keep_mask` and repeats the scalar across time for each kept trial.

ii.
```python
for i, val in enumerate(prob_left):
    current = None if pd.isna(val) else float(val)
    if i == 0 or current != prev:
        counter = 1
    else:
        counter += 1
    out[i] = counter
    prev = current
...
np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32)
```

iii. The notes explicitly justify computing it on the original trial table before filtering so block position is preserved rather than renumbered after exclusions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trial table’s `choice` column after trial filtering.

ii.
```python
choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
...
choice = map_choice(prepared.choice_raw)
```

iii. The notes say `choice` comes from raw `choice` values `{-1, 1}` after filtering, and no-choice trials are removed first.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw `choice == 1` to decoder value `0` for left, maps `choice == -1` to decoder value `1` for right, errors if other values remain, and repeats the categorical value across all 100 time bins of that trial.

ii.
```python
mapped[raw_choice == 1] = 0
mapped[raw_choice == -1] = 1
if not np.all(np.isin(raw_choice, [-1, 1])):
    raise ValueError("Unexpected choice values encountered after filtering.")
```

```python
np.full(NBINS, choice[trial_idx], dtype=np.int8)
```

iii. The notes say this mapping was checked directly against raw high-contrast trials and chosen to satisfy the task’s `left = 0, right = 1` requirement.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table’s `probabilityLeft` column after filtering.

ii.
```python
prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
...
prior = map_prior(prepared.prior_raw)
```

iii. The notes map this output from raw `probabilityLeft` values `{0.2, 0.5, 0.8}`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent rounds each `probabilityLeft` value to one decimal place, maps `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, raises an error for unexpected values, and repeats the result across all 100 bins for that trial.

ii.
```python
mapper = {0.2: 0, 0.5: 1, 0.8: 2}
for i, val in enumerate(raw_prior):
    key = round(float(val), 1)
    if key not in mapper:
        raise ValueError(f"Unexpected probabilityLeft value {val}")
    out[i] = mapper[key]
```

```python
np.full(NBINS, prior[trial_idx], dtype=np.int8)
```

iii. The notes describe this as a task-specific remapping from `probabilityLeft` into the requested three categorical values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
```

iii. The notes say wheel output should come from the `_ibl_wheel.*` files using logic equivalent to reference `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent interpolates wheel position to 1000 Hz using `interpolate_position`, computes filtered velocity with `velocity_filtered(..., corner_frequency=20, order=8)`, takes absolute value to get speed, then linearly interpolates that session-wide signal into each stimulus-aligned trial window.

ii.
```python
interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

```python
wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
```

iii. The notes say wheel speed should match reference `abs(wheel velocity)`, but direct ALF loading plus bundled wheel helpers were used because the full `SessionLoader` path was unavailable.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded with two global tertile edges computed over all kept wheel-speed timepoints across all included sessions/trials. `np.digitize` assigns bins `0`, `1`, or `2`.

ii.
```python
all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
wheel_edges = robust_tertile_edges(all_wheel)
...
wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
```

iii. The notes justify global thresholds by saying session-specific thresholds would make class labels inconsistent across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered 100-bin grid as the neural data. Trial segments are taken on `[stimOn_times - 0.5, stimOn_times + 1.5]` and resampled to the common bin centers.

ii.
```python
align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes explicitly call this a task-imposed override of the method paper’s first-movement alignment for dynamic behaviors.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or, if unavailable, `rightCamera.ROIMotionEnergy.npy`, together with the corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
for view in ("left", "right"):
    me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
    ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
    ...
    return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
```

iii. The notes and README both say whisker motion energy prefers the left camera and falls back to the right if needed.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the motion-energy trace, repairs length mismatches by trimming extra leading timestamps when the timestamp vector is longer than the signal, rejects impossible shorter-than-signal timestamp arrays, and linearly interpolates the motion-energy trace into each stimulus-aligned trial window.

ii.
```python
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
```

```python
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The notes say this was intended to mimic the reference interpolation style and add “reference-style video timestamp repair” for raw-data irregularities.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into three bins using global tertile edges computed over all kept whisker-motion-energy timepoints across included sessions/trials.

ii.
```python
all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
whisker_edges = robust_tertile_edges(all_whisker)
...
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```

iii. The notes justify global thresholds for the same reason as wheel speed: to keep category semantics comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned to the same stimulus-onset-centered 100-bin grid as the neural data, using per-trial linear interpolation on `[stimOn_times - 0.5, stimOn_times + 1.5]`.

ii.
```python
align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. The notes explicitly say this common stimulus-onset alignment was chosen because the task required it, even though the method paper aligned dynamic behaviors to first movement.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing key trial events cause trial exclusion. Missing wheel or whisker trial coverage causes trial exclusion through `wheel_mask` or `whisker_mask`. Sessions missing required behavior streams are excluded entirely. Camera-time vectors longer than motion-energy vectors are trimmed from the front. Sessions with zero good units or fewer than 2 valid trials are dropped. The agent does not pad or impute invalid trials.

ii.
```python
except FileNotFoundError:
    return spec.eid, None, "missing_required_stream"
```

```python
if video_timestamps.shape[0] > video_data.shape[0]:
    video_timestamps = video_timestamps[-video_data.shape[0]:]
```

```python
if curr_vals.shape[0] == 0:
    continue
if np.isnan(curr_vals).any():
    continue
if abs(t_beg - curr_times[0]) > binsize:
    continue
if abs(t_end - curr_times[-1]) > binsize:
    continue
```

iii. The notes say to “drop any trial lacking full valid coverage of neural, wheel, or whisker data on the common window: no padding or fabricated values,” and to exclude sessions without mandatory whisker output.

## 10-a. What are the most time-consuming steps of the code?

i. The notes identify pass 2 spike-stream reload and spike binning as the main bottleneck, especially before optimization. Session-level recursive path discovery and repeated large-array copies were also identified as expensive.

ii.
```python
results_iter = pool.map(build_session_behavior_safe, target_sessions)
...
future_to_index = {
    pool.submit(build_session_payload, prepared, wheel_edges, whisker_edges): idx
    for idx, prepared in enumerate(prepared_sessions)
}
```

```python
spikes_times = np.load(spikes_times_path, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
...
counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```

iii. In Step 6 and Step 7 of the notes, the agent says the “real bottleneck was spike-stream reload and dtype copying in pass 2,” and later reports that pass 2 build time dominated full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining scalar or Python-list loops include `compute_trial_number_in_block`, the per-trial loop in `interpolate_behavior_per_trial`, the per-trial loop in `bin_spikes_by_trial`, and the per-trial assembly loop in `build_session_payload`.

ii.
```python
for i, val in enumerate(prob_left):
    ...
```

```python
for i in range(len(align_times)):
    ...
```

```python
for trial_idx in range(len(neural_trials)):
    inp = np.vstack([...])
    out = np.vstack([...])
    session_input.append(inp)
    session_output.append(out)
```

iii. The notes emphasize that the code was optimized where it mattered most, but these loops remain as straightforward Python loops and could be reworked further if needed.

## 10-c. What processing does the code repeat multiple times?

i. It reads release/session metadata and recursively resolves ALF paths many times. It reads `clusters.metrics.pqt` once in `count_good_units` and again in `load_good_spikes_and_regions`. It computes and stores continuous wheel/whisker traces in pass 1, then later digitizes them again in pass 2 and again for processing plots.

ii.
```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
total += int((metrics["label"].to_numpy() >= 1).sum())
```

```python
metrics = pd.read_parquet(metrics_path, columns=["label"])
good_rows = metrics["label"].to_numpy(copy=False) >= 1
```

```python
wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
...
wheel_bins = digitize_three_bins(prepared.wheel_cont[trial_idx], wheel_edges)
whisker_bins = digitize_three_bins(prepared.whisker_cont[trial_idx], whisker_edges)
```

iii. The notes explicitly mention repeated recursive path discovery and repeated spike reload/copy work as targets of optimization, and the code structure still contains repeated metadata/metric reads across passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `n_good_units` in pass 1 only to gate session inclusion, then rereads all cluster metrics in pass 2. It stores full continuous wheel and whisker arrays in `PreparedSession` even though downstream decoding only uses their tertile-binned versions. It also computes plotting-only discretizations and keeps an unused `excluded_session_notes` list in metadata.

ii.
```python
n_good_units = count_good_units(spec)
if n_good_units == 0:
    return None
```

```python
wheel_cont = np.stack([wheel_interp[i] for i in np.where(keep_mask)[0]], axis=0)
whisker_cont = np.stack([whisker_interp[i] for i in np.where(keep_mask)[0]], axis=0)
```

```python
excluded_session_notes: list[dict[str, Any]] = []
...
"excluded_session_notes": excluded_session_notes,
```

iii. The notes describe the two-pass design as intentional for global discretization, but that design necessarily carries continuous dynamic traces and some session-screening work that are not themselves used by the final decoder beyond enabling later thresholding or exclusion.
