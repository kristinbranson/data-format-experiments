# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local release metadata CSV, maps rows to locally present experiment NWBs, drops passive experiments and experiments without eye tracking, then reads each retained NWB directly with `h5py`. It makes two passes: one for global label/bin statistics and one for conversion.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {int(path.stem.split("_")[-1]): path for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The notes say direct HDF5 access was chosen because the installed `pynwb/hdmf` stack could not instantiate the NWBs. Active sessions were selected because trial outcome was required; three sessions lacking eye data were excluded so every output could be produced.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` strings, assigned an index when the first retained experiment for that mouse is converted.

ii.
```python
mouse_id=str(int(row.mouse_id))
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The agent treated the SDK/release mouse identifier as the unique animal key.

## 1-c. How are the data split into sessions?

i. Every retained `ophys_experiment_id`/NWB file is a separate decoder session; simultaneous imaging planes are not merged by `ophys_session_id`.

ii.
```python
SessionInfo(ophys_experiment_id=int(row.ophys_experiment_id),
            path=file_map[int(row.ophys_experiment_id)], ...)
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
```

iii. The notes argue that neural traces are experiment-specific and therefore each experiment file should be a decoder session, while acknowledging that multiple experiments can belong to one ophys session.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the NWB `intervals/trials` table. A 30 Hz grid is made from each retained trial's `start_time` through (but not including) `stop_time`.

ii.
```python
start = float(raw["trials"]["start_time"][trial_idx])
stop = float(raw["trials"]["stop_time"][trial_idx])
grid = session_grid(start, stop)
```

iii. The agent chose the SDK-defined full trial window so pre-change flashes and the post-change response interval remain available.

## 1-e. How are trials filtered based on quality controls?

i. It retains go or catch trials and excludes aborted and auto-rewarded trials. Sessions with fewer than two retained trials are skipped. It also removes whole passive or pupil-missing experiments upstream.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. The go/catch filter follows the task instructions. Passive sessions were considered incompatible with meaningful outcomes, and missing-pupil sessions were excluded to maintain a complete five-output dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB precomputed calcium-event matrix and its event-detection timestamps, not dF/F.

ii.
```python
ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"])
events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The agent cited the strategy paper's use of detected calcium events as stronger guidance than the whitepaper's dF/F description.

## 2-b. How is the `neural` data processed?

i. Event traces are optionally masked by `valid_roi`, linearly interpolated over all neurons onto each trial's 30 Hz grid, and transposed to neurons by time.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. The notes say this follows the paper's common 30 Hz interpolation and gives one bin size despite mixed native imaging rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Released event traces are retained, with a `valid_roi` mask applied only when the event width still includes invalid cell-table rows. No extra activity-based neuron filter is used.

ii.
```python
valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
if valid_roi.sum() != events.shape[1]:
    events = events[:, valid_roi]
```

iii. The agent reasoned that released NWBs already embody segmentation and ROI QC and that ad hoc additional filtering would diverge from the SDK.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start in absolute synchronized time; event traces are evaluated on `start_time + n/30` until trial stop.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
"temporal_alignment_event": "trial start"
```

iii. The agent relied on the shared experimental clock and recorded trial start as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 33.33 ms bins (30 Hz). Neural events, running, and pupil streams are linearly resampled; categorical stimulus labels are evaluated on that grid.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
n_bins = max(1, int(math.ceil((stop - start) / dt)))
"time_bin_size": float(TIME_BIN_SIZE_MS)
```

iii. The notes justify 30 Hz as common across mixed ~11/31 Hz imaging and naturally ~30 Hz behavior, citing paper event-triggered interpolation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active stimulus-presentation `start_time`, `stop_time`, `image_name`, and `omitted`, rather than trial `initial_image_name`/`change_image_name`.

ii.
```python
stim = read_interval_table(stim_group, ["start_time", "stop_time", "image_name", "is_change", "omitted", ...])
```

iii. The agent wanted actual flashed images, gray gaps, and omissions represented over the entire trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is `gray` plus sorted non-omitted image names. Each grid point defaults to gray and receives an image code only while inside a non-omitted presentation interval.

ii.
```python
image_values = ["gray"] + sorted(image_names)
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. The agent said an explicit gray class was needed for a complete time-varying signal during 500 ms gaps and omissions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the exact same 30 Hz `grid` used to interpolate neural events.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Shared absolute timestamps and a shared query grid were used to guarantee binwise alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `omitted`, `start_time`, and `stop_time`.

ii.
```python
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
```

iii. The agent preferred the raw changed-presentation interval over constructing an impulse solely from trial `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Grid bins within a stimulus interval are labeled from `is_change & ~omitted`; all other bins are zero.

ii.
```python
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. Notes report that a one-bin impulse was initially too sparse, so the changed-image presentation window was used as a more learnable but still task-aligned target.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already boolean and is cast directly to integer categories 0 (`no_change`) and 1 (`change`); there is no numeric threshold.

ii.
```python
codes = np.zeros(query_t.shape, dtype=np.int64)
codes[assign] = changed.astype(np.int64)
```

iii. The source `is_change` field supplies the binary classification.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change label is evaluated on the same per-trial 30 Hz grid as neural activity.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The shared grid was explicitly selected to prevent cross-stream offsets.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from the NWB processed running `speed/data` and `speed/timestamps` arrays.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64)
```

iii. The notes identify this as the SDK-style filtered running-speed product rather than recomputing wheel processing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to every retained trial's 30 Hz grid in both passes; global quintile edges are collected in pass 1 and applied in pass 2.

ii.
```python
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
running_edges = robust_quintile_edges(running_all)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global bins maintain identical categories across sessions and roughly balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles over all retained trial bins define five integer classes; non-increasing tied edges are nudged by `1e-6`.

ii.
```python
percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
percentiles[i] = percentiles[i - 1] + 1e-6
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. This directly implements the required five equal-percentile categories while guarding against duplicate quantiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed and neural events are separately interpolated to the identical 30 Hz trial grid.

ii.
```python
neural_trial = interpolate_matrix(..., grid).T
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Hardware-synchronized absolute timestamps plus a shared query grid provide alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil width and height plus eye-tracking timestamps, defining diameter as their elementwise maximum. It does not explicitly read `likely_blink`.

ii.
```python
pupil_width = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"])
pupil_height = np.asarray(h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"])
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The notes characterize this as blink-masked width/height from released eye tracking and chose `max(width, height)` as diameter, although the code relies only on non-finite values rather than reading a blink flag.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Non-finite diameter samples are removed, remaining samples are linearly interpolated (including endpoint extrapolation by `np.interp`) to each 30 Hz grid, and global quintiles are computed/applied.

ii.
```python
valid = np.isfinite(pupil_diameter)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
```

iii. The agent intended invalid/blink gaps to be filled by interpolation and used global percentiles for shared categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. As with running, global 20/40/60/80 percentile edges define labels 0–4, with tied edges nudged upward.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This implements five approximately equal-frequency categories over the retained dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil samples and neural events are interpolated to the same per-trial 30 Hz absolute-time grid.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The agent cites synchronized acquisition clocks and verifies alignment through raw-to-converted spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. These are the SDK's canonical mutually exclusive outcomes for valid active go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes map to codes 0–3 in fixed order. Although conceptually static, the code broadcasts the label across every trial bin.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting simplified stacking all outputs into one time-varying matrix; the underlying value remains static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments without required eye-tracking groups are excluded. Non-finite pupil values are interpolated over; one valid value is repeated and zero valid values raise an error. Duplicate quantile edges are nudged. Invalid outcome labels and missing required HDF5 columns raise errors. There is no per-session exception recovery, and `np.interp` extends endpoint values outside coverage.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]))
```

iii. The notes document exclusion of three pupil-missing sessions and treat verified all-zero event trials as genuine sparse data rather than corruption.

## 9-a. What are the most time-consuming steps of the code?

i. The agent identifies trialwise neural-event interpolation in pass 2 as the dominant compute cost; NWB I/O/session reads are also repeated and substantial.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Sample timing was extrapolated by neuron-bin workload, and the notes state that neural interpolation remained the dominant expected cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer session and per-trial loops remain serial. Inside each trial, neural interpolation is vectorized over neurons. The small loop assigning image-name codes could also be replaced by a vectorized mapping.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    target[i] = image_to_code[str(name)]
```

iii. The agent specifically notes vectorization over neurons as an implemented speedup, leaving variable-length trials as a readable loop.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design rereads every session and recreates trial grids. Running and pupil interpolation are performed once to obtain global edges and again to build final trials. Stimulus/session metadata are also reread.

ii.
```python
raw = read_session_raw(session, load_events=False)  # pass 1
running_values.append(interpolate_vector(..., grid))
...
raw = read_session_raw(session)                     # pass 2
running_cont = interpolate_vector(..., grid)
```

iii. The agent accepted repeated lightweight behavior processing to avoid retaining the much larger neural dataset while calculating global statistics.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several raw fields (`change_time`, `trials_id`, `active`, `flashes_since_change`) are loaded but not used in final conversion. Pass-1 valid counts and native-dt dictionaries are returned mainly for logging, and optional plots compute/display continuous traces that are not stored as decoder outputs. Trial outcome is redundantly broadcast over time, increasing storage.

ii.
```python
stim = read_interval_table(stim_group, [..., "trials_id", "active", "flashes_since_change"])
return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. The extra columns aid schema inspection/diagnostics, logging supports validation, and broadcasting was chosen for a uniform output matrix; none is required by the final downstream labels themselves.
