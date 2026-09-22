# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local `ophys_experiment_table.csv`, finds downloaded NWB filenames, retains matching experiments, excludes session types containing `passive`, and opens every retained NWB directly with `h5py`. It does not filter to `project_code == "VisualBehavior"`; consequently its full run includes downloaded active single-plane and multiscope experiments.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes justify direct HDF5 as faster than AllenSDK, use the downloaded subset because the full release is unavailable locally, and exclude passive sessions because the requested task is active Visual Behavior. They claim direct reads are equivalent to `BehaviorOphysExperiment.from_nwb()`.

## 1-b. How are the data split into subjects?

i. Each processed experiment uses metadata `mouse_id`; first occurrence order creates `subjects`, and every output entry gets the corresponding `subject_idx`.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes identify `mouse_id` as the subject identifier and report 38 unique downloaded mice.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) becomes a separate output “session,” even when multiple experiments share an `ophys_session_id`. The true behavioral session ID is retained only as metadata.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    all_neural.append(result['neural'])
```

iii. The notes explicitly decide that each multiscope plane is a separate output session because planes have different neurons despite shared behavior.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each valid trial, all ophys frames satisfying `start_time <= timestamp < stop_time` are selected, so trials have variable lengths.

ii.
```python
valid_trial_idx = get_valid_trials(trial_data)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI says the SDK/NWB trial boundaries are the experiment-defined trials and that using the full window preserves pre- and post-change time-varying signals.

## 1-e. How are trials filtered based on quality controls?

i. It retains `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Experiments are rejected if fewer than two such trials exist, and individual trials with fewer than two ophys frames are skipped; the experiment is checked again after processing.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if len(valid_trial_idx) < 2: return None
if n_trial_frames < 2: continue
if len(neural_trials) < 2: return None
```

iii. This is justified directly by the task instructions; aborted trials reflect premature licks and auto-rewarded trials are not ordinary task trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is the precomputed NWB dF/F trace dataset at `processing/ophys/dff/traces/data`, with its associated timestamps.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
```

iii. The AI chose dF/F as the standard calcium-imaging signal and notes that it is already baseline-normalized and detrended by the Allen pipeline.

## 2-b. How is the `neural` data processed?

i. The only conversion is transposition from `(frames, cells)` to `(cells, frames)`, trial slicing, and casting to `float32`. Planes are not combined.

ii.
```python
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes state no recomputation, normalization, resampling, or deconvolution is needed because NWB dF/F is preprocessed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit ROI/cell quality filter is applied. Every column in the NWB dF/F matrix is retained; zero-cell experiments are rejected.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    return None
```

iii. The notes acknowledge the SDK default `exclude_invalid_rois=True` and assert that all ROIs in the downloaded NWBs are already valid, although the script does not verify `valid_roi`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned on the ophys clock and segmented from trial start through trial stop, not into a fixed window around change time.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI cites hardware synchronization and says the shared ophys frame mask guarantees temporal correspondence among neural and output streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Each experiment remains at native sampling (`median(diff(ophys_ts))`), approximately 32.3 ms for Scientifica and 93.2 ms for multiscope, while dataset metadata reports only the median across experiments (32.3 ms).

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
median_dt = np.median([m['dt_ms'] for m in session_metadata])
```

iii. The notes justify retaining native timestamps, but also recognize both ~31 Hz and ~11 Hz recordings. This conflicts with the target requirement that bins be the same size across all sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It uses stimulus-presentation `start_time`, `stop_time`, and `image_name`, aligned against ophys timestamps, rather than trial-table initial/change image fields.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    stim_data[key] = stim[key][:]
```

iii. The notes say the presentations table provides the actual flashed image and permits explicit representation of gray inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is scanned from all retained NWBs; `gray` is prepended. Each trial starts as gray, stimulus-overlap frames receive the image’s integer index, and `omitted` presentations remain gray.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
trace = np.full(n_frames, gray_idx, dtype=np.int64)
if name == 'omitted': continue
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The AI interprets “image identity (during non-grey screen)” as requiring a gray category between 250-ms flashes and says plots showed the expected ~66.7% gray fraction.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus intervals are projected onto the exact per-trial ophys timestamps used to slice neural data.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
```

iii. The AI reports spot-checking the generated identity trace against raw NWB presentation onsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It derives change events from stimulus-presentation `start_time` and boolean `is_change`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The notes identify presentation-level `is_change` as the direct source for real image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every changed presentation within a trial, it finds the first ophys frame at or after onset and sets only that single frame to one.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The AI reads “right after a change” as an onset impulse and documents the output as “1 at change onset frame.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: `is_change=False` maps to category 0 (`no_change`) and the one onset frame for `is_change=True` maps to 1 (`change`).

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
trace[frame_idx] = 1
output_values[1] = ['no_change', 'change']
```

iii. The raw field is already boolean, so the AI applied a direct binary encoding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is located in the same per-trial `trial_ts` vector as neural frames using `searchsorted`.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. The AI says selecting the first ophys frame at/after onset gives frame-level alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB `processing/running/speed/data` and its timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes identify this as the Allen processed running-speed stream sampled near 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Full-session speed is linearly interpolated to all ophys timestamps. Five percentile edges are computed separately for each experiment using all session timepoints, then applied to each trial.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The AI says interpolation synchronizes the 60-Hz stream and session-wide percentiles provide five balanced bins while keeping a consistent mapping within a recording.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Non-NaN values are divided at the experiment-wide 20th, 40th, 60th, and 80th percentiles into integer bins 0–4. NaNs are assigned 0.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
result[~valid] = 0
```

iii. Equal percentile bins were chosen because the decoder specification explicitly requests them.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is first interpolated to the full ophys timebase, then sliced with the identical trial frame mask.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
running_trial = running_at_ophys[frame_mask]
```

iii. The AI relies on hardware-synchronized timestamps and common ophys indexing.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil ellipse `area`, eye timestamps, and `likely_blink` from the NWB acquisition group.

ii.
```python
data['pupil_area'] = pt['area']['data'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes say area is available in the local NWBs and diameter can be recovered using the circular-area formula; blinks are known artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN; positive areas are converted to equivalent-circle diameter `2*sqrt(area/pi)`, then linearly interpolated to ophys timestamps. Experiment-wide percentile edges are computed and applied per trial.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The AI cites the whitepaper’s area-to-diameter formula and blink removal, and applies the same session-percentile rationale as running.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses per-experiment 20/40/60/80 percentile cutoffs to produce bins 0–4; missing values map to bin 0. If all values are missing, fallback edges are `linspace(0,1,6)` and all outputs remain 0.

ii.
```python
if len(valid) == 0:
    return np.linspace(0, 1, n_bins + 1)
...
result[~valid] = 0
```

iii. Five equal percentile bins follow the decoder request; assigning missing values to the lowest category is documented as a design choice.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Converted pupil diameter is interpolated to the experiment’s ophys timestamps and sliced with the same trial mask as neural activity.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. The AI says hardware synchronization plus common ophys indices ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It reads the trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if trial_data['hit'][idx]: return 'hit'
elif trial_data['miss'][idx]: return 'miss'
elif trial_data['false_alarm'][idx]: return 'false_alarm'
elif trial_data['correct_reject'][idx]: return 'correct_reject'
```

iii. The notes describe these as the canonical mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcome names are mapped in fixed order to 0–3 (unknown falls back to 0), then the scalar is broadcast over every trial frame in the combined `(5,T)` output.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx
```

iii. Although comments call it static per trial, the AI repeats it across time because the required single output array could not directly mix time series and a scalar.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing NWBs, no cells, no stimulus table, or too few trials cause an experiment to be skipped. Missing pupil produces all NaNs; interpolation outside coverage yields NaN; NaN running/pupil becomes category 0. Unknown images/presentations are left gray, omitted stimuli are gray, unknown outcomes become hit code 0, and image-vocabulary scan errors only warn and continue.

ii.
```python
if not os.path.exists(nwb_path): return None
if stim_data is None: return None
pupil_at_ophys = np.full(len(ophys_ts), np.nan)
result[~valid] = 0
...
except Exception as e:
    print(f"  WARNING: Could not read images from {eid}: {e}")
```

iii. The notes describe bin 0 for missing behavior as a documented design choice and retain any experiment with at least two usable trials.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large dF/F arrays from each NWB dominates, followed by per-trial stimulus mapping and slicing; the separate image-vocabulary scan adds another full pass over NWB files. Timing fields measure load and processing separately.

ii.
```python
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
```

iii. The notes estimate about 1.7 seconds loading and 0.4 seconds processing per experiment and identify direct HDF5 reads as a speed optimization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over every stimulus presentation for every trial, every valid trial, and every NWB for vocabulary collection could be reduced with interval indexing/searchsorted, precomputed presentation ranges, or table-level vectorization. Experiment I/O remains inherently iterative.

ii.
```python
for si in range(len(stim_starts)):
    ...
for trial_idx in valid_trial_idx:
    ...
for _, row in exp_table.iterrows():
```

iii. The AI was instructed to vectorize but does not document these remaining opportunities; it considered the observed full runtime acceptable.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once to collect image names and again for conversion. In multiscope sessions, identical trials, stimulus traces, running, and pupil data are loaded and recomputed independently for every plane. Within one experiment, `build_image_identity_trace` scans the full presentation list anew for each trial, and trial masks are separately rebuilt in helper functions.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
result = process_experiment(exp_id, ...)
...
img_trace, _ = build_image_identity_trace(...)
change_trace = build_image_change_trace(...)
```

iii. The notes mention the 14-second vocabulary pass but do not recognize repeated multiscope behavioral work; treating planes as sessions causes much of this duplication.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `cell_roi_ids` but never uses them; computes and stacks `output_tv` and `output_static` but discards both; creates a returned image-identity trial mask that callers ignore; and computes plotting-only/raw diagnostic objects only when requested. It also scans/stores metadata fields not used by the decoder.

ii.
```python
data['cell_roi_ids'] = seg[key]['id'][:]
...
output_tv = np.stack([...])
output_static = np.array([outcome_idx], dtype=np.int64)
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. No justification is given for the unused arrays; comments show they are remnants of considering mixed static/time-varying output representations.
