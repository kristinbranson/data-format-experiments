# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local ophys experiment CSV, matches rows to locally present NWB filenames, excludes `passive == True`, and opens each selected NWB twice: once to collect image names/behavioral percentiles and once for conversion. It does not restrict `project_code` to `VisualBehavior`.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
exp_table = exp_table[exp_table['passive'] == False]
```

iii. The trajectory says direct `h5py` access “mirrors what the SDK does under the hood”; it chose all locally available active experiments and regarded an NWB experiment as the basic data unit.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique `mouse_id` values, converted to strings; each retained experiment receives the corresponding subject index.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. The agent identified `mouse_id` as the animal identifier. It did not discuss that subjects with no retained experiment could remain in `subjects`.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id` (one imaging plane) is emitted as a separate decoder “session”; experiments sharing an `ophys_session_id` are not grouped.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    neural_all.append(final_neural)
```

iii. The trajectory explicitly decided, “Each experiment is a ‘session’ for our purposes,” despite later research noting that an ophys session can contain multiple planes.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the NWB `intervals/trials` table. For each accepted row, all ophys frames satisfying `start_time <= timestamp < stop_time` form one variable-length trial.

ii.
```python
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. The agent reasoned that full trial windows preserve the multiple pre-change flashes and post-change response and accepted variable-length sequences.

## 1-e. How are trials filtered based on quality controls?

i. It retains only trials flagged go or catch, rejects aborted and auto-rewarded trials, rejects unknown outcomes, trials with fewer than two frames, and experiments with fewer than two retained trials or no valid neurons.

ii.
```python
if nwb_data['trial_aborted'][trial_idx] or nwb_data['trial_auto_rewarded'][trial_idx]: return False
if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]): return False
if outcome == -1 or len(frame_indices) < 2: continue
```

iii. The trajectory follows the explicit go/catch and aborted/auto-reward exclusions and adds defensive outcome/frame/session checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from NWB detected calcium events at `processing/ophys/event_detection/data`, not dF/F. DFF is also loaded but unused.

ii.
```python
events = f['processing/ophys/event_detection/data'][:]
dff = f['processing/ophys/dff/traces/data'][:]
```

iii. The agent believed the paper used discrete/deconvolved calcium events to remove GCaMP decay and therefore preferred events over fluorescence.

## 2-b. How is the `neural` data processed?

i. Event matrices are filtered by ROI validity, sliced to trial frames, transposed from time-by-cell to cell-by-time, and cast to `float32`. No plane merging, normalization, or rebinning occurs.

ii.
```python
events = events[:, valid]
trial_neural = events[frame_indices, :].T
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The agent said native event data was already the desired neural representation and treated each plane independently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only rows whose NWB `valid_roi` flag is true are retained; experiments with zero such cells are skipped.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
if n_neurons == 0: return None, None, None, None, None, None
```

iii. The agent connected this to the SDK default `exclude_invalid_rois=True` and regarded it as appropriate QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are indexed directly by the ophys timestamps falling between trial start and stop. Thus trials start at the first native frame at/after `start_time`; there is no fixed change-centered alignment.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_neural = events[frame_indices, :].T
```

iii. The agent interpreted “align based on ophys timestamp” as using ophys timestamps as the common clock and full experimental trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Each experiment remains at its native imaging rate, while metadata reports one value, `1000 / median(imaging_rate)` across experiments.

ii.
```python
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
```

iii. The trajectory recognized mixed ~11 Hz and ~31 Hz data and the consistency conflict, considered resampling, but ultimately kept native rates and assumed the decoder could cope.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `image_name`, `start_time`, `stop_time`, and `omitted`, rather than trial initial/change image fields.

ii.
```python
stim_names = nwb_data['stim_image_names']
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
```

iii. The agent wanted the actual image presentation at each ophys frame and explicitly distinguished non-grey flashes from inter-stimulus grey periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Globally sorted non-omitted names receive integer codes. Presented frames receive those codes; grey gaps are forward-filled, leading gaps are backfilled from the first image, and trials with no image are skipped.

ii.
```python
all_image_names = sorted(all_image_names_set)
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
elif last_img >= 0: trial_image[k] = last_img
trial_image[:first_valid[0]] = trial_image[first_valid[0]]
```

iii. The trajectory says forward filling represents “the image identity of the image presented during the non-grey screen” continuously through grey intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus intervals are first mapped to the full ophys timestamp vector, then image codes and neural events are sliced using the identical trial frame indices.

ii.
```python
trial_image = image_at_ophys[frame_indices]
trial_neural = events[frame_indices, :].T
```

iii. The common ophys frame index was chosen to guarantee sample-wise alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from stimulus presentation `is_change`, `start_time`, and `stop_time`.

ii.
```python
stim_is_change = nwb_data['stim_is_change']
if stim_is_change[i] == 1.0:
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
```

iii. The agent treated the presentation flagged as a change as the desired immediate post-change period.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector over all ophys frames is set to one throughout each change stimulus presentation interval, then sliced by trial.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
is_change_at_ophys[mask] = 1
trial_change = is_change_at_ophys[frame_indices]
```

iii. No additional rationale beyond representing a sparse time-varying change signal was recorded.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numerical thresholding is needed: the NWB boolean/0–1 `is_change` flag becomes categories 0 (`no_change`) and 1 (`change`).

ii.
```python
output_values = [..., ['no_change', 'change'], ...]
```

iii. The agent regarded the source as already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change intervals are rasterized on ophys timestamps and extracted with the same `frame_indices` as neural activity.

ii.
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. This follows its common-ophys-timebase alignment strategy.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB running `speed/data` and `speed/timestamps`.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. The agent identified these as the wheel-derived locomotion stream exposed by the SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to every ophys timestamp. Global quintile edges are computed in a first pass over entire experiment recordings, and trial values are digitized in a second pass.

ii.
```python
running = np.interp(ophys_ts, signal_ts, signal)
running_edges = compute_percentile_bins(np.concatenate(all_running), n_bins=5)
running_binned = np.digitize(running, running_edges[1:-1])
```

iii. The agent chose interpolation for synchronized streams and global percentiles for consistent, roughly balanced decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five labels 0–4 via `np.digitize`.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(valid, percentiles)
bins = np.digitize(values, edges[1:-1])
```

iii. Equal-percentile categories were required and intended to balance class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated onto the full ophys clock and sliced with the neural trial indices.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
trial_running = running_at_ophys[frame_indices]
```

iii. The trajectory explicitly chose the ophys timestamps as the shared reference frame.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses pupil ellipse `width`, eye-tracking timestamps, and `likely_blink`; pupil area is loaded but unused.

ii.
```python
data['pupil_width'] = pupil['width'][:]
data['eye_timestamps'] = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. The agent selected fitted ellipse width as diameter and used the supplied blink detector to reject artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are changed to NaN; if more than ten valid samples remain, valid samples are linearly interpolated to ophys times. Global quintile edges are computed over full recordings, then trial values are digitized. Missing experiment-level pupil data becomes the median bin 2.

ii.
```python
pupil_raw[blink_mask] = np.nan
pupil = np.interp(ophys_ts, eye_ts[valid_mask], pupil_raw[valid_mask])
pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. The agent intended blink removal to prevent artifacts and median-bin imputation to retain experiments without usable pupil tracking.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Valid full-recording pupil samples define global 20-percentile edges; `np.digitize` maps trial samples to 0–4.

ii.
```python
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
pupil_binned = digitize_to_bins(pupil, pupil_edges)
```

iii. As for running, global equal-percentile bins were chosen for consistent balanced categories.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean pupil samples are interpolated onto the ophys clock and selected using the same trial frame indices as neural activity.

ii.
```python
pupil_at_ophys = np.interp(ophys_ts, eye_ts[valid_mask], pupil_raw[valid_mask])
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. The agent relied on synchronized hardware timestamps and the common ophys grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the four trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if trial_hit: return 0
elif trial_miss: return 1
elif trial_false_alarm: return 2
elif trial_correct_reject: return 3
```

iii. The agent viewed these mutually exclusive NWB flags as the canonical task outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true flag maps to fixed codes 0–3; no recognized flag maps to -1 and causes trial exclusion. The retained code is repeated across every timepoint, although described as static.

ii.
```python
if outcome == -1: continue
np.full(n_t, outcome, dtype=np.int64)
```

iii. Repetition was used so all five outputs fit one `(5, time)` integer matrix expected by the decoder.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil/too few valid samples yields NaNs and later median-bin imputation; blinks are interpolated across. Empty-neuron, short-trial, unknown-outcome, and under-two-trial experiments are skipped. There is no per-experiment exception handler, and `np.interp` extrapolates endpoint values outside behavior timestamp ranges.

ii.
```python
if np.sum(valid_mask) > 10: ...
else: pupil_at_ophys = np.full(len(ophys_ts), np.nan)
if neural_trials is None or len(neural_trials) < 2: continue
```

iii. The agent favored retaining data through interpolation/imputation, with structural sanity assertions after conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated full NWB reads and full-session array loading/interpolation dominate: image-name scan, behavioral-statistics pass, then conversion pass.

ii.
```python
for _, row in exp_table.iterrows():
    nwb_data = load_nwb_data(nwb_path)
# repeated again in the second pass
```

iii. The trajectory monitored the full conversion but did not explicitly profile it; its two-pass design was chosen to obtain global bin edges.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Stimulus-to-frame masks are recomputed in loops, gray image filling is a Python loop, trials are processed serially, and outcome counting uses nested loops/comprehensions.

ii.
```python
for i in range(len(stim_starts)):
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
for k in range(len(trial_image)):
    ...
```

iii. No specific vectorization justification appears; the agent prioritized straightforward code and successful decoder validation.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is reopened for image collection, full behavior collection, and full conversion; `load_nwb_data` additionally reads unused large arrays. Running and pupil interpolation is performed in both passes.

ii.
```python
# First pass
nwb_data = load_nwb_data(nwb_path)
# Second pass
nwb_data = load_nwb_data(nwb_path)
```

iii. The first pass was justified by the need for global percentiles, but the agent did not acknowledge the avoidable repeated I/O/interpolation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads dF/F, event timestamps, cell specimen IDs, pupil area, trial change times/fields, and full-session event/behavior outputs that are unused or returned then discarded. It also scans all images separately despite loading stimulus names later.

ii.
```python
data['dff'] = f['processing/ophys/dff/traces/data'][:]
data['pupil_area'] = pupil['area'][:]
return ..., running_at_ophys, pupil_at_ophys, events
```

iii. DFF was described as “fallback / comparison”; the remaining discarded work was not justified in the trajectory.
