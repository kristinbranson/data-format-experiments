# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads project metadata CSVs (`ophys_experiment_table.csv`) from the local data directory, then loads each NWB file individually using `BehaviorOphysExperiment.from_nwb_path(path)`. It does not use the SDK's cache API (`VisualBehaviorOphysProjectCache`) because the local data copy lacks the manifests directory required by the cache. Experiments are filtered to `project_code == 'VisualBehavior'` (single-plane Scientifica rigs), and passive sessions are excluded. Sessions are processed in parallel using `multiprocessing.Pool`.

ii.
```python
def select_experiments():
    et = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = {int(re.findall(r'(\d+)', f)[0]): os.path.join(NWB_DIR, f)
             for f in os.listdir(NWB_DIR) if f.endswith('.nwb')}
    et = et[et.ophys_experiment_id.isin(local.keys())].copy()
    et['path'] = et.ophys_experiment_id.map(local)
    et = et[~et.session_type.str.contains('passive')]
    et = et[et.project_code == 'VisualBehavior']
    ...

def process_session(args):
    path, want_raw, signal, normalize = args
    ds = BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that `VisualBehaviorOphysProjectCache.from_local_cache` raises an error due to missing manifests, so it falls back to `BehaviorOphysExperiment.from_nwb_path`, which returns the identical object the cache would have returned. The metadata CSVs are read with pandas for experiment selection.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata. Unique mouse IDs are collected from the successfully processed sessions and sorted.

ii.
```python
subjects = sorted(set(r['mouse'] for r in good))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. `mouse_id` is the SDK's unique animal identifier. The AI documented 37 mice in the active single-plane subset.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a single NWB file (one `ophys_experiment_id`). Since only VisualBehavior (single-plane) experiments are kept, each experiment IS a session (1 imaging plane per session). Sessions are sorted by mouse_id then date_of_acquisition.

ii.
```python
et = et.sort_values(['mouse_id', 'date_of_acquisition']).reset_index(drop=True)
```

iii. The AI documented that for the single-plane VisualBehavior data, "experiment == session" so there is no ambiguity. Multiscope multi-plane sessions are excluded entirely (Step 5 decision 1).

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. Only go and catch trials are retained (filtering out aborted and auto-rewarded). Each trial spans from `start_time` to `stop_time` (variable length, ~8s). Ophys frame indices are found via `np.searchsorted`.

ii.
```python
trials = ds.trials
sel = trials[(trials.go.values | trials.catch.values)].copy()
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
keep = (i1 - i0) > 1
```

iii. The AI noted that `go | catch` is equivalent to `~aborted & ~auto_rewarded` per the SDK's mutually exclusive trial taxonomy, as verified in the SDK source code (`Trial._get_trial_data`). This is confirmed by assertions in the code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) only go or catch trials kept (no aborted/auto-rewarded), (2) trials with fewer than 2 ophys frames are dropped, (3) sessions without eye tracking data are skipped entirely, (4) sessions with fewer than 100 valid pupil samples are skipped, (5) sessions with fewer than 2 usable trials are skipped. Additionally, passive sessions and Multiscope sessions are excluded at the experiment selection level.

ii.
```python
sel = trials[(trials.go.values | trials.catch.values)].copy()
keep = (i1 - i0) > 1
...
try:
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError('empty')
except Exception as exc:
    res['skip'] = f'no eye tracking ({exc})'
    return res
...
if good_eye.sum() < 100:
    res['skip'] = 'pupil data all NaN'
    return res
...
if n_trials < 2:
    res['skip'] = f'only {n_trials} usable trials'
    return res
```

iii. The AI documented that 3 of 168 sessions lacked eye tracking entirely and were dropped (Step 5 decision 8), leaving 165 sessions. The minimum trial requirement ensures decoder training has at least train and validation data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces.dff` (dF/F calcium fluorescence traces) from the NWB file, accessed via `ds.dff_traces.dff.values`.

ii.
```python
if signal == 'dff':
    neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
```

iii. The AI tested dF/F, events, and filtered_events, and chose dF/F because it gave substantially better decoder accuracy (Step 5 decision 3). The paper used detected events, but the AI justified the difference using the task's exception clause ("except where training a neural decoder require otherwise").

## 2-b. How is the `neural` data processed?

i. The dF/F traces are stacked into a (n_neurons, T) matrix and cast to float32. No additional normalization is applied (default `--normalize none`). The SDK pipeline has already applied motion correction, neuropil subtraction, and dF/F normalization.

ii.
```python
neural_full = np.vstack(ds.dff_traces.dff.values).astype(np.float32)
assert neural_full.shape[1] == ots.size
# no normalization by default
```

iii. The AI tested per-neuron z-scoring and found it changed accuracy by <=0.02, so raw dF/F values are kept. The valid_roi assertion confirms all ROIs are valid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. All neurons present in the NWB (all with `valid_roi == True`) are included. The AI verified that `cst.valid_roi.all()` holds in every session.

ii.
```python
cst = ds.cell_specimen_table
assert bool(cst.valid_roi.all()), 'invalid ROIs present in released data'
assert len(cst) == neural_full.shape[0]
```

iii. The AI documented that ROI filtering was already applied upstream by the Allen pipeline, and neither reference paper applies further cuts. Verified 29,097/29,097 valid ROIs across the 168 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start (`start_time`). The ophys frames from `start_time` to `stop_time` are extracted using `np.searchsorted`, giving a variable-length window per trial.

ii.
```python
i0 = np.searchsorted(ots, sel.start_time.values, side='left')
i1 = np.searchsorted(ots, sel.stop_time.values, side='left')
...
neural.append(np.ascontiguousarray(neural_full[:, a:b]))
```

iii. The alignment event is documented as `trial_start` with `off_start = 0.0` and `off_end = None` (variable length). The AI verified that consecutive trials never overlap.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Data is kept at the native ophys frame rate (~31 Hz, ~32.32 ms per frame). The time bin size is computed as the mean of the median frame intervals across all sessions.

ii.
```python
dt = float(np.mean([r['dt_median'] for r in good]))
'time_bin_size': dt * 1000.0,  # in ms
```

iii. The AI documented that all VisualBehavior sessions have consistent 31 Hz frame rate (median dt = 32.31-32.33 ms), so no resampling is needed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations` — specifically the `image_name`, `start_time`, `omitted`, and `is_change` columns from the `change_detection_behavior` stimulus block.

ii.
```python
stim = ds.stimulus_presentations
flashes = stim[stim.stimulus_block_name == 'change_detection_behavior'].copy()
f_start = flashes.start_time.values.astype(np.float64)
f_omitted = flashes.omitted.values.astype(bool)
f_name = flashes.image_name.values.astype(str)
```

iii. The AI uses the stimulus presentation schedule rather than the trials table's `initial_image_name`/`change_image_name`. This allows proper handling of omitted flashes (which carry the ongoing image forward) and provides frame-level image identity across the entire session.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is built from the stimulus schedule: each ophys frame is mapped to its enclosing flash interval via `np.searchsorted`. Omitted flashes are forward-filled so they inherit the most recent non-omitted image. A per-session local image vocabulary is built, then remapped to a global 16-image vocabulary during assembly.

ii.
```python
local_idx = np.array([img_to_local.get(n, -1) for n in f_name], dtype=np.int16)
carry = local_idx.copy()
for i in range(1, carry.size):
    if f_omitted[i]:
        carry[i] = carry[i - 1]
fi = np.searchsorted(f_start, ots, side='right') - 1
fi_clipped = np.clip(fi, 0, f_start.size - 1)
image_per_frame = carry[fi_clipped]
...
# During assembly:
remap = np.array([image_vocab.index(n) for n in r['images']], dtype=np.int8)
for o in r['output']:
    o[0] = remap[o[0]]
```

iii. The AI followed the paper's definition: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission." The forward-fill ensures image identity changes exactly at image changes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for every ophys frame in the session (via the searchsorted mapping from ophys timestamps to flash intervals), then sliced per trial using the same frame indices as the neural data.

ii.
```python
fi = np.searchsorted(f_start, ots, side='right') - 1
image_per_frame = carry[fi_clipped]
...
out[0] = image_per_frame[a:b]
```

iii. By building the image identity as a session-length array indexed by ophys frames, alignment with neural data is guaranteed — both are sliced with the same `[a:b]` range.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, which marks flashes where the image identity actually changed.

ii.
```python
f_change = flashes.is_change.values.astype(bool)
change_per_frame = f_change[fi_clipped].astype(np.int8)
change_per_frame[fi < 0] = 0
```

iii. The AI used the stimulus table's `is_change` flag, which directly identifies change flashes. This is 1 for the 750ms flash interval at a real image change and 0 otherwise.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Each ophys frame is mapped to its enclosing flash interval, and the `is_change` flag of that flash is assigned to the frame. Frames before the first flash of the behavior block are set to 0. The result is a binary indicator that is 1 during the 750ms image presentation interval of a change flash.

ii.
```python
change_per_frame = f_change[fi_clipped].astype(np.int8)
change_per_frame[fi < 0] = 0
```

iii. The 750ms window is implicit in the flash interval structure (each flash lasts 250ms image + 500ms grey = 750ms). On catch/sham-change trials, `is_change` is False, so image_change is 0.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1). No thresholding is needed. Output values are `['no_change', 'change']`.

ii.
```python
'output_values': [
    ...
    ['no_change', 'change'],
    ...
]
```

iii. N/A — inherently binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed for every ophys frame in the session via the flash-interval mapping, then sliced per trial with the same frame indices as neural data.

ii.
```python
out[1] = change_per_frame[a:b]
```

iii. Same frame-level alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
run = ds.running_speed
run_t = run.timestamps.values.astype(np.float64)
run_v = run.speed.values.astype(np.float64)
```

iii. The `running_speed` attribute provides the SDK's filtered (10 Hz low-pass) running speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite values are dropped, then running speed is linearly interpolated onto the ophys frame times using `np.interp`. The interpolated values are then discretized into 5 equal-percentile bins computed per session over the retained trial frames.

ii.
```python
good = np.isfinite(run_v)
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
...
def percentile_bins(x):
    edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int8), edges
speed_bin_all, speed_edges = percentile_bins(speed_per_frame[frame_idx])
```

iii. The AI uses per-session percentile bins (justified in Step 12 Check 1 as necessary because pupil measurements in camera pixels vary across rigs/mice). Each bin has exactly 20% occupancy per session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (quintiles) computed per session. The `np.digitize` function maps continuous values to bin indices 0-4 using the inner percentile edges (20th, 40th, 60th, 80th percentiles).

ii.
```python
def percentile_bins(x):
    edges = np.percentile(x, np.linspace(0, 100, N_BEHAVIOR_BINS + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int8), edges
```

iii. Per-session quintiles ensure balanced classes within each session. The AI justified this over global bins by noting that pupil measurements vary across rigs and mice, and the same logic was applied to running speed for consistency.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the ophys frame times before trial segmentation, so it shares the same time grid as neural data. Per-trial slicing uses the same frame indices.

ii.
```python
speed_per_frame = np.interp(ots, run_t[good], run_v[good])
...
out[2] = speed_bin[a:b]
```

iii. Hardware-synced clocks ensure valid interpolation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area`, converted to diameter via `2 * sqrt(area / pi)`. Blink frames (where pupil_area is NaN, set by the SDK's `likely_blink` processing) are excluded before interpolation.

ii.
```python
eye_t = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)
good_eye = np.isfinite(pupil_area)
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
```

iii. The AI noted that `pupil_area` is used (converted to diameter) rather than `pupil_width`. Since percentile binning is invariant to monotone transforms, the choice between area and diameter doesn't affect bin assignments. Blink frames are already NaN in the released data due to the SDK's `filter_on_blinks` processing.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink NaN frames are dropped, pupil area is converted to diameter, then linearly interpolated onto ophys frame times using `np.interp`. The interpolated values are discretized into 5 per-session equal-percentile bins.

ii.
```python
pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
...
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
```

iii. Same processing pipeline as running speed: interpolation to ophys grid, then per-session percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 per-session equal-percentile bins, identical to running speed. The AI explicitly tested per-session vs. dataset-wide binning and justified per-session (Step 12 Check 1).

ii.
```python
pupil_bin_all, pupil_edges = percentile_bins(pupil_per_frame[frame_idx])
```

iii. The AI showed that dataset-wide bins give inflated accuracy (0.445 vs 0.259) because whole sessions fall into single bins due to rig-dependent pupil pixel sizes (CAM2P.3 mean 98.6 px vs CAM2P.5 81.7 px), allowing the decoder to score by session identification rather than tracking arousal.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto ophys frame times, then sliced per trial.

ii.
```python
pupil_per_frame = np.interp(ots, eye_t[good_eye], pupil_diam[good_eye])
...
out[3] = pupil_bin[a:b]
```

iii. Same frame-level alignment as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
oc = np.full(len(sel), -1, dtype=np.int8)
for k, name in enumerate(OUTCOME_NAMES):
    oc[sel[name].values.astype(bool)] = k
assert np.all(oc >= 0), 'go/catch trial without an outcome'
```

iii. The AI verified that these four outcomes are mutually exclusive and exhaustive on go/catch trials via assertions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) corresponding to hit/miss/false_alarm/correct_reject. The code is constant across all timepoints within a trial.

ii.
```python
out[4] = oc[k]  # scalar broadcast to all timepoints
```

iii. The AI verified that hit/miss are always go trials and FA/CR are always catch trials, and that each trial has exactly one outcome.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 sessions with no eye tracking are skipped entirely (pupil is a required output).
- **Insufficient pupil data**: Sessions with fewer than 100 valid (non-NaN) pupil samples are skipped.
- **Short trials**: Trials with fewer than 2 ophys frames are dropped.
- **Few trials**: Sessions with fewer than 2 usable trials are skipped.
- **Blink frames**: Pupil NaN frames are dropped before interpolation; `np.interp` bridges the gaps.
- **Non-finite running speed**: Non-finite values dropped before interpolation.

ii.
```python
# No eye tracking
try:
    eye = ds.eye_tracking
    if eye is None or len(eye) == 0:
        raise ValueError('empty')
except Exception as exc:
    res['skip'] = f'no eye tracking ({exc})'
    return res

# Insufficient pupil data
if good_eye.sum() < 100:
    res['skip'] = 'pupil data all NaN'
    return res

# Short trials
keep = (i1 - i0) > 1

# Few trials
if n_trials < 2:
    res['skip'] = f'only {n_trials} usable trials'
    return res
```

iii. The AI documented all edge cases and their handling in CONVERSION_NOTES.md. The approach is conservative: skip problematic sessions rather than impute entire missing streams.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()` (~2.4s per file). Total per-session CPU time is ~3.2s. The AI mitigates this with multiprocessing (24 workers), reducing wall time to ~49s for 168 sessions.

ii. N/A

iii. The AI documented detailed timing: 2.4s NWB load + 0.8s processing = 3.2s per session; 49s wall with 24 workers; 13s for assembly + pickle write; 56s total.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The forward-fill loop for omitted flashes (line 170-171) iterates sequentially through flashes. This could use `pd.Series.ffill()` or a vectorized approach. However, the loop is over ~3,400 flashes (small), not the bottleneck.

ii.
```python
carry = local_idx.copy()
for i in range(1, carry.size):
    if f_omitted[i]:
        carry[i] = carry[i - 1]
```

iii. The per-trial cutting loop is minimal (just array slicing) and the AI already vectorized the heavy operations (searchsorted, interp, digitize) at the session level.

## 9-c. What processing does the code repeat multiple times?

i. The per-session sanity check (`_check_session`) re-derives some quantities (e.g., rebuilds `img_to_local` mapping) that were already computed in `process_session`. However, this is intentional for validation purposes and not a performance concern.

ii.
```python
def _check_session(res, sel, oc, i0, i1, ots, output, images):
    img_to_local = {n: i for i, n in enumerate(images)}  # rebuilt
```

iii. The duplication serves validation rather than production processing.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extensive per-session metadata (`session_info` list with cell_specimen_ids, trial times, bin edges, etc.) that is not used by the decoder. The `--show-processing` raw data is also computed for candidate sessions even when not needed in full mode.

ii.
```python
session_info.append({
    'ophys_experiment_id': r['eid'], ...
    'cell_specimen_ids': r['cell_specimen_ids'],
    'trial_start_times': r['trial_start'], ...
})
```

iii. The extra metadata aids debugging and reproducibility but is not used by the decoder training script.
