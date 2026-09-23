# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading local NWB files directly via `BehaviorOphysExperiment.from_nwb_path()`. It first reads the `ophys_experiment_table.csv` metadata CSV, cross-references with locally available NWB files, filters to active (non-passive) familiar-image sessions, then loads each experiment file individually. This bypasses the AllenSDK cache system entirely.

ii.
```python
def get_experiment_table():
    tab = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = [int(os.path.basename(f).split('_')[-1].split('.')[0])
             for f in glob.glob(os.path.join(EXP_DIR, '*.nwb'))]
    tab = tab[tab.ophys_experiment_id.isin(local)]
    tab = tab[(~tab.passive) & (tab.experience_level == 'Familiar')]
    return tab.sort_values(['ophys_session_id', 'ophys_experiment_id'])

# In process_session:
path = os.path.join(EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
```

iii. The AI chose direct NWB loading because the data was already available locally. It explored the AllenSDK cache API first but found direct NWB loading simpler and faster. The filtering to active familiar sessions was justified by the Vip-Sst paper's methodology which restricted analysis to familiar image-set sessions during active behavior.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. Each mouse that has at least one valid processed session is added to the subjects list.

ii.
```python
if res['mouse'] not in data['subjects']:
    data['subjects'].append(res['mouse'])
data['subject_idx'].append(data['subjects'].index(res['mouse']))
```

iii. The `mouse_id` field from the metadata table uniquely identifies each animal. Subjects are added dynamically as sessions are processed.

## 1-c. How are the data split into sessions?

i. Sessions correspond to unique `ophys_session_id` values. Multiple experiments (imaging planes) with the same `ophys_session_id` are grouped and merged into a single session. The experiment table is grouped by `ophys_session_id`.

ii.
```python
sessions = list(tab.groupby('ophys_session_id'))
# In process_session:
for _, row in exp_rows.iterrows():
    path = os.path.join(EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
    exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
```

iii. Grouping by `ophys_session_id` ensures that multiple imaging planes recorded simultaneously in a multi-plane (Mesoscope) session are treated as a single session with merged neural populations, since they share the same behavioral data.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `trials` table. Go and catch trials are included; aborted and auto-rewarded trials are excluded. Trials must have a valid `change_time`. Each trial is aligned to the change time (or sham change time for catch trials) with a fixed window of [-2.0, +4.0] seconds, yielding 60 time bins of 100ms each.

ii.
```python
trials = ex0.trials
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
# ...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The AI justified go+catch trial selection as matching the task instructions. The fixed window [-2, +4]s was chosen to fit inside every trial (confirmed by surveying that minimum pre-change time was 2.79s and post-change window is ~4.2s). Alignment to change time was chosen because it is the key behavioral event in the change detection task.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials without a valid `change_time` are excluded. Trials with an unrecognized outcome (not hit/miss/false_alarm/correct_reject) are dropped via a `keep` filter. Sessions with fewer than 2 valid trials, or sessions where running/pupil data is entirely NaN, are skipped.

ii.
```python
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
# ...
keep = [k for k in range(ntrials) if out_trials[k] >= 0]
if len(keep) < 2:
    return None
```

iii. Excluding aborted and auto-rewarded trials follows the task instructions. The minimum 2-trial threshold ensures sessions can be used for decoder training/validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `filtered_events` — the AllenSDK's detected calcium events convolved with a half-normal (half-Gaussian) filter. This is accessed via `ex.events` and the `filtered_events` column.

ii.
```python
EVENTS_COL = 'filtered_events'
# In process_session:
ev = ex.events
traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
```

iii. The AI initially tried raw detected events but found that `filtered_events` (the SDK-recommended smoothed version) gave substantially better decoder performance (image identity validation accuracy 0.33 vs 0.25). The SDK tutorials recommend filtered_events as the standard neural signal for analysis. The Vip-Sst paper also used detected calcium events.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes three processing steps: (1) calcium events are summed into 100ms bins using a cumulative sum approach, (2) multiple imaging planes within a session are concatenated into a single neural population, and (3) each neuron is normalized by its standard deviation across all extracted bins of the session.

ii.
```python
def bin_events_matrix(traces, timestamps, t_edges):
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)], axis=1)
    return cs[:, idx[1:]] - cs[:, idx[:-1]]

# Per-neuron normalization:
sd = neural_all.reshape(neural_all.shape[0], -1).std(axis=1)
sd[sd == 0] = 1.0
neural_all = (neural_all / sd[:, None, None]).astype(np.float32)
```

iii. The 100ms binning was chosen to provide a consistent time resolution across single-plane (31 Hz) and multi-plane (10.7 Hz) sessions. Per-neuron SD normalization prevents high-variance cells from dominating the decoder's PCA projection, since event magnitudes vary by orders of magnitude across cells.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons beyond what the AllenSDK already performs. A guard against timestamp-trace length mismatches truncates to the minimum length.

ii.
```python
n = min(traces.shape[1], len(ts))
traces, tss = traces[:, :n], ts[:n]
```

iii. The AllenSDK pipeline already applies ROI filtering and quality control. The length mismatch guard handles edge cases in the NWB files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the change time (or sham change time for catch trials). A fixed window of [-2.0, +4.0] seconds relative to the change time is used. The calcium events are summed within 100ms bins defined by this window.

ii.
```python
OFF_START = -2.0
OFF_END = 4.0
BIN_SIZE = 0.1
# ...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The instructions say "temporally align based on ophys timestamp." The AI aligned to change_time, which is the key behavioral event. The [-2, +4]s window was chosen to capture both pre-change stimulus presentations and the full post-change response period while fitting within every trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned into 100ms bins. The native ophys frame rate varies between ~31 Hz (single-plane) and ~10.7 Hz (multi-plane), so rebinning to a common 100ms resolution provides consistent time bins across all sessions.

ii.
```python
BIN_SIZE = 0.1  # seconds
# bin_edges():
n = int(round((OFF_END - OFF_START) / BIN_SIZE))
edges = OFF_START + BIN_SIZE * np.arange(n + 1)
```

iii. The 100ms bin size was chosen to unify the temporal resolution across single-plane and multi-plane sessions. The metadata reports `time_bin_size: 100.0` ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `start_time` and `image_name` columns of non-omitted flashes within the `change_detection` stimulus block.

ii.
```python
sp = ex0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
flash = sp[~sp.omitted.astype(bool)]
flash_start = flash.start_time.values.astype(float)
flash_img = np.array([image_names.index(n) for n in flash.image_name.values])
```

iii. The stimulus_presentations table provides the precise timing and identity of each image flash, enabling accurate assignment of image identity at each time bin. A global list of 8 familiar images (image set A) is used for consistent categorical encoding across sessions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin center, the most recent flash is identified via `np.searchsorted`. The image identity is set to the image name of that most recent flash. Image names are mapped to integer codes 0-7 using a fixed global list of 8 images.

ii.
```python
image_names = ['im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077', 'im085']
# In process_session:
j = np.searchsorted(flash_start, tc, side='right') - 1
j = np.clip(j, 0, len(flash_start) - 1)
img_trials.append(flash_img[j].astype(np.int64))
```

iii. The image identity is the identity of the currently or most recently presented image. Using stimulus_presentations ensures that image changes within a trial are captured at the correct time. The image name list is hardcoded to the 8 familiar (set A) images since all filtered sessions use this set.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each bin center within the trial window, which are the same bin centers used for the neural data. Both share the same temporal grid defined by `OFF_START + BIN_SIZE * k + BIN_SIZE/2`.

ii.
```python
edges_rel, centers_rel = bin_edges()
tc = ct + centers_rel
j = np.searchsorted(flash_start, tc, side='right') - 1
```

iii. By computing image identity at the same bin centers as the neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically the `is_change` column and `start_time` of change presentations.

ii.
```python
chg = sp[sp.is_change.astype(bool)]
change_starts = chg.start_time.values.astype(float)
```

iii. The `is_change` flag in stimulus_presentations identifies actual image changes. This includes only real changes (not catch/sham changes).

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each time bin center, the code checks whether the bin falls within 750ms of the most recent actual image change. If so, image_change is 1; otherwise 0. The 750ms window corresponds to one flash interval (250ms image + 500ms gray).

ii.
```python
i = np.searchsorted(change_starts, tc, side='right') - 1
is_chg = np.zeros(T, dtype=np.int64)
valid = i >= 0
is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
```

iii. The 750ms window matches the image presentation interval used in the experiment.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 = no change, 1 = change). No thresholding is needed beyond the 750ms window criterion.

ii. See 4-b.

iii. The binary encoding directly represents whether a change event is occurring at each time bin.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same bin centers as the neural data.

ii. See 4-b.

iii. Same temporal grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ex0.running_speed`, using `timestamps` and `speed` columns.

ii.
```python
run = ex0.running_speed
run_t = run.timestamps.values.astype(float)
run_v = run.speed.values.astype(float)
```

iii. The SDK's `running_speed` attribute provides the standard locomotion data from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 100ms bin using `bin_mean`, then NaN values are filled via linear interpolation (`fill_nans`). The resulting values are then discretized into 5 equal-percentile bins computed per-session.

ii.
```python
r = fill_nans(bin_mean(run_v, run_t, te))
# ...
run_q = quantize([run_trials[k] for k in keep])
```

iii. Bin-averaging aligns the running speed to the same temporal grid as the neural data. NaN filling handles bins with no running speed samples. Percentile-based quantization ensures roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins. The quantile edges are computed per-session from all trials within that session.

ii.
```python
def quantize(values, nq=NQUANTILES):
    v = np.concatenate([np.ravel(a) for a in values])
    edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
    return [np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
            for a in values]
```

iii. Per-session quantization adapts to session-specific distributions of running behavior.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is averaged within the same time bins used for neural data, ensuring alignment. Both use the same `t_edges` defined by `change_time + edges_rel`.

ii.
```python
te = ct + edges_rel
r = fill_nans(bin_mean(run_v, run_t, te))
```

iii. Using the same bin edges guarantees temporal alignment between running speed and neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ex0.eye_tracking`, using `pupil_area`, `likely_blink`, and `timestamps` columns. Diameter is computed from area as `2 * sqrt(area / pi)`.

ii.
```python
eye = ex0.eye_tracking
area = eye.pupil_area.values.astype(float)
blink = eye.likely_blink.values.astype(bool)
area = np.where(blink, np.nan, area)
diam = 2.0 * np.sqrt(area / np.pi)
```

iii. The AI chose `pupil_area` and computed diameter via the circle formula, rather than using `pupil_width` directly. Blink frames are set to NaN before computing diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter via `2*sqrt(area/pi)`. Blink frames are set to NaN, then NaN values are interpolated. The resulting diameter is averaged within each 100ms bin, NaN-filled again, then discretized into 5 equal-percentile bins per-session.

ii.
```python
area = np.where(blink, np.nan, area)
diam = 2.0 * np.sqrt(area / np.pi)
diam = fill_nans(diam)
# ...
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. Converting area to diameter provides a linear measure of pupil size. NaN interpolation bridges blink gaps. Per-session quantization adapts to individual session distributions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal-percentile bins computed per-session.

ii. See `quantize` function in 5-c.

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — pupil diameter is averaged within the same time bins used for neural data.

ii.
```python
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
```

iii. Same bin edges ensure temporal alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
tr = sel.iloc[k]
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
else:
    o = -1
```

iii. These four outcomes are the canonical trial categories for the go/no-go change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes 0-3 via an if/elif chain. Trials with an unrecognized outcome get code -1 and are excluded from the final dataset. The outcome code is replicated across all time bins within a trial (static per-trial variable).

ii.
```python
out_trials.append(o)
# ...
keep = [k for k in range(ntrials) if out_trials[k] >= 0]
# ...
np.full(T, out_trials[k], dtype=np.int64)
```

iii. The fixed mapping ensures consistent encoding. Excluding unrecognized outcomes (-1) is a safety measure, though in practice all go/catch trials should have one of the four canonical outcomes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: If `process_session` raises an exception, the session is skipped with a warning.
- **Missing eye tracking**: If eye tracking is empty or all NaN after blink removal, the session is skipped (returns None).
- **Missing running/pupil in bins**: NaN values from bins with no samples are filled via linear interpolation (`fill_nans`). If interpolation fails (all NaN), the session is skipped.
- **Trace-timestamp mismatch**: Neural traces are truncated to match timestamp length.
- **Unrecognized outcomes**: Trials with outcome code -1 are excluded.
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.

ii.
```python
try:
    res = process_session(sid, rows, image_names)
except Exception as e:
    print(f'session {sid} failed: {e}')
    continue
if res is None:
    continue
# ...
if pupil_v is None:
    return None
# ...
n = min(traces.shape[1], len(ts))
```

iii. The approach is conservative: any session with insufficient data quality is skipped rather than filled with defaults.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB experiment file via `BehaviorOphysExperiment.from_nwb_path()`. Each file is ~1GB and contains full-session neural, behavioral, and stimulus data. The full conversion of 92 sessions took approximately 1.5 hours.

ii. N/A

iii. I/O-bound loading of large NWB files dominates runtime, with each session taking ~13 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial to compute outputs (image identity, image change, running, pupil). The neural binning was initially per-cell per-trial but was vectorized using cumulative sums (`bin_events_matrix`). The remaining per-trial output computations could potentially be batched but are not the bottleneck.

ii.
```python
def bin_events_matrix(traces, timestamps, t_edges):
    """Sum of calcium event magnitudes in each bin, for all cells at once."""
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)], axis=1)
    return cs[:, idx[1:]] - cs[:, idx[:-1]]
```

iii. The AI explicitly identified and fixed the neural binning bottleneck during development, vectorizing from per-cell to all-cells-at-once using cumulative sums.

## 9-c. What processing does the code repeat multiple times?

i. The code does not repeat processing. Each session is loaded and processed once. However, the `quantize` function is called per-session (within `process_session`), computing quantile edges independently for each session rather than once globally.

ii.
```python
run_q = quantize([run_trials[k] for k in keep])
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. Per-session quantization is a design choice rather than repeated processing. It means each session gets its own bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `session_info` metadata for each session (ophys_session_id, experiment_ids, cre_line, etc.) which is stored in metadata but not used by the decoder. The `fill_nans` function performs linear interpolation on pupil data before binning, though some of this interpolation may be redundant after bin_mean already handles NaNs.

ii.
```python
info = {
    'ophys_session_id': int(session_id),
    'ophys_experiment_ids': [int(e) for e in exp_rows.ophys_experiment_id],
    'mouse_id': str(row0.mouse_id),
    'cre_line': str(row0.cre_line),
    # ...
}
```

iii. The session_info is useful for debugging and provenance but not consumed by the decoder training pipeline.
