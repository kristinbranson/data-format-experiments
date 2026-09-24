# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI ignored the Allen SDK project cache / `get_ophys_experiment_table()` route used by the reference and instead globbed the local NWB release directly. `list_session_files()` sorts every `*.nwb` file in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files) and, in `--full` mode, processes all of them; `--sample` takes the first 2. Each file is opened with `pynwb.NWBHDF5IO` and wrapped into an AllenSDK `BehaviorOphysExperiment` via `BehaviorOphysExperiment.from_nwb(nwbfile)`, which then exposes `dff_traces`, `ophys_timestamps`, `trials`, `stimulus_presentations`, `running_speed` and `eye_tracking` exactly as the SDK cache route would. No project-metadata CSV is consulted at run time, so no filtering on `project_code` is applied: all 284 local experiments are taken, which in fact comprises 239 `VisualBehavior` (single-plane, ~31 Hz) and 45 `VisualBehaviorMultiscope` (~11 Hz) experiments. All cells present in `dff_traces` are loaded (42,147 total, matching `ophys_cells_table.csv` for those 284 experiment ids exactly). The NWB `io` handle is stashed on the experiment object (`exp._nwb_io = io`) and never closed.

ii.
```python
DATA_ROOT = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_ROOT / 'behavior_ophys_experiments'

def list_session_files(sample=False):
    files = sorted(NWB_DIR.glob('*.nwb'))
    return files[:2] if sample else files

def load_experiment(nwb_path):
    io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
    nwbfile = io.read()
    exp = BehaviorOphysExperiment.from_nwb(nwbfile)
    exp._nwb_io = io
    return exp
```
```python
files = list_session_files(sample=sample)
print(f'processing {len(files)} sessions')
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
```

iii. From CONVERSION_NOTES.md Step 4 / Step 5 Key Decision 1: *"Convert only locally available NWB experiment files, not the full project metadata release, because only those sessions are actually present."* The AI explicitly recorded the discrepancy that "project metadata describe a larger release (1936 experiments, 107 subjects) than the locally available NWB subset" and resolved it by converting only what is on disk. Trajectory step 22 shows the loader was corrected after `from_nwb` initially failed on a path string. No justification is given anywhere for including both project codes — the AI never examined `project_code` at all, and the two-frame-rate consequence is not documented.

## 1-b. How are the data split into subjects?

i. Subject identity is read per experiment from the AllenSDK metadata dictionary (`exp.metadata['mouse_id']`) and stored on the session record. After all sessions are processed, the unique mouse ids are sorted to form `subjects`, and `subject_idx` maps each converted session to its index in that list. This produced 38 subjects (37 `VisualBehavior` mice plus one extra mouse, 457841, contributed only by the Multiscope experiments).

ii.
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))
...
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: *"metadata mouse_id → subjects / subject_idx: Map unique mouse IDs to subject list and per-session indices. Session order follows converted session list."* `mouse_id` is the SDK's canonical animal identifier, so no further derivation is needed.

## 1-c. How are the data split into sessions?

i. The AI defines one "session" as one NWB *experiment* file, i.e. one imaging plane. There is no grouping by `ophys_session_id`. Each file becomes one entry in `data['neural']`/`data['output']`/`subject_idx`/`brain_region_idx`, giving 284 sessions. For the 239 single-plane `VisualBehavior` experiments this is a 1:1 correspondence with a behavioural session, but the 45 `VisualBehaviorMultiscope` experiments come from only 8 physical sessions, so those behavioural sessions appear 4–8 times each as separate "sessions" carrying identical trial timing, identical running/pupil/image/outcome labels, and only a different subset of neurons. This is directly visible in the verification log as repeated trial counts (`209` ×7, `287` ×7, `309` ×7, `406` ×7, `196` ×5, `239` ×5, …) and in mouse 457841 being credited with 45 sessions.

ii.
```python
sessions = []
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
...
data = {
    'neural': [s['neural'] for s in sessions],
    ...
}
```

iii. No explicit justification is recorded. CONVERSION_NOTES.md Step 2 treats "284 local NWB sessions" as the session count throughout, and Step 5 says "Session order follows converted session list" — i.e. the AI simply equated file = session and never checked whether multiple files share an `ophys_session_id`. The `ophys_experiment_table.csv` that contains `ophys_session_id` was read during Step 2 exploration but not used by the conversion script.

## 1-d. How are the data split into trials?

i. Trials come from the SDK-provided `exp.trials` table. For each retained row, the trial window is `[start_time, stop_time)` and the included ophys frames are those whose `ophys_timestamps` fall in that half-open interval (`ts >= start & ts < stop`). Trial length is therefore variable (224–391 frames at 31 Hz; 77–~130 frames at 11 Hz) rather than a fixed window around the change. `stop_time` is used if present, with a fallback to `end_time`.

ii.
```python
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
    trial_ts = ts[idx]
    trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: *"Use experiment-defined trial intervals from the trial table; include Go and Catch trials, exclude Aborted and Auto-rewarded trials."* This follows the Decoder Task instruction to "segment each recording session into individual trials based on how they are defined in the experiment". Using the full trial window (rather than a window locked to the change) is what makes the time-varying image-identity and image-change outputs meaningful.

## 1-e. How are trials filtered based on quality controls?

i. `valid_trials_table()` drops any row flagged `aborted` or `auto_rewarded` (it also tolerates alternative column spellings `auto_rewarded_trial` / `is_auto_rewarded`, and `.fillna(False)` guards against missing flags), then keeps only rows where `go | catch` is true (falling back to a `trial_type` string match if the boolean columns are absent). At the frame level, a trial is skipped if it contains fewer than 2 ophys frames. There is no `change_time.notna()` requirement, no minimum-trials-per-session filter, and no session-level exclusion of passive sessions (82 of the 284 experiments are `..._passive`, where the lick spout is retracted; these dominate the `miss` class at 68.8% of the trial-outcome distribution).

ii.
```python
def valid_trials_table(trials):
    df = trials.copy()
    for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
        if col in df.columns:
            df = df[~df[col].fillna(False)]
    keep = None
    if 'go' in df.columns and 'catch' in df.columns:
        keep = df['go'].fillna(False) | df['catch'].fillna(False)
    elif 'trial_type' in df.columns:
        keep = df['trial_type'].astype(str).str.lower().isin(['go', 'catch'])
    if keep is not None:
        df = df[keep]
    return df.reset_index(drop=True)
```
```python
    if idx.size < 2:
        continue
```

iii. CONVERSION_NOTES.md Step 4: *"Task explicitly requires Go and Catch only, excluding Aborted and Auto-rewarded. Use trial table fields to filter to valid Go/Catch trials and define trial outcomes from the same source."* The `.fillna(False)` guards and the alternate-column-name loop are the AI's defensive handling for schema variation across the release (Step 5 "Check for variables indicating valid data periods"). The ≥2-frame rule is an unstated guard against degenerate windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the per-cell dF/F trace, `exp.dff_traces['dff']` (one 1-D array per ROI, indexed by `cell_specimen_id`), stacked into an `(n_neurons, T_session)` float32 matrix. The AI initially used `exp.events` (detected calcium events, which is what the paper's analyses use) and switched to dF/F after measuring the sparsity of the event representation.

ii.
```python
def get_cell_matrix(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
        cell_ids = np.asarray(dff.index)
        traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
        mat = np.stack(traces, axis=0)
        return mat, cell_ids, 'dff'
    raise RuntimeError('Could not access dff traces')
```

iii. Trajectory step 29: *"`events` are extremely sparse (`frac_zero ≈ 0.9975`), while dF/F is much denser (`frac_zero ≈ 0.0096`). The all-zero neural trial warnings are therefore expected with events... we should switch the neural representation to dF/F for the decoder task, even though the paper used events for some analyses, because the provided decoder appears to work better with dense time-series and our current event representation yields many empty trials."* CONVERSION_NOTES.md Step 4 records the same discrepancy ("Paper states neural analyses used detected calcium events... Prefer event-based neural representation if accessible cleanly... otherwise verify what reference code uses and justify any fallback"), and Step 7 notes verification passed "after switching neural data to dF/F".

## 2-b. How is the `neural` data processed?

i. No processing at all beyond stacking: the SDK-provided dF/F traces (already motion-corrected, neuropil-corrected and baseline-normalised by the Allen pipeline) are stacked in ROI order and cast to float32. No z-scoring, no smoothing, no detrending, no baseline subtraction, no cross-plane merging (each file is its own session, see 1-c).

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. Implied by CONVERSION_NOTES.md Step 5 mapping: *"ophys events or dF/F traces per cell on ophys clock → neural: Slice into trial windows on common ophys timestamps; matrix shape (n_neurons, n_timepoints)."* The AI treats the SDK output as already fully processed and applies only the trial slicing the target format requires. float32 is used per the instructions' guidance on appropriate dtypes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control is applied. Every ROI in `dff_traces` is kept — 42,147 cells in total, which is exactly the number of rows in `ophys_cells_table.csv` for those 284 experiment ids, so nothing is dropped or added. There is no filtering on SNR, on `valid_roi`, on trace variance, or on all-zero traces.

ii. No code — the absence is the decision. The only cell-level bookkeeping is:
```python
cell_ids = np.asarray(dff.index)
...
'cell_ids': cell_ids,
```
(used only for its length, to size `brain_region_idx`).

iii. CONVERSION_NOTES.md Step 3 "Neuron curation rules": *"Paper excerpt indicates use of detected calcium events; exact cell-quality filters still need confirmation from code/SDK and possibly whitepaper."* The AI never returned to confirm this, but the implicit rationale (also visible in Step 1 notes) is that the AllenSDK release already exposes only the segmentation-QC-passed ROIs, so no extra filter is warranted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything is aligned to `exp.ophys_timestamps`, the common ophys clock, exactly as the Decoder Task requires ("Temporally align based on ophys timestamp"). The alignment event is the trial boundary, not the change: frames are selected by a boolean mask `(ts >= start_time) & (ts < stop_time)` and the identical `idx` is used to slice the neural matrix and to build every output row, guaranteeing sample-for-sample correspondence. Metadata records `temporal_alignment_event = 'ophys timestamps within experiment-defined trial window'` and `off_start = off_end = None` (since trials are variable-length and not locked to a single event).

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
...
'temporal_alignment_event': 'ophys timestamps within experiment-defined trial window',
'off_start': None,
'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2: *"Use `ophys_timestamps` as the master clock for all neural and behavioral/stimulus outputs."* Step 4 resolution: *"Align all time-varying streams to ophys timestamps, then derive per-trial outputs from aligned streams/image intervals."* Deriving every output from the same `trial_ts` vector is the AI's stated mechanism for guaranteeing no temporal misalignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, no resampling, no smoothing: data stay on the native ophys frame grid of each experiment. `time_bin_size` is reported as the median inter-frame interval of the **first file only**, ×1000 → 32.31 ms (≈31 Hz, the single-plane `VisualBehavior` rate). Because the 45 Multiscope experiments were also included (see 1-a/1-c), 45 of the 284 sessions are actually sampled at ≈11 Hz (≈93 ms bins) — visible in the verification log as sessions with mean T ≈ 89–93 frames for the same ~8.5 s trials, versus ≈246–274 frames elsewhere. The single declared `time_bin_size` therefore misdescribes 16% of the dataset, and the requirement "Time bins should be the same size for all trials and sessions" is not met.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2 only states that the ophys clock is the master time base; no rebinning decision is documented and the mixed frame rate is never mentioned. The implicit rationale is that the ophys timestamps already define a regular grid, so resampling would only lose information — which is correct *within* a project code but not *across* the two that were mixed in.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `exp.stimulus_presentations`, specifically the `image_name` and `start_time` columns (the `end_time` column exists but is deliberately not used — see 3-b). It is **not** derived from the trials table's `initial_image_name`/`change_image_name`. This means the label tracks the actual flash sequence, including the `omitted` pseudo-image that AllenSDK writes into `image_name` for omission flashes (4% of flashes), which becomes its own output class. A `blank` class covers any frame not overlapped by a stimulus interval (0.1% of frames). The resulting vocabulary is 18 values: 16 real images (image sets A and B), plus `blank` and `omitted`.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_start = np.asarray(stim['start_time'], dtype=float)
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```
```python
image_names = set(['blank', 'omitted'])
for p in probe_files:
    exp = load_experiment(p)
    stim = exp.stimulus_presentations.copy()
    if 'image_name' in stim.columns:
        image_names.update(stim['image_name'].dropna().astype(str).unique().tolist())
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"stimulus_presentations.image_name over non-grey image periods → output[0] image identity ... Label only during image presentation periods; grey periods may use a dedicated background/blank class if needed for full time series consistency."* The AI chose the stimulus table over the trials table because it is the direct record of what was on the monitor and because it lets a single trial contain more than the two images (initial/change) that the trials table exposes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Three steps. (1) Each stimulus row's effective interval is redefined as `[start_time, start_time + 0.75)` rather than the true ~250 ms flash duration, so the grey inter-stimulus interval inherits the label of the flash that preceded it. (2) For each trial, the stimulus rows overlapping the trial window are found, clipped to the trial boundaries, and the corresponding frames of a per-frame `image_series` are filled in flash order. Frames not covered by any interval keep the default `blank`. (3) Names are mapped to integer codes through a **global** dictionary built once at start-up — but that dictionary is built from only the first 8 NWB files, not from all 284; any image name absent from those 8 probe files would be silently coerced to `blank` by the `.get(name, blank)` default. (It happened to be safe here: the 8 probe files span both image set A and image set B, so all 16 images were captured.)

ii.
```python
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
stim['_end_time'] = stim_end
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
prev_name = None
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```
```python
# Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice.
probe_files = files[:min(8, len(files))]
...
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. The 750 ms expansion is the AI's most explicitly justified decision. Trajectory step 29: *"`stimulus_presentations` has `end_time` and image flashes last ~0.25 s, not the full 0.75 s interval; the 0.75 s image-presentation interval described in methods includes grey periods between flashes. Our current output labels most of each trial as `blank`, which explains why image identity decoding is below chance... A reasonable fix is to expand each active image presentation to the 750 ms interval beginning at `start_time`."* This is anchored on the quoted methods text recorded in Step 3: *"By image presentation interval we refer to the 750 ms interval beginning with each image presentation."* CONVERSION_NOTES.md Step 7 confirms the effect: "Image identity distribution became balanced across image classes with omitted fraction ~3.7% and blank ~0.1%." The probe-subset shortcut is justified only by the inline comment above (avoiding a second full pass over the NWB files).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on `trial_ts`, the same ophys timestamp vector used to slice `neural`, so the two are index-aligned by construction. Interval membership uses the half-open rule `(trial_ts >= s0) & (trial_ts < s1)`, matching the half-open rule used for the trial window itself, and the interval is clipped to `[start, stop)` before masking so no frame outside the trial can be touched. The image row is then stacked into the `(5, n_timepoints)` output matrix whose second axis is identical to `neural`'s.

ii.
```python
smask = (trial_ts >= s0) & (trial_ts < s1)
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
session_neural.append(trial_neural)
session_output.append(output)
```

iii. Same rationale as 2-d: every stream is resolved onto `trial_ts`, so alignment is guaranteed rather than asserted. CONVERSION_NOTES.md Step 4: *"Align all time-varying streams to ophys timestamps, then derive per-trial outputs from aligned streams/image intervals."*

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived **indirectly**, from consecutive `image_name` values in `stimulus_presentations` within the trial window: whenever the current flash's name differs from the previous flash's name, the first frame of the current flash is marked 1. It is not derived from `trials.change_time`/`trials.go`, and it does not use the `is_change` (or `is_sham_change`) boolean column that `stimulus_presentations` already provides. Because `omitted` is treated as an image name (see 3-a), every omission generates two spurious transitions (image→omitted and omitted→image). Quantitatively this is roughly half the positives: at ~4% omission rate and ~11 flashes per 8.5 s trial there are ~0.9 omission-driven flags per trial versus ~0.89 real change flags, and the observed positive rate of 0.7% × 264 frames ≈ 1.85 flags per trial matches that sum. Catch (sham-change) trials produce no flag, since the image genuinely does not change.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
prev_name = None
for _, srow in stim_trial.iterrows():
    ...
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"image identity transitions → output[1] image change: Binary time-varying series, 1 immediately after image identity changes, else 0. Must reflect change events within trial-aligned time series."* The AI is reading the instruction "Have value of 1 right after a change in image identity" literally and operationally — it detects a change in whatever label it is emitting as image identity. The interaction with omissions is nowhere discussed in the notes or the trajectory.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Only the transition detection described in 4-a. The marker is a **single-frame impulse**: `change_series[hit[0]] = 1`, where `hit[0]` is the first ophys frame inside the new flash's 750 ms interval. At 31 Hz that is a 32 ms pulse; at 11 Hz a 93 ms pulse. No smoothing, no widening to the flash or the flash+grey interval, and no restriction to `go` trials. The result is a very sparse target: 0.7% positives overall, i.e. class weights of roughly 1:142.

ii.
```python
        if prev_name is not None and name != prev_name:
            hit = np.flatnonzero(smask)
            if hit.size:
                change_series[hit[0]] = 1
```
```python
'output_names': ['image_identity', 'image_change', 'running_speed_bin', 'pupil_diameter_bin', 'trial_outcome'],
'output_values': [image_values, ['no_change', 'change'], ...],
```

iii. Same as 4-a — the AI read "right after a change" as "at the first sample after the change" and implemented the minimal such encoding. No further justification appears in CONVERSION_NOTES.md; the notes only record the resulting distribution ("image_change is highly sparse (0.6%)" in the sample, trajectory step 26) as an observation, without acting on it.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required or applied: the variable is constructed as a binary indicator directly, with value names `['no_change', 'change']` and observed range `[0, 1]` in the verification log.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
change_series[hit[0]] = 1
...
'output_values': [image_values, ['no_change', 'change'], [f'bin_{i}' for i in range(5)], [f'bin_{i}' for i in range(5)], TRIAL_OUTCOME_VALUES],
```

iii. The Decoder Output spec calls image change a "binary variable", so it is already categorical and no discretisation step is needed. This is consistent with CONVERSION_NOTES.md Step 5 Key Decision 6, which restricts percentile discretisation to running speed and pupil diameter only.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the impulse is placed at an index into `trial_ts`, the same frame grid as `neural`, and the row is stacked into the same `(5, n_timepoints)` matrix. The index `hit[0] = np.flatnonzero(smask)[0]` is the first ophys frame at or after the new flash's onset (clipped to the trial), so the marker lands at most one frame (32 ms / 93 ms) after the true stimulus transition.

ii.
```python
smask = (trial_ts >= s0) & (trial_ts < s1)
...
hit = np.flatnonzero(smask)
if hit.size:
    change_series[hit[0]] = 1
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
```

iii. Same rationale as 2-d and 3-c: all outputs are built on the shared `trial_ts` index, so alignment with `neural` is structural. CONVERSION_NOTES.md Step 5 Key Decision 2.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `exp.running_speed`, the SDK's running DataFrame. `get_running_series()` pulls the `timestamps` column and the `speed` column (with positional fallbacks to the first and second columns if those names are absent) as float arrays. This is the encoder-derived linear speed in cm/s on the behaviour clock (~60 Hz), which is faster than the ophys clock and therefore has to be resampled.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
...
run_t, run_v = get_running_series(exp.running_speed.copy())
```

iii. CONVERSION_NOTES.md Step 1 identifies `BehaviorOphysExperiment.running_speed` as the loader for "running speed time series aligned to timestamps", and Step 5 maps it directly to output[2]. It is the only running signal the SDK exposes, so no selection was needed; the column-name fallbacks are defensive handling for schema variation.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps. (1) Resampling: linear interpolation of the 60 Hz speed trace onto the trial's ophys timestamps with `np.interp`, done per trial (so the same session-level stream is re-interpolated once per trial). `np.interp` clamps rather than extrapolating, so frames outside the running-trace span silently take the first/last speed value. (2) Discretisation into 5 bins with `np.digitize` against pre-computed edges (see 5-c). No smoothing, no |speed| rectification, no thresholding of negative values (the trace contains negatives; the lowest edge is −12.66 cm/s).

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"running_speed → output[2] running speed bin: Interpolate/align to ophys timestamps, discretize into 5 equal-percentile bins over valid samples. Time-varying categorical output."* Linear interpolation is the standard way to move a faster behavioural stream onto the ophys grid without inventing structure, and the AI's Step 4 resolution requires all streams on the ophys clock.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins defined by quintile edges `np.quantile(run_all, [0, .2, .4, .6, .8, 1])`, applied globally to every session. Critically, `run_all` is **not** the full dataset: it is the concatenation of the raw (un-resampled, un-trial-segmented, ~60 Hz) running samples from only the **first 8 NWB files** — 8 experiments from 2 mice. The resulting edges, printed in `conversion_full_out.txt`, are `[-12.66, -0.0435, 0.0195, 0.1086, 5.707, 53.64]`: three of the five edges sit within ±0.11 cm/s of zero. Applied to all 284 sessions the bins are far from equal-occupancy — the verification log reports `bin_0 0.150, bin_1 0.096, bin_2 0.104, bin_3 0.166, bin_4 0.484`, i.e. the top bin holds 48% of the data instead of 20%.

ii.
```python
probe_files = files[:min(8, len(files))]
for p in probe_files:
    exp = load_experiment(p)
    rt, rv = get_running_series(exp.running_speed.copy())
    run_vals.append(rv[np.isfinite(rv)])
run_all = np.concatenate(run_vals) if run_vals else np.array([0.0])
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
```
```python
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The documented decision (CONVERSION_NOTES.md Step 5 Key Decision 6) is *"Compute five equal-percentile bins for running speed and pupil diameter using valid pooled samples, then apply bin edges within all sessions."* The 8-file restriction is justified only by the inline code comment *"Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice"* and by trajectory step 33/96, where the AI interrupted the first full run because `collect_global_info` was loading all 284 files: *"patch `convert_data.py` to avoid preloading all sessions in `collect_global_info`... compute running/pupil bin edges from a small sample of sessions in `--full` mode to reduce startup cost."* The resulting non-uniformity was never checked against the verification output.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Alignment is by interpolation onto `trial_ts` — the same ophys timestamps used to slice `neural` — so the running row has exactly `n_timepoints` entries index-matched to the neural columns. Because the running stream is sampled faster than the ophys frames, interpolation is a downsampling and introduces no lag.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series])
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2 and Step 4 resolution: all streams are placed on the ophys clock before any output is formed, so alignment does not need a separate mechanism.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `exp.eye_tracking`. `pick_pupil_series()` searches a priority list of column names — `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, `pupil_size` — and takes the first that exists. In the actual Allen data `pupil_diameter` does not exist, so **`pupil_area`** is selected (not `pupil_width`). The whole `eye_tracking` access is wrapped in `try/except` so a session without eye tracking degrades gracefully. The `likely_blink` column is never referenced; blink frames are nevertheless excluded, because AllenSDK already writes `NaN` into `pupil_area` exactly on the `likely_blink` rows (verified: NaN fraction 0.0919 == `likely_blink` fraction 0.0919), and the AI drops non-finite samples before interpolating.

ii.
```python
def pick_pupil_series(eye_tracking):
    if eye_tracking is None or len(eye_tracking) == 0:
        return None, None, None
    cols = list(eye_tracking.columns)
    time_col = 'timestamps' if 'timestamps' in cols else None
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    if value_col is None:
        value_col = next((c for c in cols if 'pupil' in c.lower() and np.issubdtype(eye_tracking[c].dtype, np.number)), None)
    if time_col is None or value_col is None:
        return None, None, None
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```
```python
try:
    eye = exp.eye_tracking.copy()
except Exception:
    eye = None
pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"pupil diameter / eye tracking pupil area-equivalent → output[3] pupil diameter bin: Align to ophys timestamps, derive diameter-like measure if needed... Handle missing data carefully; likely use valid samples only for percentile computation."* The AI deliberately wrote a name-agnostic picker rather than hard-coding a column, on the grounds that it did not know which pupil measure the release exposes. Because the output is percentile-binned, a monotone re-parameterisation of pupil size (area vs diameter) was treated as interchangeable.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Non-finite samples are dropped — which removes the blink frames, since AllenSDK NaNs them. (2) If at least 2 valid samples remain, the trace is linearly interpolated onto `trial_ts` with `np.interp` (interpolating *across* the removed blink gaps, and clamping outside the eye-tracking span). (3) The interpolated values are `np.digitize`d into 5 bins. If the session has no usable pupil signal, the whole row is filled with zeros — i.e. missing pupil is silently reported as bin 0 rather than as a distinct "missing" class. No smoothing, no per-session normalisation, no conversion of area to diameter.

ii.
```python
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    if valid.sum() >= 2:
        pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
        pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
    else:
        pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. Same as 5-b/6-a: CONVERSION_NOTES.md Step 5 mapping requires alignment to the ophys clock and "valid samples only", and Step 5 Key Decision 6 requires five percentile bins. The zeros fallback is the AI's "sensible default" for the instruction "Handle missing data appropriately"; it is not separately justified in the notes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed, and the same defect, amplified. Edges are `np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])` over the pooled raw `pupil_area` samples of only the first 8 NWB files. Those 8 files come from 2 mice on one rig, and their pupil areas are large: the printed edges are `[234.0, 9526.9, 13362.6, 16680.9, 22682.0, 133731.0]`. Most of the remaining 276 sessions have pupil areas below the first interior edge, so they collapse into bin 0. The verification log shows `bin_0 0.835, bin_1 0.112, bin_2 0.027, bin_3 0.017, bin_4 0.008` — nothing like five equal percentile bins — and the per-session breakdown shows dozens of sessions with a bin_0 fraction of exactly 1.000 (the entire session lands in one class, so pupil is unpredictable in principle for those sessions).

ii.
```python
    try:
        eye = exp.eye_tracking.copy()
        pt, pv, _ = pick_pupil_series(eye)
        if pv is not None:
            pupil_vals.append(pv[np.isfinite(pv)])
    except Exception:
        pass
pupil_all = np.concatenate(pupil_vals) if pupil_vals else np.array([0.0])
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
```
```python
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. Identical to 5-c: the stated decision is percentile bins over "valid pooled samples" (Step 5 Key Decision 6), but the implementation substitutes an 8-file probe for the pool to save a second loading pass (inline comment; trajectory step 96). The AI's own verification output printed the 0.835/0.112/0.027/0.017/0.008 distribution, but Step 10 ("Critical Review 1", where distributions were to be checked against expectations) was left NOT STARTED, so it was never caught.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: interpolation onto `trial_ts`. The eye-tracking camera runs at ~60 Hz on a hardware-synced clock, and the SDK's `timestamps` column is already in the same session time base as `ophys_timestamps`, so `np.interp(trial_ts, pupil_t, pupil_v)` yields one value per ophys frame, index-matched to the neural columns. Blink gaps are bridged by interpolation rather than left as holes, so the row is always exactly `n_timepoints` long.

ii.
```python
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2 / Step 4 resolution — all time-varying streams are resolved onto `ophys_timestamps` before outputs are assembled.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — checked in that fixed order, with an `'other'` fallback if none is true. Since aborted and auto-rewarded rows were already removed, every remaining go/catch trial matches exactly one of the four; the verification log confirms the observed range is `[0, 3]`, i.e. `'other'` is never used in practice.

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']

def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"trials outcome fields (hit/miss/false alarm/correct reject etc.) → output[4] trial outcome: Static per-trial categorical label repeated or stored as per-trial vector. Include only Go/Catch, exclude Aborted and Auto-rewarded."* Step 4 resolution likewise says to "define trial outcomes from the same source" as the trial-type filter. These are the SDK's canonical change-detection outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The label string is mapped to an integer through a fixed global dictionary `{hit:0, miss:1, false_alarm:2, correct_reject:3, other:4}` and then **broadcast across all timepoints of the trial** with `np.full`, so that it forms a constant row of the `(5, n_timepoints)` output matrix rather than a per-trial scalar. The `output_values` entry advertises 5 names although only 4 ever occur. The resulting distribution is hit 0.186 / miss 0.688 / false_alarm 0.011 / correct_reject 0.115 — strongly miss-dominated because the 82 passive sessions (where the lick spout is retracted) were retained and contribute only misses and correct rejects.

ii.
```python
outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: *"Make image identity, image change, running bin, and pupil bin time-varying; make trial outcome static per trial."* Trajectory step 21 explains why it is nevertheless emitted as a repeated row: *"make `output` for each trial a single `(5, T)` int array by repeating trial outcome across time, so all outputs are time-varying and categorical"* — a uniform `(n_output, n_timepoints)` matrix is what the target format and `train_decoder.py` expect, and the instruction says "If at all possible, make it time-varying".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several classes of imperfection, all by silent defaulting rather than by flagging:
- **Missing/renamed columns**: `stop_time` falls back to `end_time`; running `timestamps`/`speed` fall back to positional columns 0/1; auto-reward flags are searched under three alternative names; the pupil column is searched through a six-name priority list plus a regex-ish fallback over any numeric column containing "pupil".
- **Missing flags**: `.fillna(False)` before negating `aborted`/`auto_rewarded`, so NaN flags do not propagate.
- **Missing eye tracking**: `try/except` around `exp.eye_tracking`; if it raises, or the table is empty, or fewer than 2 finite samples remain, the pupil row becomes all zeros (bin 0) — indistinguishable downstream from a genuinely small pupil.
- **Blinks / NaN pupil samples**: dropped via `np.isfinite` before interpolation, so blink artefacts do not enter the trace (and `np.digitize` never sees a NaN, which would otherwise be pushed into the top bin).
- **Degenerate trial windows**: trials yielding fewer than 2 ophys frames are skipped; the half-open mask means trials extending past the end of the recording are simply truncated.
- **Out-of-range interpolation**: `np.interp` clamps to the first/last sample rather than producing NaN, so frames before/after a behavioural stream's coverage take edge values.
- **Unknown image names**: `image_to_idx.get(name, image_to_idx['blank'])` maps anything outside the probe-derived vocabulary to `blank`.
- **Unknown outcomes**: `infer_trial_outcome` returns `'other'`.

What is *not* handled: there is no `try/except` around a whole session's load or processing, so one corrupt NWB file would abort the entire 30-minute run; NaNs in `running_speed` are not filtered before `np.interp`; and no session is dropped for having too few trials (not triggered here — the smallest session has 39 trials).

ii.
```python
stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
...
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
...
try:
    eye = exp.eye_tracking.copy()
except Exception:
    eye = None
...
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
if valid.sum() >= 2:
    pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
    ...
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
if idx.size < 2:
    continue
...
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. Driven by the instruction "Handle missing data appropriately (consult references, use sensible defaults, document)" and CONVERSION_NOTES.md Step 5 ("Handle missing data carefully; likely use valid samples only for percentile computation"). Trajectory steps 24 and 22 show two of these guards being added reactively after crashes (`stimulus_presentations` has no `stop_time`; `from_nwb` needs an `NWBFile`, not a path), which is why the style is uniformly "probe for the column, else fall back".

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is NWB I/O plus AllenSDK object construction: `NWBHDF5IO.read()` + `BehaviorOphysExperiment.from_nwb()` + materialising `dff_traces`, `stimulus_presentations`, `running_speed` and `eye_tracking`. The full run took **1887 s (~31.5 min)** for 284 sessions, i.e. ~6.6 s/session, with the per-session log line showing 3.5 s for small sessions and >20 s for 400–500-neuron sessions — scaling with cell count confirms the trace read/stack is the bottleneck. Secondary costs: the nested per-trial × per-stimulus-flash `iterrows()` loop (~85,230 trials × ~11 flashes ≈ 1M pandas row objects); the per-trial `np.flatnonzero((ts >= start) & (ts < stop))` which rescans the entire session timestamp vector once per trial; and finally pickling ~14 GB of float32 dF/F to disk. The script does print per-session timings (`time={...:.2f}s`), satisfying the "print timing information to find bottlenecks" instruction.

ii.
```python
for i, p in enumerate(files, 1):
    s0 = time.time()
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
    print(f'processed {i}/{len(files)} {p.name} trials={len(sess["neural"])} neurons={len(sess["cell_ids"])} source={sess["neural_source"]} time={time.time()-s0:.2f}s')
```

iii. Trajectory step 36: *"The script is processing sessions at about 3.5–4.4 seconds per session after startup... At that rate, 284 sessions will take roughly 17–21 minutes plus overhead, which is somewhat above the 15-minute target but not wildly so. Since the conversion is already running and making steady progress without errors, it is more efficient to let it continue than to interrupt again."* By step 44 the AI revised the estimate to ~30 min and still chose not to optimise: *"stopping now would waste substantial completed work."* The Step 6 "Code inefficiencies identified" and "Code speedups added" sections of CONVERSION_NOTES.md were left as unfilled `[Note]` placeholders.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
1. **The inner stimulus loop** `for _, srow in stim_trial.iterrows()` — the worst offender. It builds a pandas Series per flash and a full-length boolean mask per flash. Both the image labelling and the change detection could be done with a single `np.searchsorted(stim_start, trial_ts, side='right') - 1` to map every frame to its flash index, then one fancy-index assignment for identity and one `np.diff`/`np.flatnonzero` for transitions — O(T log F) instead of O(F·T) with Python-level overhead.
2. **The trial-window search** `np.flatnonzero((ts >= start) & (ts < stop))` — an O(T_session) scan per trial. `np.searchsorted(ts, [start, stop])` + `np.arange` is O(log T) and is exactly what the reference uses.
3. **The per-trial stimulus sub-selection** `stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]` — a full boolean scan and DataFrame copy of the ~4,800-row stimulus table for each of the session's ~300 trials; a `searchsorted` on the sorted `start_time` array would give the slice bounds directly.
The outer `for _, tr in trials.iterrows()` loop is harder to remove (trials are ragged), but it too should iterate over numpy arrays rather than pandas rows.

ii.
```python
    stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
    prev_name = None
    for _, srow in stim_trial.iterrows():
        s0 = max(start, float(srow['start_time']))
        s1 = min(stop, float(srow['_end_time']))
        smask = (trial_ts >= s0) & (trial_ts < s1)
```
```python
    idx = np.flatnonzero((ts >= start) & (ts < stop))
```

iii. No justification is recorded — CONVERSION_NOTES.md Step 6 leaves both efficiency sections as `[Note]` placeholders. The only optimisation the AI reasoned about explicitly was the double-loading of NWB files (trajectory step 96), not the inner loops; and having decided at step 36/44 to let the 30-minute run finish, it never revisited them.

## 9-c. What processing does the code repeat multiple times?

i. **Redundant NWB loads.** `collect_global_info()` opens and fully materialises the first 8 NWB files; `main()` then opens all 284 — so those 8 are loaded twice. Worse, after the main loop `main()` calls `load_experiment(files[0])` a **third** time, re-reading an entire NWB file just to take the median of `np.diff(ophys_timestamps)`, a quantity already available from the session it had just processed. The inline comment claiming the probe-subset "avoids loading all NWB files twice" is therefore only partly true, and the code still contradicts it for file 0.

Other repetition: `exp.running_speed` / `exp.stimulus_presentations` / `exp.eye_tracking` are each `.copy()`-ed (full DataFrame copies) and `stim` is copied twice more inside `build_session`; the session-level running and pupil traces are re-interpolated from scratch for every trial instead of once per session onto the full `ophys_timestamps` (as the reference does); and `stim` is re-sorted per session even though `stimulus_presentations` is already time-ordered.

ii.
```python
# Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice.
probe_files = files[:min(8, len(files))]
for p in probe_files:
    exp = load_experiment(p)
    ...
```
```python
    for i, p in enumerate(files, 1):
        exp = load_experiment(p)
        ...
    dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```
```python
    stim = exp.stimulus_presentations.copy().sort_values('start_time')
    ...
    stim = stim.copy()
    stim['_end_time'] = stim_end
    run_t, run_v = get_running_series(exp.running_speed.copy())
```

iii. The probe subset is justified by the inline comment and trajectory step 96 (*"avoid preloading all sessions in `collect_global_info`... to reduce startup cost"*). The third load of `files[0]` and the repeated per-trial interpolation are not mentioned anywhere; CONVERSION_NOTES.md Step 6 "Code inefficiencies identified" is an empty `[Note]`.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed, stored or advertised and then never used:
- **`--show-processing` is a dead flag.** It is declared in `parse_args()` and never read; there is no plotting code and no `matplotlib` import, so no `processing_<session_id>.png` was ever produced (confirmed: only `predictions.png` and `sample_trials.png` exist, both written by `train_decoder.py`). Step 7's requirement to visually verify alignment and discretisation was therefore silently skipped.
- **Unused per-session bookkeeping.** `image_names_seen` (appended once per flash per trial — ~1M list appends over the run) and `trial_outcomes_seen` are accumulated, `sorted(set(...))`-ed, and returned in the session dict, but never read when assembling `data`. `neural_source` is returned and printed but only ever equals `'dff'`. `cell_ids` is built as a full array when only `len()` is used.
- **Wasteful dtypes and empty arrays.** The output matrix is cast to `int64` although every value fits in `int8` — ~900 MB of the pickle is padding. A fresh `np.zeros((0, len(trial_ts)), dtype=np.float32)` is allocated for each of the 85,230 trials to represent "no input".
- **Stale/incorrect metadata.** `'neural_source_preference': 'events_then_dff_fallback'` is written into the output metadata, but the events path was removed from the code — the pickle now documents behaviour the script does not have.
- **Never-closed file handles.** `exp._nwb_io = io` keeps 284 HDF5 handles alive with no purpose; nothing ever closes them.

ii.
```python
ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
```
```python
        image_names_seen.append(name)
    ...
    return {
        ...
        'cell_ids': cell_ids,
        'neural_source': neural_source,
        'image_names_seen': sorted(set(image_names_seen)),
        'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
```
```python
        session_input.append(np.zeros((0, len(trial_ts)), dtype=np.float32))
        output = np.vstack([...]).astype(np.int64)
```
```python
            'neural_source_preference': 'events_then_dff_fallback',
```

iii. None of this is justified in CONVERSION_NOTES.md. The dead `--show-processing` flag and the stale `neural_source_preference` key are residue from the earlier events-based draft (trajectory steps 21–29) that was never cleaned up; Step 13 ("Documentation and Cleanup") was left NOT STARTED, as were Steps 10 and 12, so no review pass ever removed them.
