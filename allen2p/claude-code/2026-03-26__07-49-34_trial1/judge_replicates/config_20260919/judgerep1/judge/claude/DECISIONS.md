# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache`. It reads the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) with pandas to enumerate experiments, intersects that table with the set of NWB files actually present on disk, and then reads every NWB file directly with `h5py`. Two filters are applied to the experiment table:

- keep only experiments whose NWB file was downloaded (284 files present);
- drop every session whose `session_type` contains "passive" (OPHYS_2 and OPHYS_5), leaving 202 experiments / 174 unique `ophys_session_id` / 38 mice.

No filter on `project_code` is applied, so the 34 active `VisualBehaviorMultiscope` experiments (6 real sessions, all from mouse 457841, imaged at ~11 Hz) are loaded alongside the 168 active `VisualBehavior` single-plane experiments (~31 Hz). Per file the AI pulls dF/F traces + timestamps, running speed + timestamps, pupil tracking + blink flags, the `trials` interval table, and the `Natural_Images_*_presentations` stimulus table.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)

    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()

    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()

    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table
```

```python
def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]
        ...
        trials = f['intervals']['trials']
        ...
```

iii. From CONVERSION_NOTES.md Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". Step 10 Check 3 argues the direct read is "Equivalent" to `BehaviorOphysExperiment.from_nwb()`, and that skipping the SDK's `exclude_invalid_rois=True` is safe because "all ROIs in downloaded NWB files are valid". Step 2/Step 5 Key Decision 10 justify dropping passive sessions: "Active: OPHYS_1, 3, 4, 6 … Passive: OPHYS_2, 5 … We should only use active sessions for the decoder task." The AI explicitly recorded in Step 4 that the download contains two project codes ("VisualBehavior (239), VisualBehaviorMultiscope (45)") and decided to keep both: "We have a mix of VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) data … Need to handle both ~31 Hz and ~11 Hz ophys frame rates."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values taken from the experiment-table row of each processed experiment. Mice are registered lazily, in order of first appearance (experiments are iterated sorted by `ophys_experiment_id`), into `subjects` / `subject_idx`. Result: 38 subjects.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
```
and in the final dict:
```python
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "`mouse_id` → `subjects`, `subject_idx` — Map unique mice to indices". The AI cross-checked the count against the data ("Subjects (downloaded) | 38") and against the papers ("Total mice | 82 | Piet et al."), concluding in Step 4 that "Only subset downloaded; proceed with 38".

## 1-c. How are the data split into sessions?

i. A "session" in the output is **one ophys experiment (one imaging plane)**, not one `ophys_session_id`. The experiment table is iterated row-by-row and each row produces one entry in `neural`/`output`/`subject_idx`/`brain_region_idx`. `ophys_session_id` is recorded in `metadata['session_info']` but is never used to group planes. For the 168 single-plane `VisualBehavior` experiments this is a 1:1 mapping, but the 6 multiscope sessions are emitted as 34 separate "sessions" that repeat the same trials, running speed, pupil and outcome data with different neurons. The output therefore has 202 sessions from 174 real recordings, and mouse 457841 alone contributes 34 of the 202 sessions.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    if result is None:
        continue
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```
```python
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    'session_type': result['session_type'],
    ...
})
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 9: "**Multiscope handling**: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." Step 4 notes the same fact: "For Multiscope sessions, multiple experiments (planes) per session share the same trials/behavior."

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is valid if it is a Go **or** Catch trial and is neither Aborted nor Auto-rewarded. The trial window is the full `start_time` → `stop_time` interval (variable length, ~8 s, ~260 ophys frames at 31 Hz / ~90 at 11 Hz); ophys frames are selected with a half-open mask `ophys_ts >= start_time` and `ophys_ts < stop_time`.

ii.
```python
def get_valid_trials(trial_data):
    """Get indices of valid trials (Go + Catch, excluding Aborted and Auto-rewarded)."""
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)

    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]
```
```python
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]

    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: "**Trial definition**: Use `start_time` and `stop_time` from trials table for Go and Catch trials only", and Step 3 "Trial curation rules (for this decoder task): Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials" — a direct transcription of the task instruction. The AI sanity-checked a low-trial-count session (39 trials) in the trajectory and confirmed the cause was 1078 aborted trials in a poorly performing mouse, i.e. legitimate.

## 1-e. How are trials filtered based on quality controls?

i. Four filters:
- trial-level: `(go | catch) & ~aborted & ~auto_rewarded` (see 1-d);
- trials whose window contains fewer than 2 ophys frames are dropped;
- experiments with fewer than 2 valid trials before processing, or fewer than 2 surviving trials after processing, are dropped entirely (returns `None`);
- experiments with 0 cells, a missing NWB file, or no stimulus-presentation table are dropped entirely.

No filtering on `change_time` validity, on engagement/reward rate, or on behavioral performance is applied.

ii.
```python
valid_trial_idx = get_valid_trials(trial_data)

if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
    return None
```
```python
if n_trial_frames < 2:
    continue
```
```python
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. The ≥2-trial rule follows the format spec quoted in the instructions ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). CONVERSION_NOTES.md Step 10 Check 5: "Sessions with very few valid trials (e.g., 39) are retained if >=2 trials". The AI also documented (Step 3) the session-level QC that Allen already applied upstream ("<1000 saturated pixels, <20% photobleaching, … peak d-prime >= 1.0"), and so applied no further session QC of its own.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. dF/F calcium traces, read from the NWB path `processing/ophys/dff/traces/data` (stored as `(n_frames, n_cells)`), with their timebase from `processing/ophys/dff/traces/timestamps`. Deconvolved events (`processing/ophys/event_detection`) were examined and deliberately not used.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]

dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 1: "**Neural data**: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 1 records that both are precomputed in the NWB ("dF/F is pre-computed in NWB files (no need to compute from scratch)"; "Events (deconvolved spikes) also pre-computed via FastLZeroSpikeInference").

## 2-b. How is the `neural` data processed?

i. Essentially none. The `(n_frames, n_cells)` array is transposed to `(n_cells, n_frames)`, sliced by the trial frame mask, and cast to `float32`. No smoothing, z-scoring, baseline subtraction, normalization or plane merging is performed (planes are never merged — see 1-c).

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 3 records that the Allen pipeline already produced dF/F "with 600s median filter baseline, detrending with 3.33s median filter", and Step 10 Check 3 lists "dF/F computation | Pre-computed in NWB | Pre-computed (600s median filter + detrending) | Yes". So the AI's position is that the SDK's processing is the reference processing and nothing further should be applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is done in the script. In particular the SDK's default `exclude_invalid_rois=True` is bypassed because `h5py` is used directly; the AI checked that this is harmless because the released NWB files contain only valid ROIs. Total neurons kept: 29,444 across 202 experiments (mean 146, range 4–666).

ii. No filtering code exists. The only related guard is:
```python
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. CONVERSION_NOTES.md Step 10 Check 3: "ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid", and Check 4: "Neurons | 29,444 | Matches cells_table.csv | PASS". (I verified both claims independently: `valid_roi` is all-True in the sampled NWB files, and the 29,444 count exactly equals the number of rows in `ophys_cells_table.csv` for the 202 active experiments.) In the trajectory (step 60) the AI considered "add the valid_roi filter for safety" and then dropped the idea after confirming it was a no-op.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the **ophys timebase**, with each trial spanning trial `start_time` to `stop_time`. A boolean mask over the full-session `ophys_timestamps` selects frames with `start_time <= t < stop_time`; that same mask indexes dF/F, running speed and pupil, so all streams share the same frame grid by construction. Trials are variable length. `metadata['off_start']` and `metadata['off_end']` are set to `None`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
n_trial_frames = frame_mask.sum()
...
neural = dff[:, frame_mask].astype(np.float32)
...
running_trial = running_at_ophys[frame_mask]
pupil_trial = pupil_at_ophys[frame_mask]
```
```python
'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
'off_start': None,
'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "**Temporal alignment**: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time." This follows the instruction "Temporally align based on ophys timestamp." Step 3 notes all streams are hardware-synced ("Synchronization: All streams synced via NI PCI-6612 at 100 kHz"), which is what licenses resampling the behavioral streams onto the ophys clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of the neural data is performed — trials are kept at each experiment's native ophys frame rate. Because both rigs are included, the dataset contains **two different bin sizes**: ~32.3 ms (~31 Hz, Scientifica, 168 experiments) and ~90 ms (~11 Hz, Multiscope, 34 experiments). `metadata['time_bin_size']` is filled with the **median across experiments** (32.32 ms), which is wrong for the 34 multiscope sessions; the true per-experiment `dt_ms` is preserved in `metadata['session_info']`. The consequence is visible in the verification log: trial length T ranges from 77 to 389 frames (mean 233) whereas the reference, restricted to one rig, has T in 217–391.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
'dt': float(dt),
```
```python
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2: "**Time bin**: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." Step 10 Check 5 repeats this as a known edge case: "Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial". The trajectory (step 52) shows the AI noticed the effect ("the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica)") but chose not to act on it.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentation table** `intervals/Natural_Images_*_presentations` — specifically `image_name`, `start_time` and `stop_time` — not from the trials table. Every ophys frame is labelled with the image that was physically on screen at that moment; frames in the 500 ms inter-stimulus grey period, and frames during `omitted` flashes, get a dedicated `'gray'` category. This makes image identity a 17-class variable (grey + 16 images, image sets A and B pooled) that toggles on and off several times per trial; grey accounts for 66.9 % of all frames.

ii.
```python
# Initialize to gray (ISI)
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)

stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']

for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    if s_stop < trial_start:
        continue
    if s_start >= trial_stop:
        break
    name = stim_names[si]
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    # Skip omitted stimuli (they're gray screen)
    if name == 'omitted':
        continue
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "**Image identity**: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." The AI treats the 66.9 % grey fraction as its main correctness check (Step 9 consistency table: "Stimulus timing | 250ms on, 500ms off | 66.9% gray | Yes (expected 66.7%)"), which it also states in the trajectory: "The gray screen fraction (66.9%) matches the expected 66.7% (500ms gray / 750ms total cycle)."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes through a single global vocabulary built before conversion: every NWB file in the (already filtered) experiment table is re-opened once just to read its `image_name` column, `'omitted'` is dropped, the union is sorted, and `'gray'` is prepended so that grey = code 0 and the 16 natural images occupy codes 1–16. The same vocabulary is written to `output_values[0]`, so codes are comparable across sessions even though each session only shows 8 of the 16 images.

ii.
```python
def get_all_image_names(exp_table):
    all_names = set()
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        try:
            with h5py.File(nwb_path, 'r') as f:
                for key in f['intervals'].keys():
                    if 'Natural_Images' in key or 'natural_images' in key:
                        names = f['intervals'][key]['image_name'][:]
                        for n in names:
                            if isinstance(n, bytes):
                                n = n.decode('utf-8')
                            if n != 'omitted':
                                all_names.add(n)
                        break
        except Exception as e:
            print(f"  WARNING: Could not read images from {eid}: {e}")
    return sorted(all_names)
```
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
...
output_values = [
    image_names_list,          # image identity categories
    ...
]
```

iii. The AI's Step 9 consistency table justifies the global vocabulary: "Images/session | 8 | 8 per session (16 total across sets) | Yes" — image set A and image set B sessions must not collide in code space. `'gray'` is prepended so that its index is a fixed 0. The scan cost was measured and accepted ("Image name collection adds ~14s overhead", Step 7).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The trace is built on exactly the same frame grid as the neural data: `build_image_identity_trace` recomputes the identical half-open mask `(ophys_ts >= trial_start) & (ophys_ts < trial_stop)` and assigns each frame the image whose `[start_time, stop_time)` interval contains that frame's ophys timestamp. Frames are therefore aligned to the physical flash boundaries at ophys resolution, with no interpolation or shifting.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. CONVERSION_NOTES.md Step 10 Check 2 lists a spot-check against raw NWB: "Image identity trace | 803736273 | PASS - np.array_equal=True, verified at stimulus onset", and Step 5's planned checks include "Verify image identity changes at correct times (compare is_change with actual image transitions)" and "Verify no temporal misalignment by plotting neural + stimulus for sample trials" (the `--show-processing` plots).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the stimulus presentation table's `is_change` flag together with the presentation's `start_time`. Because `is_change` marks only flashes whose image actually differs from the previous flash, catch trials (sham change, image unchanged) automatically get no change marker, matching the intent of restricting the marker to real changes.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    s_start = stim_starts[si]
    if s_start < trial_start or s_start >= trial_stop:
        continue
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "`is_change` from stimulus presentations → `output[1]`: image_change — Binary, 1 at change onset frame, 0 otherwise — Time-varying". The AI preferred the stimulus table over the trials table because it was already using the stimulus table for image identity, giving one consistent source for stimulus-locked events; Step 5's planned check "Check that image_change=1 aligns with actual change in image_identity" is the internal-consistency argument for this choice.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Minimal: allocate a zero vector of trial length, and for each change flash whose onset falls inside the trial window find the first ophys frame at or after the onset with `np.searchsorted` and set that single element to 1. No smoothing, no window extension, no per-trial normalization.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)
    if n_frames == 0:
        return np.array([], dtype=np.int64)

    trace = np.zeros(n_frames, dtype=np.int64)
    ...
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
    return trace
```

iii. The AI read the instruction ("Have value of 1 right after a change in image identity, otherwise 0") literally: the first ophys frame at or after change onset is the frame "right after" the change. Its docstring states this: "Build a binary trace that is 1 at the first frame after an image change."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 2 categories, `output_values[1] = ['no_change', 'change']`, with exactly **one** positive frame (~32 ms) per go trial and zero positive frames on catch trials. Resulting class balance across the whole dataset is 99.6 % `no_change` / 0.4 % `change`.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
```
```python
output_full[1] = change_trace
```

iii. The AI recognised and explicitly defended the imbalance rather than widening the window. CONVERSION_NOTES.md Step 12: "**image_change** (1.27x): Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily. The decoder correctly identifies change timepoints but the signal is very sparse. Not a conversion bug." Step 10 Check 4 treats the sparsity as a passing sanity check: "Image change fraction | 0.372% | ~0.3-0.4% | PASS".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same frame grid as the neural data — `build_image_change_trace` rebuilds the identical `(ophys_ts >= trial_start) & (ophys_ts < trial_stop)` mask and `searchsorted`s the change onset into that trial's ophys timestamps, so the positive frame is the first imaging frame at or after the physical change onset.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```
```python
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
...
output_full[1] = change_trace
```

iii. Same justification as 3-c: all output streams are built on the ophys frame grid so that no cross-stream interpolation or offset is ever needed. The `--show-processing` figure plots image identity and image change on a common time axis for the first three trials of a session specifically to make the alignment visually checkable ("Change events aligned to stimulus onset", Step 7).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The NWB `processing/running/speed` time series (`data` + `timestamps`), which is the Allen-pipeline running speed (60 Hz, 10 Hz low-pass Butterworth-filtered upstream) — the same object the SDK exposes as `dataset.running_speed`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. CONVERSION_NOTES.md Step 2 identifies the source ("Running: `processing/running/speed` (270K samples @ 60 Hz)") and Step 3 records that the filtering is already done upstream ("Running speed: 10 Hz lowpass Butterworth filter"), so the AI takes the stream as-is.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The whole-session running trace is linearly interpolated onto the whole-session ophys timestamps once per experiment (`bounds_error=False, fill_value=np.nan`), then each trial's values are read out by the trial frame mask and discretized (see 5-c). No filtering, clipping or sign handling is applied.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)
```
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: "**Running speed**: Interpolate from 60 Hz to ophys timestamps using linear interpolation." The AI planned and reported a check on the resulting values ("Verify running speed range is reasonable (typically 0-80 cm/s)"; Step 10 Check 2: "Running speed bins | 803736273 | PASS - np.array_equal=True").

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins, with the bin edges computed **per experiment** from that experiment's entire ophys-resampled running trace (including timepoints outside any trial), then applied to the trial slices. Outer edges are replaced by ±inf so nothing falls outside, `np.digitize` + `np.clip` produce codes 0–4, and NaN is forced to bin 0. Resulting dataset-wide distribution: 0.190 / 0.199 / 0.204 / 0.208 / 0.200.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```
```python
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "**Percentile bins for running/pupil**: Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)." The AI verified the outcome in Step 12: "**running_speed**: Well-balanced bins", and again in the trajectory: "running_speed: Well-balanced (19-21% per bin)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. By construction: the interpolation target is the full-session `ophys_timestamps` array, and the per-trial slice uses the same `frame_mask` that slices dF/F. So running-speed sample *k* of a trial and neural column *k* of the same trial are the same imaging frame.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
...
running_trial = running_at_ophys[frame_mask]
```

iii. Step 3 of CONVERSION_NOTES.md justifies resampling onto the ophys clock at all: "Synchronization: All streams synced via NI PCI-6612 at 100 kHz". The `--show-processing` plot overlays the raw interpolated speed and its binned version on the trial time axis ("Running and pupil bins distribute across 0-4", Step 7).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/area` (the ellipse-fit pupil area, 30 Hz) plus its `timestamps`, and `acquisition/EyeTracking/likely_blink/data`. Diameter is computed from area as `2*sqrt(area/pi)`. (The SDK's `pupil_width` column, used by the reference, is available in the same group; empirically the two are near-identical — I measured r = 0.9997 between `width` and `2*sqrt(area/pi)` on experiment 775614751.)

ii.
```python
if 'EyeTracking' in f.get('acquisition', {}):
    et = f['acquisition']['EyeTracking']
    if 'pupil_tracking' in et:
        pt = et['pupil_tracking']
        data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
        data['pupil_timestamps'] = pt['timestamps'][:]
        data['likely_blink'] = et['likely_blink']['data'][:]
    else:
        data['pupil_area'] = None
else:
    data['pupil_area'] = None
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Pupil diameter**: Compute from pupil area as `2*sqrt(area/pi)`." Step 10 Check 3 claims the formula is the reference one: "Pupil | area -> diameter via 2*sqrt(area/pi), blinks=NaN | Same formula in whitepaper | Yes". Step 3 records the upstream processing: "Pupil: DeepLabCut tracking, ellipse fit, blinks detected and set to NaN".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN, diameter is computed from the non-NaN positive areas, and the resulting array — **still containing NaN** — is passed to `scipy.interpolate.interp1d`. Linear interpolation through NaN samples propagates NaN, so blinks are *not* bridged: the blink frames plus the interpolation intervals touching them come out NaN at the ophys grid and are subsequently forced to bin 0 by `apply_percentile_bins`. Measured on three experiments, this makes 1.7 %–7.9 % of ophys frames "bin 0 by default" (blink fractions 1.6 %–7.0 %); dropping blink rows before interpolation, as the reference does, yields 0 % NaN. Separately, if an experiment has no eye-tracking group at all, the pupil trace is all-NaN and hence **all bin 0** for the entire session; three of the 202 converted experiments (795953296, 806456687, 833631914) are in this state and were kept.

ii.
```python
if nwb_data['pupil_area'] is not None:
    pupil_area = nwb_data['pupil_area'].copy()
    likely_blink = nwb_data['likely_blink']
    pupil_ts = nwb_data['pupil_timestamps']

    # Set blink frames to NaN
    pupil_area[likely_blink] = np.nan

    # Compute diameter from area: diameter = 2 * sqrt(area / pi)
    pupil_diameter = np.full_like(pupil_area, np.nan)
    valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
    pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)

    # Interpolate to ophys timestamps
    pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Set blink frames to NaN, then interpolate. Discretize non-NaN values." Step 10 Check 5 makes NaN→bin 0 an explicit choice rather than an oversight: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)", and the trajectory (step 62) elaborates: "since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified. I'll leave it as is." The all-NaN no-eye-tracking case is not discussed anywhere.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five equal-percentile bins with edges computed per experiment over that experiment's whole ophys-resampled pupil trace, ±inf outer edges, `np.digitize` + `np.clip`, NaN → bin 0. Because the edges are computed over the whole session but applied only to trial windows, and because NaN is dumped into bin 0, the realised dataset-wide distribution is not equal: 0.200 / 0.199 / 0.214 / 0.221 / 0.166.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```
```python
output_values = [
    ...
    [f'bin_{i}' for i in range(5)],  # pupil diameter percentile bins
    ...
]
```

iii. Same Key Decision 8 as running speed. The AI noticed and accepted the residual imbalance in Step 12: "**pupil_diameter**: Slightly unbalanced due to blink frames mapped to bin 0. Pupil-neural coupling is weaker than stimulus encoding. Expected." and "pupil_diameter: Slightly unbalanced (16.6% to 22.1%), 1.41x chance is reasonable" (trajectory step 75).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the full-session ophys timestamps first, then sliced with the same trial `frame_mask` as the dF/F, so pupil sample *k* and neural column *k* are the same imaging frame.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same justification as 5-d (hardware-synced clocks, single shared frame grid). CONVERSION_NOTES.md Step 10 Check 2 reports a spot-check: "Pupil diameter bins | 775614751 | PASS - np.array_equal=True, blinks=NaN confirmed".

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order. A fifth fallback label `'unknown'` exists in the helper but has no slot in `output_values`.

ii.
```python
def get_trial_outcome(trial_data, idx):
    """Get the trial outcome for a given trial index."""
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "Trial outcome (hit/miss/FA/CR) → `output[4]`: trial_outcome — Categorical, static per trial — 4 categories". Step 10 Check 2 verifies them against the raw file: "Trial outcomes (first 20 trials) | 803736273 | PASS - all match NWB hit/miss/FA/CR". Step 9's consistency table compares the resulting rates to behavioural expectations (hit 30.7 %, miss 56.8 %, FA 1.8 %, CR 10.7 %, all marked "Yes").

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to an integer 0–3 by position in `outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']` and then broadcast as a constant across every timepoint of the trial, i.e. it is stored as row 4 of the `(5, n_timepoints)` output array rather than as a separate per-trial vector. An unmatched label would silently fall back to index 0 (`'hit'`); in practice go/catch non-aborted non-auto-rewarded trials always match one of the four, so this branch is not exercised.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
output_full[0] = img_trace
output_full[1] = change_trace
output_full[2] = running_binned
output_full[3] = pupil_binned
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The instruction lists trial outcome as "Static per-trial" while the target format allows `(n_output, n_timepoints)` or `(n_output,)`; since the other four outputs are time-varying, the AI packed all five into one `(5, T)` array and constant-filled the static one. The code comments record the reasoning explicitly: "The format says: (n_output, n_timepoints) or (n_output,) … Let's make output as (5, T) where last row is constant".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases (all silent-skip or default-value, with a printed warning):
- missing NWB file → experiment skipped (`return None`);
- zero cells → experiment skipped;
- no stimulus-presentation table found → experiment skipped;
- fewer than 2 valid trials before or after processing → experiment skipped;
- a trial window containing < 2 ophys frames → trial skipped;
- behavioural samples outside the recorded range, and blink frames, → NaN → discretized to bin 0;
- no eye-tracking group at all → pupil set to all-NaN → **every** timepoint of that session becomes pupil bin 0, and the session is still kept (3 of 202 sessions: 795953296, 806456687, 833631914);
- a stimulus name not in the global vocabulary → the frame keeps its `'gray'` label (`continue`);
- `get_all_image_names` wraps its per-file read in try/except.

There is no try/except around `process_experiment` itself, so an unexpected failure in one file would abort the whole run.

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}")
    return None
...
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
...
if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}")
    return None
```
```python
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0  # NaN gets bin 0 (lowest bin)
```
```python
if n_trial_frames < 2:
    continue
```

iii. CONVERSION_NOTES.md Step 10 Check 5 ("Check for edge cases") lists only three items — low-trial sessions retained if ≥2 trials, the multiscope frame-rate difference, and "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)" — and Step 10's conclusion is "No critical issues found. All checks pass." The pupil-bin spot-check the AI ran ("blinks=NaN confirmed") was performed on 775614751, which *does* have eye tracking, so the all-NaN sessions were never exercised.

## 9-a. What are the most time-consuming steps of the code?

i. NWB file I/O dominates. The script instruments load vs. process time per experiment and prints both. Summing the 202 printed lines in `conversion_full_out.txt`: 380.8 s loading (73 %) vs 138.0 s processing (26 %), on a 544.7 s total run; the `get_all_image_names` pre-pass adds a further 14.4 s. Within `load_nwb_data` the dF/F read (`(140204, n_cells)` float array) plus the 145 k-sample running/eye arrays are the bulk. Within "process", the dominant cost is the per-trial stimulus-table scan in `build_image_identity_trace` / `build_image_change_trace`.

ii.
```python
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
...
print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
      f"dt: {result['dt']*1000:.1f}ms, "
      f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

iii. CONVERSION_NOTES.md Step 7 "Run Time Estimates" attributes the cost to loading and projects the full run: "Load NWB | ~1.7s | ~340s; Process trials | ~0.4s | ~80s; Total | ~3.7s | ~750s (~12.5 min)", plus "Image name collection adds ~14s overhead". Because the projection (12.5 min) was under the 15-minute budget in the instructions, the AI did no further optimisation; the actual run came in at 9.1 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI identified none — the "Code inefficiencies identified" / "Code speedups added" rows of the Step 6 template were left unfilled. Genuinely vectorizable loops in the delivered code:

- `build_image_identity_trace` and `build_image_change_trace` each walk the **whole stimulus-presentation table from index 0** for **every** trial (`for si in range(len(stim_starts))`, ~4,800 presentations × ~257 trials × 2 functions per experiment). Both could be replaced by a single `np.searchsorted` of the presentation boundaries into `ophys_ts`, computed once per session; this is the bulk of the 138 s of "process" time.
- `image_names_list.index(name)` and `outcome_names.index(outcome)` are linear list scans inside those loops; a dict lookup is O(1).
- The per-trial boolean mask `(ophys_ts >= t_start) & (ophys_ts < t_stop)` scans all ~140,000 session timestamps for each trial; `np.searchsorted` on the sorted timestamps gives the same slice in O(log n).
- The per-trial loop itself (`for trial_idx in valid_trial_idx`) could be replaced by index arithmetic on precomputed start/stop indices.
- `'data' in pt['area']` performs a membership test on an h5py `Dataset`, which falls back to iterating all ~145,000 elements.

ii. The hot loop:
```python
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    if s_stop < trial_start:
        continue
    if s_start >= trial_stop:
        break
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```
and the thrice-recomputed mask:
```python
# in process_experiment
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
# again in build_image_identity_trace
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
# and again in build_image_change_trace
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
```

iii. No justification is given — the AI's only efficiency reasoning is the Step 7 runtime projection, which came in under the 15-minute threshold and therefore triggered no optimisation pass. The script's own docstring/instructions called for vectorized loops, but Step 6 records only feature notes ("Loads NWB files directly via h5py (fast, no AllenSDK overhead)"), not a loop audit.

## 9-c. What processing does the code repeat multiple times?

i. Three real repeats:

1. **Every NWB file is opened twice.** `get_all_image_names` opens all 202 files and reads the entire `image_name` column purely to build the code vocabulary (14.4 s); `load_nwb_data` then re-opens each file and re-reads the same column during conversion. A single pass that collected names while processing (with a post-hoc remap) would eliminate this.
2. **The trial frame mask is computed three times per trial** over the full ~140,000-element `ophys_timestamps` array — once in `process_experiment`, once in `build_image_identity_trace`, once in `build_image_change_trace` — instead of being computed once and passed down.
3. **The stimulus presentation table is rescanned from index 0 for every trial**, in two separate functions, rather than once per session (see 9-b).

ii.
```python
all_image_names = get_all_image_names(exp_table)   # pass 1 over every NWB
...
result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)  # pass 2
```
```python
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```
(`build_image_identity_trace` returns `trial_mask` as its second value; the caller discards it with `_` and uses its own copy.)

iii. No justification is given for the repeats; the AI only noted and accepted the cost of the extra file pass ("Image name collection adds ~14s overhead", Step 7). The global-vocabulary requirement (16 images across image sets A and B) is a legitimate reason to need names up front, which is why the pre-pass exists at all.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:

- **Dead output construction.** `output_tv` (an `np.stack` of the four time-varying rows) and `output_static` are built for every one of the 51,992 trials and then never used — `output_full` is assembled from scratch immediately below.
- **`discretize_percentile()`** is defined (22 lines) and never called; `compute_session_percentile_edges` + `apply_percentile_bins` do the work.
- **`cell_roi_ids`** is read out of the image-segmentation group for every experiment and never used.
- **`is_change` in the trials table** is read into `trial_data` and never used (the stimulus table's `is_change` is used instead).
- **`interpolate_to_ophys` / percentile edges are computed over the entire session**, including the large fraction of timepoints outside any Go/Catch trial, most of which are then discarded (this one is intentional — the edges depend on it — but the interpolated values themselves are mostly thrown away).
- `all_input` allocates a `(0, T)` float32 array per trial even though the task has no decoder inputs.

ii.
```python
# Stack outputs
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)

# Combine: (5, n_trial_frames) for time-varying, last row repeated
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
output_full[0] = img_trace
...
```
```python
def discretize_percentile(values, n_bins=5):
    """Discretize values into n_bins equal percentile bins.
    ...
    """
    # never called
```
```python
all_input.append([np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in result['neural']])
```

iii. No justification is offered — these are leftovers from the AI's iteration on the output layout, visible in the surviving comments around `output_tv` ("Actually, static outputs should be (n_output,) shape … But the format expects a single array per trial… Let's make output as (5, T) where last row is constant"). None of them affect correctness, and their combined cost is small relative to the 73 % of runtime spent on file I/O.
