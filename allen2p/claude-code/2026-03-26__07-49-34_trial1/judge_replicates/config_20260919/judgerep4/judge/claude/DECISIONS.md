# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the NWB (HDF5) files directly with `h5py`. Discovery of what exists is done by (1) reading the project metadata CSV `project_metadata/ophys_experiment_table.csv`, (2) globbing `behavior_ophys_experiments/*.nwb` to find which experiments were actually downloaded, and (3) intersecting the two. It then applies one scope filter — drop any `session_type` containing "passive" — and sorts by `ophys_experiment_id` for reproducibility. Of the 284 downloaded NWB files (239 `VisualBehavior` + 45 `VisualBehaviorMultiscope`, 247 unique `ophys_session_id`, 38 mice), 202 experiments survive (168 single-plane + 34 multiscope), spanning 174 unique `ophys_session_id` and 38 mice. **No `project_code` filter is applied**, so the Multiscope experiments are included. Each surviving experiment is then opened once and all streams (ophys timestamps, dF/F, running speed, eye tracking, trials table, stimulus presentations table) are read out of it in a single `h5py` pass.

ii.
```python
def load_experiment_table():
    """Load the ophys experiment table and filter to active sessions with downloaded NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get list of downloaded NWB experiment IDs
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)

    # Filter to downloaded experiments
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()

    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()

    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table
```

```python
def load_nwb_data(nwb_path):
    """Load all relevant data from a single NWB file."""
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        ...
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed']      = f['processing']['running']['speed']['data'][:]
        ...
        trials = f['intervals']['trials']
        ...
        for key in f['intervals'].keys():
            if 'Natural_Images' in key or 'natural_images' in key:
                stim_key = key
                break
```

iii. From CONVERSION_NOTES Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". Step 10 Check 3 asserts this is "Equivalent" to `BehaviorOphysExperiment.from_nwb()`, since dF/F, running speed and pupil are all pre-computed and stored in the NWB, so the SDK only unpacks them. Passive sessions are excluded because (Step 2) "We should only use active sessions for the decoder task" — in passive replay the lick spout is retracted, so trial outcome is not a meaningful behavioural variable. Step 10 Check 2 reports five `np.allclose` / `np.array_equal` spot checks against freshly re-opened raw NWB files (neural dF/F, running bins, image identity, trial outcomes, pupil bins), all passing, which is the AI's evidence that the h5py path reproduces the SDK path.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the experiment table, taken from the per-experiment row and registered in first-encountered order (which, because the table is sorted by `ophys_experiment_id`, is roughly chronological). `subject_idx` records, for each output session, the index of its mouse. Result: 38 subjects.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
...
# Map subject
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. Not separately argued in CONVERSION_NOTES beyond the Step 5 mapping table entry "`mouse_id` → `subjects`, `subject_idx` | Map unique mice to indices". Step 2 cross-checks the count (38 subjects in the downloaded subset vs. 82 in the full published dataset) and Step 4 records the discrepancy as "Only subset downloaded; proceed with 38".

## 1-c. How are the data split into sessions?

i. **One output "session" == one `ophys_experiment_id` (one imaging plane)**, not one `ophys_session_id`. The script iterates row-by-row over the experiment table and appends one entry to `neural`/`output`/`subject_idx`/`brain_region_idx` per experiment. For the 168 single-plane `VisualBehavior` experiments this is 1:1 with a recording session. For the 34 surviving `VisualBehaviorMultiscope` experiments (8 real `ophys_session_id`s, all from a single mouse, 457841) it is not: the planes of one simultaneously-recorded session are emitted as 4–6 separate "sessions" that share identical behaviour, identical trials and identical output labels, and that run at ~11 Hz instead of ~31 Hz. Final output: 202 sessions, of which mouse 457841 contributes 34 (17%). Passive sessions (OPHYS_2, OPHYS_5) are dropped at the table-filter stage.

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
    all_subject_idx.append(subject_map[mouse_id])
    all_brain_region_idx.append(
        np.full(result['n_cells'], region_map[brain_region], dtype=np.int64)
    )
```
(`ophys_session_id` is recorded only as descriptive metadata, never used to group.)

iii. CONVERSION_NOTES Step 5, Key Decision 9: "**Multiscope handling**: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." Key Decision 10: "**Passive sessions excluded**: Only use active behavior sessions." Step 4 notes the consequence was seen ("For Multiscope sessions, multiple experiments (planes) per session share the same trials/behavior"; "Need to handle both ~31 Hz and ~11 Hz ophys frame rates") and Step 10 Check 5 lists "Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial" as a known edge case, but no mitigation was applied.

## 1-d. How are the data split into trials?

i. Trials come from the NWB built-in `intervals/trials` table. A trial is the full window `[start_time, stop_time)`, giving variable-length trials. Frames are selected by a boolean mask on `ophys_timestamps`. A trial is kept only if it is Go **or** Catch and is neither Aborted nor Auto-rewarded, and only if it contains at least 2 ophys frames. 51,992 trials over 202 sessions (mean ≈ 257/session); T ranges 77–389 frames (mean 233).

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
    t_stop  = trial_data['stop_time'][trial_idx]

    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()

    if n_trial_frames < 2:
        continue

    neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "**Trial definition**: Use `start_time` and `stop_time` from trials table for Go and Catch trials only." Step 3 records the task structure that motivates the full window (change time drawn from a truncated exponential 2.25–8.25 s, mean ~4.2 s; 250 ms flash / 500 ms grey cycle), so the `start_time`→`stop_time` window naturally contains several pre-change flashes plus the post-change response window — which is what makes the time-varying outputs meaningful. Step 7 sanity-checked an anomalous 39-trial session and traced it to 1,078 aborted trials in a poorly-performing mouse, concluding the trial split was behaving correctly.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, at three levels:
- **Trial type**: `(go | catch) & ~aborted & ~auto_rewarded`.
- **Trial length**: trials yielding `< 2` ophys frames are dropped.
- **Session, pre-processing**: an experiment with `< 2` Go/Catch trials, with 0 cells, with no stimulus-presentations interval, or with a missing NWB file is dropped entirely (`return None`).
- **Session, post-processing**: an experiment that ends up with `< 2` usable trials is dropped entirely.

No engagement/reward-rate filter, no d-prime filter, and no neuron-level filter are applied.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
    return None
...
    if n_trial_frames < 2:
        continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules (for this decoder task): Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials" — taken straight from the Decoder Task section of the instructions. Step 3 also notes the whitepaper's reasons: aborted trials are early-lick trials where the change was never shown, and auto-rewarded trials (first 5 of a session, plus after 10 consecutive misses) deliver a free reward that biases the response. The ≥2-trial rule follows the target-format requirement "There needs to be at least two trials within each session in order to evaluate the decoder performance". Step 10 Check 5 explicitly accepts low-trial sessions: "Sessions with very few valid trials (e.g., 39) are retained if >=2 trials". Piet et al.'s engagement threshold (>2 rewards/min) is recorded in Step 3 but deliberately not applied, since the decoder task says nothing about engagement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The pre-computed dF/F traces stored at `processing/ophys/dff/traces/data` in the NWB, with the matching frame times at `processing/ophys/dff/traces/timestamps`. The HDF5 array is stored as (n_frames, n_cells) and is transposed to (n_cells, n_frames). Deconvolved `event_detection` traces were explicitly considered and rejected.

ii.
```python
# Ophys timestamps
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]

# DFF traces: shape (n_frames, n_cells) -> transpose to (n_cells, n_frames)
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "**Neural data**: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 1 confirms both are present in the NWB (`DFFTraces`, and `Events` from FastLZeroSpikeInference) and Step 3 records the SDK's dF/F recipe (600 s median-filter baseline, 3.33 s median-filter detrending), i.e. the normalisation has already been done upstream.

## 2-b. How is the `neural` data processed?

i. Essentially none. The dF/F matrix is transposed to (n_cells, n_frames), sliced per trial with the trial frame mask, and cast to `float32`. No smoothing, no z-scoring, no baseline subtraction, no normalisation, no rebinning, and no merging of imaging planes (each plane is its own output session — see 1-c). Each neuron is tagged with the experiment's `targeted_structure` (VISp or VISl) for `brain_region_idx`.

ii.
```python
dff = nwb_data['dff_traces']  # (n_cells, n_frames)
n_cells, n_frames = dff.shape
...
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
...
brain_region = exp_row['targeted_structure']
...
all_brain_region_idx.append(
    np.full(result['n_cells'], region_map[brain_region], dtype=np.int64)
)
```

iii. CONVERSION_NOTES Step 1: "dF/F is pre-computed in NWB files (no need to compute from scratch)"; Step 3 Processing Details records that the Allen pipeline already applies motion correction, neuropil correction, baseline normalisation and detrending. Step 10 Check 3 lists "dF/F computation | Pre-computed in NWB | Pre-computed (600s median filter + detrending) | Yes" as matching the reference. `float32` is used per the instructions' "Use appropriate data types (float32 vs. float64)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filter is applied. All ROIs present in the NWB `dff` matrix are kept. The AI considered adding an explicit `valid_roi == True` filter and decided against it after verifying it would be a no-op.

ii. N/A — there is no filtering code. The relevant fact is that the raw matrix is used unmodified:
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
...
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```
(`cell_roi_ids` is read from `image_segmentation` but never used.)

iii. CONVERSION_NOTES Step 1 notes that the SDK's `exclude_invalid_rois=True` is the default curation step, and Step 10 Check 3 records: "ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid". The trajectory (step 60) shows the AI briefly intended to "add the valid_roi filter for safety" and then dropped it. I independently confirmed the claim: for experiments 775614751 and 788490510 the `cell_specimen_table/valid_roi` arrays are all-True and their lengths (89, 142) equal both the dF/F cell count and the `ophys_cells_table.csv` count, so the published NWBs contain only valid ROIs. Neuron totals also reconcile (29,444 neurons over 202 experiments, mean 146, vs. 42,147 over all 284 downloaded experiments, mean 148).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything is placed on the ophys timebase, and each trial is the set of ophys frames whose timestamp falls in `[trial.start_time, trial.stop_time)`. The alignment event is therefore **trial start**, and the same boolean `frame_mask` is reused to slice neural, running and pupil, so all streams are aligned by construction. `metadata['off_start']` and `metadata['off_end']` are set to `None` because the window is variable-length rather than a fixed offset from an event.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
neural = dff[:, frame_mask].astype(np.float32)
...
running_trial = running_at_ophys[frame_mask]
pupil_trial   = pupil_at_ophys[frame_mask]
```
```python
'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
'off_start': None,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "**Temporal alignment**: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time." This follows the instruction "Temporally align based on ophys timestamp." Step 3 records that all hardware streams are synchronised by an NI PCI-6612 board at 100 kHz, which is the AI's justification for treating running/eye/stimulus times as directly comparable to ophys times. Step 10 Check 2 verified alignment by spot-checking the image-identity trace against stimulus onsets in the raw NWB (`np.array_equal` = True), and `--show-processing` plots per-trial neural, image identity, image change, running and pupil on a common time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning or resampling of the neural data.** Each session keeps its native ophys frame rate. The per-experiment bin size is the median inter-frame interval. In the delivered dataset this is **not uniform**: 168 sessions are at 32.3 ms (~30.9 Hz, Scientifica) and 34 sessions are at 93.2 ms (~10.7 Hz, Multiscope). A single scalar `metadata['time_bin_size']` is written as the **median across sessions**, i.e. 32.32 ms — which is wrong by ~3× for the 34 Multiscope sessions. The per-session true value is preserved only inside `metadata['session_info'][i]['dt_ms']`.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
# Compute median time bin size across sessions
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**Time bin**: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." Step 4 lists "Need to handle both ~31 Hz and ~11 Hz ophys frame rates" as an open item and Step 10 Check 5 restates it as an accepted edge case. The trajectory (step 52) shows the AI noticed the consequence directly — "the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica), because Multiscope records at ~11 Hz" — but it did not reconcile this with the target-format requirement "Time bins should be the same size for all trials and sessions", nor did it flag that the reported `time_bin_size` misdescribes 34/202 sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentations** table (`intervals/Natural_Images_*_presentations`), specifically `image_name`, `start_time` and `stop_time` — not the trials table. Each ophys frame is labelled with the image actually on screen at that instant; frames in the 500 ms inter-stimulus grey period, and frames during omitted flashes, are labelled with an explicit extra category `'gray'`. The category list is `['gray'] + sorted(unique image names)` = 17 classes (gray + 16 images across image sets A and B).

ii.
```python
# Initialize to gray (ISI)
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)

stim_starts = stim_data['start_time']
stim_stops  = stim_data['stop_time']
stim_names  = stim_data['image_name']

for si in range(len(stim_starts)):
    s_start = stim_starts[si]; s_stop = stim_stops[si]
    if s_stop < trial_start:  continue
    if s_start >= trial_stop: break
    name = stim_names[si]
    if isinstance(name, bytes): name = name.decode('utf-8')
    # Skip omitted stimuli (they're gray screen)
    if name == 'omitted':  continue
    if name in image_names_list: img_idx = image_names_list.index(name)
    else: continue
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "**Image identity**: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." The Step 5 mapping table sources it from "`image_name` from stimulus presentations". The AI's consistency argument (Step 9 / Step 10 Check 4) is that the resulting `gray` fraction should equal the duty cycle of the stimulus: expected 500/750 = 66.7 %, measured 66.9 % — it treats this as the sanity check that the flash windows were mapped onto ophys frames correctly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A single global, deterministic name→integer mapping shared by all sessions. Before any trial is processed, every NWB file in the (filtered) experiment table is opened and its `image_name` column scanned; `'omitted'` is dropped; the union is sorted; `'gray'` is prepended so it is always index 0. Per frame the value is a plain `int64` index into that list, stored as row 0 of the per-trial output matrix. The names are exported in `output_values[0]` and in `metadata['image_names']`.

ii.
```python
def get_all_image_names(exp_table):
    """Scan a few NWB files to collect all unique image names across sessions."""
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
                            if isinstance(n, bytes): n = n.decode('utf-8')
                            if n != 'omitted': all_names.add(n)
                        break
        except Exception as e:
            print(f"  WARNING: Could not read images from {eid}: {e}")
    return sorted(all_names)
```
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names   # 'gray' is always index 0
...
output_full[0] = img_trace
```

iii. Not argued at length; the Step 5 mapping table says "Map to categorical integer, time-varying at ophys rate | 8 images + gray screen". A global mapping is required because the dataset mixes image set A and image set B (Step 2: "Images per session | 8 unique natural images"; conversion log: 16 unique images overall), so per-session codes would not be comparable. Step 7 flagged and accepted the consequence: "`image_identity` has value 16 meaning some sessions use different image sets". Step 9 checks the resulting distribution (gray 0.669, each image ≈ 0.020–0.022) against the expected 66.7 % grey duty cycle.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the trial's ophys timestamps, using the same `[t_start, t_stop)` window as the neural slice, so it is frame-for-frame aligned by construction. Within the trial, a frame gets image *i* iff its ophys timestamp falls inside that flash's `[start_time, stop_time)` (i.e. inside the 250 ms presentation); otherwise it keeps the default `gray`. There is no interpolation, shifting or smoothing.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    ...
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```
The identical window is used for the neural data:
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5, Key Decision 4 (align everything to ophys timestamps). Step 5 Planned Sanity Checks include "Verify image identity changes at correct times (compare is_change with actual image transitions)" and "Verify no temporal misalignment by plotting neural + stimulus for sample trials"; Step 10 Check 2 reports "Image identity trace | 803736273 | PASS - np.array_equal=True, verified at stimulus onset", and `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) show neural, image identity and image change on a shared time-from-trial-start axis.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` flag of the **stimulus presentations** table, together with that presentation's `start_time`. (The trials table's `is_change`/`change_time` columns are read into `trial_data` but never used for this output.) Because `is_change` marks only flashes where the image actually differs from the previous one, catch trials — where the "change" is a sham repeat of the same image — contribute no change frames at all.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    """Build a binary trace that is 1 at the first frame after an image change."""
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)
    if n_frames == 0:
        return np.array([], dtype=np.int64)
    trace = np.zeros(n_frames, dtype=np.int64)
    stim_starts = stim_data['start_time']
    is_change   = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        ...
```

iii. Step 5 mapping table: "`is_change` from stimulus presentations | `output[1]`: image_change | Binary, 1 at change onset frame, 0 otherwise | Time-varying". Using the stimulus table rather than the trials table keeps `image_change` definitionally consistent with `image_identity` (3-a), which is also built from that table — Step 5's planned sanity check "Check that image_change=1 aligns with actual change in image_identity" depends on that. I verified the catch-trial consequence numerically: the measured change fraction is 0.372 %, and (hit+miss = 87.5 % of trials are Go) × (1 frame / 233 mean frames) = 0.375 %, confirming catch trials carry no change frame.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Minimal, and deliberately **sparse**: for each change flash inside the trial, `np.searchsorted` finds the first ophys frame at or after the flash onset and sets exactly **that one frame** to 1. Everything else is 0. No window, no smoothing, no extension over the flash or the following grey period. With ~31 Hz sampling this makes the positive class one 32 ms bin per Go trial — 0.4 % of all timepoints (the AI's own log: `image_change: {no_change (0.996), change (0.004)}`).

ii.
```python
        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue

        # Find the first ophys frame at or after the change onset
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1

    return trace
```

iii. The AI read the instruction "Have value of 1 right after a change in image identity, otherwise 0" literally: Step 5 mapping table says "Binary, 1 at change onset frame, 0 otherwise". It then defended the resulting imbalance rather than the labelling width — Step 12: "**image_change** (1.27x): Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily. The decoder correctly identifies change timepoints but the signal is very sparse. Not a conversion bug." Step 10 Check 4 treats the 0.372 % change fraction as a passing statistic ("~0.3-0.4% | PASS").

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required or performed — the variable is binary by construction. The trace is initialised to 0 (`no_change`) and set to 1 (`change`) at change-onset frames; it is `int64` and its declared value names are `['no_change', 'change']`. The verification log confirms the realised range is exactly [0, 1] in every session.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
    trace[frame_idx] = 1
```
```python
output_values = [
    image_names_list,          # image identity categories
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
```

iii. The instruction already specifies a binary variable ("Image change, binary variable"), so no discretisation rule was needed; the AI simply recorded the two value names. Step 9's consistency table and Step 10 Check 4 report the binary distribution (0.996 / 0.004) rather than any threshold.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same construction as image identity: the trial mask `[t_start, t_stop)` on `ophys_ts` defines `trial_ts`, and `np.searchsorted(trial_ts, s_start)` picks the first ophys frame at or after the change. So the label lands on the same frame index grid as the neural slice, with at most half a frame (~16 ms at 31 Hz, ~47 ms at 11 Hz) of quantisation error, and always rounds *forward* (the change is never marked before it happened).

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```
vs. the neural slice built from the identical window:
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Same justification as 3-c: CONVERSION_NOTES Step 5 Key Decision 4 (align everything on the ophys timebase). Step 5 planned sanity check "Check that image_change=1 aligns with actual change in image_identity"; Step 10 Check 2 verified the image-identity trace against raw stimulus onsets, and the `--show-processing` panels plot `image_change` directly beneath `image_identity` on the same axis so the 1 can be seen to coincide with the identity transition.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The NWB running module: `processing/running/speed/data` (cm/s, ~60 Hz, already 10 Hz-lowpass-filtered by the Allen pipeline) with its own `timestamps`. This is the same array the SDK exposes as `dataset.running_speed`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed']      = f['processing']['running']['speed']['data'][:]
```
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
```

iii. Step 5 mapping table: "`running_speed` | `output[2]`: running_speed_bin". Step 2 records the stream's location and rate ("Running: `processing/running/speed` (270K samples @ 60 Hz)"); Step 3 records that the Allen pipeline already applies a "10 Hz lowpass Butterworth filter", so the AI treats the stored signal as final and applies no additional filtering.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps. (1) Linear interpolation from the 60 Hz encoder timebase onto the full session's ophys timestamps, with `bounds_error=False, fill_value=np.nan` so any ophys frame outside the running record becomes NaN. This is done **once per experiment** for the whole session, before trial segmentation. (2) Percentile discretisation into 5 bins (see 5-c). No smoothing, rectification, or absolute-value step is applied — negative speeds (backwards wheel motion) are kept as-is.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    """Interpolate a behavioral signal to ophys timestamps using linear interpolation."""
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
running_trial  = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. Step 5, Key Decision 7: "**Running speed**: Interpolate from 60 Hz to ophys timestamps using linear interpolation." Justified in Step 3 by the hardware synchronisation ("All streams synced via NI PCI-6612 at 100 kHz"), so the two clocks are directly comparable. Interpolating the whole session once (rather than per trial) is the efficiency choice that also guarantees identical treatment of every trial. Step 5 planned sanity check "Verify running speed range is reasonable (typically 0-80 cm/s)"; Step 10 Check 2 reports "Running speed bins | 803736273 | PASS - np.array_equal=True" against the raw NWB.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count percentile bins at the 0/20/40/60/80/100th percentiles — but the edges are computed **per session, over the entire session's ophys frames** (including inter-trial and non-trial periods), not globally across the dataset and not restricted to trial frames. The outer edges are replaced with ±inf so nothing falls outside, NaNs are excluded from the percentile computation, and NaN frames are then assigned bin 0.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    """Compute percentile bin edges from session-wide values (excluding NaN)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    """Apply pre-computed percentile bin edges to values."""
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```
```python
# Compute session-wide percentile bin edges
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
```

iii. Step 5, Key Decision 8: "**Percentile bins for running/pupil**: Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)." This implements the instruction "discretized into five equal percentile bins" at the session level. The AI verified the resulting marginals in Step 12: "**running_speed**: Well-balanced (19-21% per bin)" — pooled fractions were 0.190 / 0.199 / 0.204 / 0.208 / 0.200. Per-session fractions are much less uniform (bin 0 ranges 0.058–0.363 across sessions) because the edges come from all session frames while only trial frames are exported; this was not examined.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Because the interpolation target is the session's full `ophys_timestamps` array, `running_at_ophys` is index-for-index parallel with the dF/F matrix's time axis. The trial slice then reuses the *same* `frame_mask` object that slices the neural data, so alignment is exact and cannot drift. Binning happens after slicing and is a pointwise map, so it does not disturb alignment.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
neural = dff[:, frame_mask].astype(np.float32)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
...
output_full[2] = running_binned
```

iii. Step 5, Key Decision 4 ("Align to ophys timestamps"), resting on the Step 3 finding that all streams are hardware-synced at 100 kHz. The `--show-processing` panels overplot the raw interpolated cm/s trace (dashed, right axis) against the binned trace (solid, left axis) on the trial's time axis, which is the AI's visual check that the bin trace tracks the continuous signal with no lag; Step 7 records "Running and pupil bins distribute across 0-4" with no anomalies.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the fitted pupil-ellipse **area**, ~30 Hz) plus its `timestamps`, and `acquisition/EyeTracking/likely_blink/data`. Diameter is then computed from area rather than read from a width/height column. If the `EyeTracking` group or `pupil_tracking` is absent, pupil is set to all-NaN for that experiment.

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

iii. Step 5 mapping table: "`pupil_tracking/area` → diameter | `output[3]`: pupil_diameter_bin". Step 2 records the available columns ("Eye tracking: `acquisition/EyeTracking/pupil_tracking` (area, height, width @ 30 Hz)") and "Blinks: `acquisition/EyeTracking/likely_blink`". Step 3 records the upstream processing ("DeepLabCut tracking, ellipse fit, blinks detected and set to NaN"). Step 10 Check 3 claims the area→diameter formula is "Same formula in whitepaper".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Four steps: (1) frames flagged `likely_blink` are set to NaN **in the source array**; (2) diameter is computed as `2·sqrt(area/π)` for the remaining strictly-positive samples (non-positive areas also become NaN); (3) the resulting array — NaNs included — is passed to `interp1d` and evaluated on the session's ophys timestamps; (4) the result is percentile-binned (6-c). Critically, the blink samples are **not removed before interpolation**, so `interp1d` cannot bridge them: every ophys frame in a blink interval (and in the two segments adjoining it) evaluates to NaN and is subsequently mapped to bin 0, the *smallest-pupil* category. Experiments with no eye tracking at all get an all-NaN trace and therefore a constant bin-0 output.

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
```python
result[~valid] = 0   # in apply_percentile_bins: NaN -> bin 0
```

iii. Step 5, Key Decision 6: "**Pupil diameter**: Compute from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate. Discretize non-NaN values." The NaN→bin-0 mapping was raised and explicitly dismissed in the trajectory (step 62): "The NaN pupil -> bin 0 issue: since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified. I'll leave it as is since it's a valid design choice." Step 10 Check 5 repeats: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)." Step 12 acknowledges the cost: "**pupil_diameter** (1.41x): Slightly unbalanced due to blink frames mapped to bin 0." The realised per-session bin-0 fractions confirm this is not slight: they run up to 0.427 and are exactly 1.000 in three sessions that have no eye tracking.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five equal-count percentile bins (0/20/…/100th) computed per session over all that session's non-NaN ophys-resampled pupil values, with ±inf outer edges, `np.digitize` + `np.clip`, and NaN→bin 0. The same two helper functions are reused.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial  = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
output_full[3] = pupil_binned
```
```python
output_values = [ ..., [f'bin_{i}' for i in range(5)],  # pupil diameter percentile bins
                  ... ]
```

iii. Step 5, Key Decision 8 (same rule as running speed, applied per session), implementing the instruction "discretized into five equal percentile bins". Step 12 checked the outcome: "**pupil_diameter**: Slightly unbalanced (16.6% to 22.1%)". The imbalance is entirely attributable to the blink/missing-data handling described in 6-b, since the percentile edges themselves are by construction equal-count over the valid samples.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Exactly as for running speed: the pupil signal is resampled once per experiment onto the session's full `ophys_timestamps`, then sliced with the same `frame_mask` used for the neural data, so pupil sample *t* and dF/F column *t* refer to the same ophys frame. No per-trial interpolation and no shifting.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Step 5, Key Decision 4, plus the Step 3 note that the eye-tracking camera, the running encoder and the 2P frame clock are all synchronised on the same 100 kHz NI board, so interpolating the 30 Hz eye signal onto the ~31 Hz ophys clock is valid. Step 10 Check 2 records "Pupil diameter bins | 775614751 | PASS - np.array_equal=True, blinks=NaN confirmed", and the `--show-processing` panel overlays raw interpolated pupil against the binned trace on the trial time axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — tested in that fixed order for the trial's row index. The category list is `['hit', 'miss', 'false_alarm', 'correct_reject']` (codes 0–3). There is a fallback string `'unknown'` for a trial matching none of the four.

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
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. Step 5 mapping table: "Trial outcome (hit/miss/FA/CR) | `output[4]`: trial_outcome | Categorical, static per trial | 4 categories". These are the SDK's canonical labels for the change-detection task and, with aborted and auto-rewarded trials already removed, they exactly partition the Go/Catch trials. Step 9 cross-checks the realised marginals against behavioural expectation (hit 30.7 % vs "~30% typical", miss 56.8 % vs "~57%", FA 1.8 % vs "~2%", CR 10.7 % vs "~11%"), and Step 10 Check 2 verified the first 20 trials of session 803736273 against the raw NWB booleans.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The label string is mapped to its index in `outcome_names` and then **broadcast across every timepoint of the trial**, becoming row 4 of the `(5, n_trial_frames)` output matrix. So although it is conceptually static per trial, it is stored as a constant time series — which is what lets all five outputs share a single rectangular array per trial. The `'unknown'` fallback is mapped to index `0`, i.e. it would be silently relabelled as `'hit'`.

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

iii. The instructions require that all outputs for a trial be a single array of shape `(n_output, n_timepoints)` or `(n_output,)`, and four of the five outputs are time-varying; the inline comments in `process_experiment` show the AI working through exactly this ("static outputs should be (n_output,) shape … The format says: (n_output, n_timepoints) or (n_output,) … Let's make output as (5, T) where last row is constant"). The instructions also say "If at all possible, make it time-varying". The `'unknown'`→0 fallback is not discussed anywhere; in practice it never fires, because after the aborted/auto-rewarded exclusion every Go/Catch trial has exactly one of the four flags set, and the verification log shows `trial_outcome` ranges within [0, 3] with a plausible distribution.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing NWB file** for a listed experiment → warn and skip the experiment (`return None`).
- **Zero cells** → warn and skip the experiment.
- **No stimulus-presentations interval found** → warn and skip the experiment.
- **Fewer than 2 Go/Catch trials**, before or after processing → warn and skip the experiment.
- **Trials with < 2 ophys frames** → skip the trial. (Trials whose `stop_time` runs past the end of the recording need no explicit clipping because the boolean `frame_mask` on `ophys_ts` is inherently bounded.)
- **Missing / out-of-range behavioural samples** → `interp1d(..., bounds_error=False, fill_value=np.nan)` yields NaN, and `apply_percentile_bins` maps NaN → bin 0.
- **Missing eye tracking entirely** → all-NaN pupil trace → constant bin 0 for the whole session (3 sessions in the output).
- **Blink frames** → NaN → bin 0 (see 6-b).
- **Unreadable image_name column** during the pre-pass → caught by `try/except`, warn and continue.
- **Unrecognised trial outcome** → silently coded as 0 (`'hit'`).
- **Unrecognised image name** → `continue`, leaving the frame as `gray`.

Not handled: there is no `try/except` around `process_experiment`, so a genuinely corrupt NWB would abort the whole run (it did not occur — all 202 experiments processed).

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}");  return None
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}");  return None
if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}");  return None
if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}");  return None
...
    if n_trial_frames < 2:
        continue
```
```python
f = interpolate.interp1d(signal_ts, signal, kind='linear',
                         bounds_error=False, fill_value=np.nan)
...
result[~valid] = 0   # NaN -> bin 0
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```
```python
except Exception as e:
    print(f"  WARNING: Could not read images from {eid}: {e}")
```

iii. The instructions ask to "Handle missing data appropriately (consult references, use sensible defaults, document)". The AI's documented rationale is Step 10 Check 5: "Sessions with very few valid trials (e.g., 39) are retained if >=2 trials" (the ≥2 floor comes from the target format's "at least two trials within each session"), and "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)" — i.e. it preferred keeping the class count at the specified 5 over introducing a 6th "missing" category. The conversion log shows no warnings were emitted in the full run, which the AI took (Step 10) as evidence that no experiment hit any of these paths; the all-NaN-pupil sessions nevertheless survived silently because the missing-pupil branch produces a legal, if degenerate, output rather than a warning.

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented load vs. process time separately and reported both per experiment. Reading the NWB file dominates: ~1.4–1.8 s per experiment to load (mostly pulling the full `(n_frames, n_cells)` dF/F array into memory) versus ~0.1–0.5 s to segment and build outputs. On top of that there is a 14.4 s one-off pre-pass (`get_all_image_names`) that re-opens all 202 NWB files. Full run: 544.7 s (9.1 min), ~2.7 s/session — inside the instructions' 15-minute target, so no optimisation was undertaken.

ii.
```python
t0 = time.time()
...
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
...
print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
      f"dt: {result['dt']*1000:.1f}ms, "
      f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

iii. CONVERSION_NOTES Step 7 Run Time Estimates: "Load NWB ~1.7s/session → ~340s; Process trials ~0.4s → ~80s; Total ~3.7s → ~750s (~12.5 min)"; "Image name collection adds ~14s overhead." Step 6 justifies the h5py route partly on this basis ("fast, no AllenSDK overhead"). The realised 9.1 min came in under the 12.5 min estimate, so Step 9's instruction to "optimize bottlenecks first" was not triggered.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not identify any (CONVERSION_NOTES Step 6's "Code inefficiencies identified" and "Code speedups added" fields were left effectively empty, and the only speed argument made is h5py-vs-SDK). Reading the code, the clear candidates are:
- `build_image_identity_trace` and `build_image_change_trace` each scan the session's stimulus-presentations table **from index 0 on every trial**, breaking only once they pass the trial's stop time. With ~4,800 presentations and ~250 trials that is a quadratic ~1.2 M Python iterations per experiment, done twice. Both could be replaced by two `np.searchsorted` calls to locate the presentations overlapping the trial, plus vectorised index assignment.
- `image_names_list.index(name)` is a linear list scan executed inside that inner loop; a dict lookup (as the human reference uses) is O(1).
- Inside those loops, `(trial_ts >= s_start) & (trial_ts < s_stop)` builds a fresh boolean array per presentation; `np.searchsorted` on the sorted `trial_ts` would give the same frame span directly.
- The outer per-trial loop itself could be vectorised into a single `np.searchsorted(ophys_ts, [start_times, stop_times])` pass.

ii. The unvectorised inner scan, in both builders:
```python
    for si in range(len(stim_starts)):
        s_start = stim_starts[si]
        s_stop = stim_stops[si]
        if s_stop < trial_start:
            continue
        if s_start >= trial_stop:
            break
        ...
        if name in image_names_list:
            img_idx = image_names_list.index(name)
        ...
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```

iii. No justification is offered, because the inefficiency was never noted. The implicit defence is the Step 7 timing analysis: processing is ~0.4 s per experiment against ~1.7 s of I/O, so the loops are not the bottleneck and the run finished well inside the 15-minute budget the instructions set as the trigger for optimisation.

## 9-c. What processing does the code repeat multiple times?

i. Three repeats, all identified by reading the code rather than by the AI:
- **Every NWB file is opened twice.** `get_all_image_names` opens all 202 files to read the `image_name` column (14.4 s), and `process_experiment` then re-opens each one. The image names could have been accumulated during the single main pass.
- **The trial frame mask is computed three times per trial**, each time as a full comparison over the session's ~150,000-element `ophys_timestamps` array: once in `process_experiment`, once inside `build_image_identity_trace`, and once inside `build_image_change_trace`. The mask is not passed down; only `ophys_ts`, `t_start`, `t_stop` are.
- **The stimulus-presentations table is re-scanned from index 0 for every trial**, twice per trial (identity builder + change builder), instead of advancing a cursor or using `searchsorted`.

Not repeated: NWB I/O within an experiment (one `h5py` context reads all streams), the running/pupil interpolation (once per session, not per trial), and the percentile edges (once per session).

ii.
```python
# in process_experiment
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
img_trace, _   = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
change_trace   = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```
```python
# recomputed in build_image_identity_trace
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
# and again in build_image_change_trace
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
```
```python
# first full pass over every NWB
all_image_names = get_all_image_names(exp_table)
...
# second full pass over every NWB
nwb_data = load_nwb_data(nwb_path)
```

iii. Not discussed in CONVERSION_NOTES; Step 6 records no inefficiencies. The only related remark is Step 7's "Image name collection adds ~14s overhead", stated as a fact rather than as something to remove. The design does have a defensible motive — the global image-code mapping has to be known before any trial is encoded, and doing it as a cheap pre-pass avoids a second in-memory assembly stage — but the same could have been achieved with one pass plus deferred encoding, as the human reference does.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of dead work, none expensive:
- **`output_tv` and `output_static` are computed and immediately thrown away** (lines 412–413). `output_full` is then built from scratch from the same four traces plus `outcome_idx`. The `np.stack` and dtype cast are pure waste, and the surrounding comment block shows this is a leftover from the AI reasoning its way to the final format in-place.
- **`discretize_percentile()` is defined but never called** — it is superseded by the `compute_session_percentile_edges` / `apply_percentile_bins` pair.
- **`cell_roi_ids` is read out of `image_segmentation`** on every file and never used.
- **`is_change` is read from the trials table** and never used (the stimulus-table `is_change` is used instead), as is the stimulus table's `omitted` column (omissions are detected by comparing `image_name == 'omitted'`).
- Minor: `build_image_identity_trace` returns `trial_mask` as a second value, which every caller discards (`img_trace, _ = ...`).

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
def discretize_percentile(values, n_bins=5):   # never called
    ...
```
```python
            if 'id' in seg[key]:
                data['cell_roi_ids'] = seg[key]['id'][:]   # never used
```
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
             'initial_image_name', 'change_image_name', 'is_change']:
```
(`change_time`, `initial_image_name`, `change_image_name` and `is_change` are all loaded from the trials table and none is used.)

iii. Not documented — Step 13's cleanup was about moving analysis artefacts into `/app/cache/`, not about pruning dead code in `convert_data.py`. The cost is negligible relative to the ~1.7 s/experiment of NWB I/O, which is presumably why it survived; the `output_tv`/`output_static` remnant in particular is a readability hazard, since a reader could reasonably believe it is the array that gets stored.
