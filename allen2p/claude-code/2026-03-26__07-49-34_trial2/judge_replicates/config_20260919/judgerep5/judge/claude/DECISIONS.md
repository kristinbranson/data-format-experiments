# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object API. It reads the released NWB files directly with `h5py` from the local copy of the dataset, and uses the project metadata CSV (`ophys_experiment_table.csv`) only for experiment-level metadata (mouse id, targeted structure, cre line, `passive` flag). The set of data to convert is defined as: every `behavior_ophys_experiment_*.nwb` file present on disk (284 files) whose row in the experiment table has `passive == False` → **202 active experiments from 38 mice**. No filter on `project_code` is applied, so both `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` (multi-plane) experiments are included. Each NWB file is opened once per pass; the script makes **two full passes** over all files (pass 1 to accumulate running/pupil values for global percentile edges, pass 2 for the actual conversion).

ii.
```python
NWB_DIR = 'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = 'data/visual-behavior-ophys-1.1.0/project_metadata/'

def get_experiment_metadata():
    """Load experiment metadata and identify available NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    nwb_map = {}
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_map[eid] = f

    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    # Only active behavior sessions
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map
```
Inside `process_experiment`, every stream is read straight out of the HDF5 tree:
```python
dff_data  = f['processing']['ophys']['dff']['traces']['data'][:]      # (timepoints, neurons)
ophys_ts  = f['processing']['ophys']['dff']['traces']['timestamps'][:]
stim      = f['intervals'][stim_key]          # image-flash presentations
trials    = f['intervals']['trials']
running_speed = f['processing']['running']['speed']['data'][:]
pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
```
Two-pass driver:
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):       # Pass 1
    stats = process_experiment(nwb_map[eid], row, collect_stats_only=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):       # Pass 2
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
```

iii. From CONVERSION_NOTES Step 1/4 and the trajectory: the AI read the SDK source (`BehaviorOphysExperiment.from_nwb_path`, `DFFTraces.from_nwb`, `RunningSpeed.from_nwb`, `EyeTrackingTable.from_nwb`, `Trials.from_nwb`) to learn exactly which NWB paths each SDK accessor maps to, then replicated those reads with `h5py` (Step 10 Check 3 records "h5py direct NWB read ≡ `BehaviorOphysExperiment.from_nwb_path()`"). Passive experiments were dropped because "Passive sessions have no meaningful trial outcomes (no licking)" (Step 5, decision 3). The AI noted that the 284 local files are a subset of the 1,165 released planes and treated the subset as the population to convert.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained (active) experiments, cast to `str` and sorted. `subject_idx` for each session is the index of that experiment's mouse in this list. 38 subjects result.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. `mouse_id` is the Allen metadata table's canonical animal identifier; no justification beyond that is given in CONVERSION_NOTES (Step 2 records "38 in our active subset" as the expected count and Step 9 verifies 38 subjects in the output).

## 1-c. How are the data split into sessions?

i. **One NWB file (= one imaging plane = one `ophys_experiment_id`) is treated as one "session."** Planes recorded simultaneously within the same `ophys_session_id` are *not* merged. Because `VisualBehaviorMultiscope` experiments are included, the 6 active multiscope sessions are emitted as 34 separate pseudo-sessions that repeat the same trials/behaviour with different neurons (visible in the verification log as runs of identical trial counts — `209` ×7, `287` ×7, `309` ×7, `239` ×5, `196` ×5 — and as "Subject 457841: 34 sessions"). Sessions appear in ascending `ophys_experiment_id` order, not chronological order.

ii.
```python
active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    result = process_experiment(nwb_map[eid], row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
```

iii. Trajectory step 31: "Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session." CONVERSION_NOTES Step 2 states "Each NWB file = one imaging plane from one session" and Step 4 acknowledges "For multiscope sessions, one session can have multiple experiments (planes)", but no merging was implemented and the consequence (duplicated trials across pseudo-sessions) is not analysed.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go | catch) & ~aborted & ~auto_rewarded`. The trial's time window is defined *indirectly*: all image-flash presentations whose `trials_id` equals the trial's `id`. Each such presentation becomes one 750 ms time bin, so trials have variable length (mean 11.66 bins ≈ 8.7 s; observed windows match the table's `start_time`→`stop_time` to within one flash).

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    # Find stimulus presentations for this trial
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules — Per task instructions: Include Go and Catch trials, exclude Aborted and Auto-rewarded") and Step 5 decision 1 ("Time bin = 750ms (1 per stimulus flash): Natural task unit … Each bin = 250 ms image + 500 ms grey"). The AI's Step 10 Check 4 verifies the resulting Go/Catch split is exactly 87.5 % / 12.5 %, matching the 7/8–1/8 change-transition matrix in the whitepaper.

## 1-e. How are trials filtered based on quality controls?

i. Filters actually applied: (a) aborted and auto-rewarded trials removed; (b) trials with zero associated stimulus presentations skipped; (c) experiments with zero `valid_roi` cells skipped; (d) experiments yielding `< 2` trials skipped (format requirement); (e) whole passive experiments excluded upstream. No engagement/d-prime/lick-bout filter and no explicit `change_time` validity check are applied (the AI verified `(go|catch)&~aborted&~auto` is already equivalent to the reference's `~aborted & ~auto & change_time.notna()`). No `try/except` guard exists around per-experiment processing. The AI reports 0 experiments were actually skipped (minimum trials per session = 39).

ii.
```python
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
...
stim_key = find_stim_key(f)
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
    return None
...
trial_stim_indices = np.where(stim_trials_id == tid)[0]
if len(trial_stim_indices) == 0:
    continue
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```

iii. CONVERSION_NOTES Step 3 lists the whitepaper's session QC criteria (z-drift, d-prime ≥ 1, etc.) but notes these were already applied by Allen before release, so no re-derivation was attempted. Step 5 decision 3 justifies dropping passive experiments. Step 10 Check 5 records "Minimum trials per session: 39 (above the 2-trial minimum)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. dF/F traces stored at `processing/ophys/dff/traces/data` (shape `(timepoints, ROIs)`), with their companion `timestamps` (ophys frame times), column-filtered by `valid_roi` from `processing/ophys/image_segmentation/cell_specimen_table`. Calcium `events` / `filtered_events` were deliberately not used.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]

cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]        # (timepoints, n_valid_neurons)
n_neurons = dff_valid.shape[1]
```

iii. CONVERSION_NOTES Step 1: "dF/F is pre-computed in NWB files — no need to compute from raw fluorescence"; Step 4/5 decision 2: "Use dF/F (not events): dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs." The AI noted the Piet et al. paper uses events but judged dF/F better suited to a time-resolved decoder.

## 2-b. How is the `neural` data processed?

i. The only processing is **averaging dF/F across the ophys frames falling inside each 750 ms stimulus bin**, implemented with a prefix-sum (cumsum) trick. No z-scoring, baseline subtraction, smoothing, neuron-level normalisation, or cross-plane merging is applied. Bins with zero ophys frames silently receive 0.0.

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0                 # 0.75 s
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')

dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. CONVERSION_NOTES Step 6: "Cumulative sum-based bin averaging (avoids per-bin boolean masking)"; Step 5 decision 1 justifies the bin itself. No further processing was applied because "dF/F is pre-computed in NWB files" with Allen's motion correction, neuropil correction and detrending already applied (Step 1/Step 3).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `valid_roi` boolean column of the `cell_specimen_table` is used to select neurons, matching the SDK default `exclude_invalid_rois=True`. Experiments with 0 valid ROIs are dropped. No SNR, event-rate, or activity-level filter is applied. (In this release the dF/F matrix already contains only valid ROIs — e.g. experiment 775614751 has 89 dF/F columns and 89/89 `valid_roi==True` — so the filter is a correct but no-op safeguard.)

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
n_neurons = dff_valid.shape[1]
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
```

iii. CONVERSION_NOTES Step 1: "`CellSpecimens.__init__` with `exclude_invalid_rois=True` | cell_specimens.py | CURATION | Filter to valid ROIs only"; Step 5 decision 4: "valid_roi filter: Use only cells marked valid_roi=True in NWB, matching SDK default behavior." Step 3 enumerates what `valid_roi` excludes (unions, duplicates >70 % overlap, edge ROIs, apical dendrites, ghost cells, etc.), i.e. the AI relies on Allen's published classifier rather than inventing its own QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is **stimulus-flash-locked, on the ophys timebase**. For every image presentation in the session, `np.searchsorted` on `ophys_timestamps` finds the frames in `[stim_start, stim_start + 0.75 s)`; those frames are averaged into one bin. A trial's neural matrix is the concatenation of the bins of that trial's presentations, so bin 0 of every trial begins at the trial's first flash onset (≈ `trial start_time`, verified: trial 1 of exp. 775614751 starts at 310.60 s, first flash at 310.62 s). `metadata['temporal_alignment_event']` is recorded as "Stimulus presentation onset (each 750ms image flash)", and `off_start` / `off_end` are set to `None`.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')
...
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
'off_start': None,
'off_end': None,
```

iii. Trajectory step 35: "I need to align everything to ophys timestamps as the time base. For each trial, I'll find which ophys timepoints fall within that window …". Step 41 / Step 5 decision 1: locking bins to flash onsets makes every time bin correspond to one image presentation so that "The task variables (image identity, change) naturally align to this". `off_start/off_end` were set to `None` because trials have variable length and the alignment event repeats within a trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — aggressive rebinning.** Native ophys sampling is ~30.94 Hz (CAM2P single-plane, 32.3 ms) or ~10.73 Hz (MESO mesoscope, 93.2 ms). The AI rebins every session to a common **750 ms** bin (one bin per stimulus presentation: 250 ms image + 500 ms grey), reducing a ~8 s trial from ~250 frames to ~11.7 bins. `metadata['time_bin_size'] = 750.0`.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_ends = all_stim_starts + bin_duration
...
'time_bin_size': TIME_BIN_MS,
```

iii. Trajectory step 41 and CONVERSION_NOTES Step 4/5: the target format demands "Time bins should be the same size for all trials and sessions", but the dataset mixes 10.73 Hz and 30.94 Hz rigs, so native binning would violate that requirement. 750 ms was chosen because it is "1. Consistent across all equipment types, 2. Aligned to the natural task structure, 3. Each time bin corresponds to one image presentation, 4. The task variables (image identity, change) naturally align to this, 5/6. Behavioral variables and neural activity can be averaged within each bin." Notably, the AI's own intermediate reasoning (step 41) considered this "too coarse for capturing the temporal dynamics the decoder needs — onset transients, sustained responses, offset effects all get averaged together" and briefly favoured an 11 Hz resample, but the final implementation uses 750 ms and CONVERSION_NOTES does not record that reservation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The `image_name` column of the image-flash presentations table (`intervals/<Natural_Images_…_presentations>`), one entry per flash, together with `omitted` (encoded as the literal image name `'omitted'`). The presentation table is located by `find_stim_key`, which skips `trials`, `spontaneous_*` and `natural_movie_*`.

ii.
```python
def find_stim_key(f):
    for key in f['intervals']:
        if key == 'trials' or key.startswith('spontaneous') or key.startswith('natural_movie'):
            continue
        return key
    return None
...
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
stim_omitted = stim['omitted'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. CONVERSION_NOTES Step 5 mapping table: "image_name → output[0], Categorical encoding (8 images), source stimulus_presentations, Forward-fill for omitted". Using the presentations table (rather than the trials table's `initial_image_name`/`change_image_name`) gives the actual image shown on each flash and lets omissions be handled explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Two steps: (1) flashes whose name is `'omitted'` are **forward-filled** from the previous flash (back-filled if the omission is the trial's first flash); (2) names are mapped to integers by a global, alphabetically-sorted image list built once for the whole dataset. The global list is collected from only the **first 10 experiments** (`collect_all_image_names`), with an unseen name silently falling back to code 0 via `.get(img, 0)`. In practice the 10 sampled experiments do contain all 16 images in the dataset (verified independently), giving the identical mapping to the reference: `im000→0 … im106→15`.

ii.
```python
def collect_all_image_names(active_exps, nwb_map):
    all_images = set()
    sample_exps = active_exps.head(min(10, len(active_exps)))
    for _, row in sample_exps.iterrows():
        ...
            for n in names:
                name = n.decode() if isinstance(n, bytes) else str(n)
                if name != 'omitted':
                    all_images.add(name)
    return sorted(all_images)
...
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 decision 5: "Omitted stimuli: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions." Step 6: "Sample 10 experiments for image name collection (not all 202)" is listed as a speed optimisation. Step 10 Check 4 reports "8 images per session … 16 total across sets A/B" and a ~6 % share per image, consistent with two 8-image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Perfectly index-aligned by construction: the image code array is indexed by `trial_stim_indices`, the exact same presentation indices used to build the neural bins, so `output[0][:, k]` and `neural[:, k]` describe the same 750 ms window. The code is stacked into row 0 of the `(5, n_bins)` output array.

ii.
```python
images = stim_image_name[trial_stim_indices].copy()
...
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ...
], axis=0)  # (5, n_bins)
```

iii. Not separately argued; it follows from Step 5 decision 1 (one bin = one stimulus presentation), which makes stimulus-derived outputs exactly co-indexed with the neural bins. The AI's Step 10 Check 2 spot-checks confirmed "Image ID Match: True" against the raw NWB for 3 sessions.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean column of the image-flash presentations table.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 1 identifies `is_change_event()` in `stimulus_processing.py` as the SDK function that computes this field, and Step 5 maps "is_change → output[1], Binary (0/1), source stimulus_presentations, 1 at change flash only". Using the pre-computed SDK field avoids re-deriving change events from image-name transitions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Only `NaN → 0` coercion and a cast to int. No smoothing, no window extension, no separate handling of catch trials (catch trials are *sham* changes with `is_change == False`, so they automatically receive all-zeros).

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. No explicit justification beyond the Step 5 mapping. The AI verified the consequence in Step 10 Check 4: "Go trials all have change: 45,477 with change, 0 without; Catch trials no change: 0 with change, 6,515 without", and overall 7.5 % of bins flagged as change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `{0 = no_change, 1 = change}` with value names `['no_change', 'change']`. Because a bin is one 750 ms flash+grey period, exactly one bin per go trial is 1 — i.e. the "right after a change" window is implicitly one image presentation wide (250 ms image + 500 ms grey). Measured distribution: 92.5 % / 7.5 %.

ii.
```python
change_value_names = ['no_change', 'change']
...
'output_names': ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome'],
```

iii. Step 5 mapping: "1 at change flash only". Step 3 records the task structure (250 ms stimulus + 500 ms grey = 750 ms interval) that makes one bin the natural "right after the change" unit.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same indexing as image identity — `stim_is_change[trial_stim_indices]`, so the change flag sits in exactly the bin whose dF/F frames span `[change_flash_onset, +750 ms)`. Stored as row 1 of the output array.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
output_arr = np.stack([ot['image_identity'], ot['image_change'], ...], axis=0)
```

iii. Follows from the one-bin-per-flash design; verified in Step 10 Check 2 ("Change Match: True" for sessions 0, 50, 150 against raw NWB) and by the AI's `--show-processing` plots of image identity and change per trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — i.e. the 10 Hz-lowpass Butterworth-**filtered** running speed in cm/s (not `speed_unfiltered`), the same stream the SDK's `RunningSpeed.from_nwb()` returns by default.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts    = f['processing']['running']['speed']['timestamps'][:]
```

iii. CONVERSION_NOTES Step 1: "Running speed has both raw and 10 Hz lowpass Butterworth filtered versions"; Step 3: "Running speed: 10 Hz lowpass Butterworth filter, z-score ≥ 10 transients set to NaN"; Step 10 Check 3 records "Filtered speed from NWB ≡ RunningSpeed.from_nwb() — Same source".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The ~60 Hz speed trace is **averaged within each 750 ms stimulus bin** (prefix-sum over the running samples in `[stim_start, stim_start+0.75)`), giving one mean speed per bin; bins with no samples get 0.0. The bin means (not the raw samples) are then accumulated across the whole dataset in pass 1 and used to define the percentile edges.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends   = np.searchsorted(running_ts, all_stim_ends,   side='left')
run_cumsum = np.concatenate([[0], np.cumsum(running_speed)])
...
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
running_values.extend(running_binned.tolist())
```

iii. Step 5 mapping: "running speed → output[2], Average in 750ms bins, discretize to 5 percentile bins". Step 6 lists the cumsum-based averaging as the efficiency optimisation. Averaging (rather than point-sampling/interpolating) is the consistent choice given a 750 ms bin.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile** bins. Edges are the 0/20/40/60/80/100th percentiles of *all* bin-averaged running values pooled across every retained trial of every session (computed in pass 1); the outermost edges are replaced by `±inf` so nothing falls outside, and assignment uses `np.digitize` on the 4 interior edges, clipped to `[0, 4]`. Value names `['speed_q1' … 'speed_q5']`. The realised distribution is exactly 20 % per bin.

ii.
```python
def discretize_values(values, n_bins, bin_edges=None):
    if bin_edges is None:
        valid = values[~np.isnan(values)]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, bin_edges
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
```

iii. Step 5 decision 7: "Discretization: Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins." Global (rather than per-session) edges keep the class definition comparable across sessions; the instruction itself requires "five equal percentile bins".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running bins use the *same* `all_stim_starts` grid and the *same* `trial_stim_indices` as the neural bins, only with `searchsorted` run against the running-encoder timestamps instead of the ophys timestamps. Both streams therefore describe identical wall-clock windows; running is row 2 of the output array.

ii.
```python
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts,   all_stim_starts, side='left')
run_bin_starts   = np.searchsorted(running_ts, all_stim_starts, side='left')
```

iii. CONVERSION_NOTES Step 3: "Temporal alignment: All data streams synchronized via NI PCI-6612 at 100 kHz", so the timestamps of the three streams are already in a common clock and can be windowed independently. Step 10 Check 2 records "Running Match: True" for the three spot-checked sessions.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (ellipse area) with its `timestamps`, plus `acquisition/EyeTracking/likely_blink/data` for blink rejection. `pupil_width` was **not** used; diameter is derived from the area. Three active experiments have no `EyeTracking` group at all.

ii.
```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
if has_eye_tracking:
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
    pupil_ts       = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
    likely_blink   = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. CONVERSION_NOTES Step 4: "Pupil tracking | area, width, height available | pupil_area with NaN for blinks | DeepLabCut, blink detection z>3 | Use area → compute diameter, interpolate NaNs". Trajectory step 35 shows the AI first inspected `pupil_tracking` (found it stores x,y centre) before locating the separate `area` dataset. Step 1 records `determine_likely_blinks()` (z-score 3.0, 2-frame dilation) as the SDK's blink QC.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pipeline: blink frames → NaN; non-positive areas → NaN; area converted to an equivalent-circle diameter `d = 2·sqrt(A/π)`; NaNs linearly interpolated over sample index; then averaged within each 750 ms stimulus bin (NaN-aware prefix sums, so a bin with no valid samples stays NaN).

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
pupil_area[pupil_area <= 0] = np.nan
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diameter = interpolate_nans(pupil_diameter)
...
pup_valid = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum = np.concatenate([[0], np.cumsum(pup_filled)])
pup_count_cumsum = np.concatenate([[0], np.cumsum(pup_valid.astype(np.float64))])
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
    if n_valid > 0:
        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Step 5 mapping: "pupil area → diameter | 2*sqrt(area/pi), interpolate NaN, avg in bins, 5 percentile bins | Interpolate blinks"; Step 5 decision 6: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging." Area→diameter conversion makes the variable a true diameter rather than a single ellipse axis.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same machinery as running speed: five equal-percentile bins from globally pooled, NaN-excluded bin means; `np.digitize` on the interior edges; value names `['pupil_q1' … 'pupil_q5']`. Residual per-trial NaNs are interpolated within the trial; a trial that is *entirely* NaN (i.e. an experiment with no eye tracking) is assigned the **middle bin (2)** for every time point. Realised distribution: 19.6 / 19.6 / 21.4 / 19.6 / 19.6 % — the bin-2 excess is exactly the 3 eye-tracking-less experiments.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
...
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Step 5 decision 7 (global 2-pass percentile edges) and decision 6 (NaN interpolation). The middle-bin fallback is the AI's "least-informative" default for unrecoverable missing pupil data; it is not separately discussed in CONVERSION_NOTES.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical construction to running speed: `searchsorted` of the same `all_stim_starts`/`+0.75 s` boundaries against the ~30 Hz eye-tracking timestamps, indexed by the same `trial_stim_indices`. Pupil occupies row 3 of the output array.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends   = np.searchsorted(pupil_ts, all_stim_ends,   side='left')
...
output_arr = np.stack([image_identity, image_change, running_disc, pupil_disc, outcome], axis=0)
```

iii. Same rationale as running speed — all streams carry hardware-synchronised timestamps (Step 3), so windowing each stream on the shared bin boundaries guarantees alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. Step 5 mapping: "hit/miss/fa/cr → output[4], Categorical (4 classes), source trials table, Static per trial". These are the canonical go/no-go change-detection outcome labels described in the whitepaper and reproduced in the SDK's `Trials` class (Step 1).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A priority cascade maps the booleans to integers `hit=0, miss=1, false_alarm=2, correct_rejection=3`, with a fallback of `1` (miss) if none is set. The scalar is then **broadcast across all time bins** of the trial so the output array stays `(5, n_bins)`. Measured distribution: hit 30.3 %, miss 57.2 %, FA 1.7 %, CR 10.8 %.

ii.
```python
if trial_hit[trial_idx]:
    outcome = 0  # Hit
elif trial_miss[trial_idx]:
    outcome = 1  # Miss
elif trial_fa[trial_idx]:
    outcome = 2  # False Alarm
elif trial_cr[trial_idx]:
    outcome = 3  # Correct Rejection
else:
    outcome = 1  # Default to Miss
...
np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
...
outcome_value_names = ['hit', 'miss', 'false_alarm', 'correct_rejection']
```

iii. The target format prefers time-varying outputs ("If at all possible, make it time-varying"), so a per-trial scalar is tiled across bins rather than stored as a length-1 vector. Step 10 Check 4 verifies "Trial outcomes | Hit+Miss for Go, FA+CR for Catch | Confirmed".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Blinks / bad pupil samples** → NaN, then linear interpolation over sample index; non-positive areas also NaN'd.
- **Experiment with no `EyeTracking` group** (3 of 202) → pupil is all-NaN → every bin assigned the middle percentile bin (2). This fabricates a label rather than dropping the session or flagging it.
- **Omitted stimulus flashes** → image identity forward-filled (back-filled at trial start); `is_change` NaN → 0.
- **Empty bins** (no ophys / running samples inside a 750 ms window) → silently left at `0.0` rather than NaN or interpolated.
- **Trials with no associated stimulus presentations** → skipped; **experiments with 0 valid ROIs or no image-presentation table** → skipped; **experiments with < 2 trials** → skipped.
- **Unknown image name** → silently coded 0 (`.get(img, 0)`); **unmatched outcome** → silently coded 1 (miss).
- There is **no `try/except`** around per-experiment processing, so a corrupt file would abort the whole run.

ii.
```python
def interpolate_nans(values):
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return values  # All NaN, can't interpolate
    if valid.sum() == len(values):
        return values  # No NaN
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result
...
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 decisions 5 and 6 cover omissions and pupil NaNs. Step 10 Check 5 ("Check for edge cases") asserts "No NaN/Inf in neural data; All output values are valid integers; No non-integer outputs found" and Step 10 concludes "No issues found. All checks passed." The missing-eye-tracking fallback and the silent `.get(img, 0)` / `outcome = 1` defaults are not discussed.

## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O plus decompression of the full-session HDF5 arrays inside `process_experiment` — `dff_data[:]` (e.g. 149,472 × 89 floats), `running_speed[:]` (~270 k samples), `pupil` (~136 k samples) — and the `np.cumsum(dff_valid, axis=0)` prefix-sum over the whole session. Because the design makes **two complete passes** over all 202 files, this cost is paid twice: Pass 1 = 261.6 s, Pass 2 = 246.6 s, total 510.4 s (metadata load 0.0 s, image-name collection 1.0 s, pickling 1.1 s).

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]
...
dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
print(f"  Pass 1 completed in {time.time()-t0:.1f}s")
print(f"\nPass 2 completed in {time.time()-t0:.1f}s")
```

iii. CONVERSION_NOTES Step 7 records the extrapolation from the 2-session sample (14.6 s → ~24 min) and the trajectory (step 61) shows the AI judged "25 minutes isn't unreasonable for this scale" and chose to keep the two-pass design after switching the bin-boundary search to `np.searchsorted`. Actual full run came in at 8.5 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three per-bin Python loops survive inside the per-trial loop and each could be a single fancy-indexed expression over the precomputed cumsum arrays:
- the neural bin loop `for bi, si in enumerate(trial_stim_indices)`,
- the running bin loop,
- the pupil bin loop.
Since `ophys_bin_starts`/`ophys_bin_ends` are already computed for *all* presentations up front, `neural_matrix` could be produced for the entire session at once as `(dff_cumsum[ends] - dff_cumsum[starts]) / counts[:, None]` and then sliced per trial. The `omitted` forward-fill is also an element-wise Python loop where `np.maximum.accumulate` on a validity index would do. The outer `for trial_idx in np.where(trial_mask)[0]` loop and the `for _, row in active_exps.iterrows()` driver could additionally be parallelised across files (the work is embarrassingly parallel and I/O-bound).

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
```

iii. The AI describes the cumsum approach as its vectorisation (Step 6: "Cumulative sum-based bin averaging (avoids per-bin boolean masking)", "np.searchsorted for efficient bin boundary finding"), and trajectory step 61 shows it replaced an earlier per-bin boolean search with `searchsorted`. It did not go the further step of removing the remaining Python-level bin loops, judging the ~24 min estimate acceptable.

## 9-c. What processing does the code repeat multiple times?

i. **The entire per-experiment pipeline runs twice.** `process_experiment(..., collect_stats_only=True)` in Pass 1 re-reads every NWB file and executes *all* of: valid-ROI filtering, `np.cumsum` over the full dF/F matrix, per-trial neural bin averaging, image forward-fill and code lookup, change-flag extraction, and outcome classification — the `if collect_stats_only: continue` guard sits *after* all of that work, so only the running/pupil lists survive. Additionally, `collect_all_image_names` opens 10 of those same NWB files a third time just to read `image_name`, and `discretize_values` is called once per trial (re-doing the `np.digitize`/clip setup) instead of once per session on a concatenated array.

ii.
```python
    for trial_idx in np.where(trial_mask)[0]:
        ...
        # --- Neural data: vectorized bin averaging ---
        neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for bi, si in enumerate(trial_stim_indices):
            ...
        # --- Image identity ---  (forward-fill, code lookup)
        # --- Image change ---
        # --- Running / pupil bin averaging ---
        # --- Trial outcome ---
        ...
        if collect_stats_only:
            continue          # <-- everything above is thrown away in Pass 1
```
```python
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        stats = process_experiment(nwb_path, row, collect_stats_only=True)   # Pass 1
    ...
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        result = process_experiment(nwb_path, row, collect_stats_only=False) # Pass 2
```

iii. The two-pass structure itself is justified in Step 5 decision 7 ("Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins") and Step 6 ("Pass 1: Collect running speed and pupil statistics …; Pass 2: Full conversion"). The docstring states "If collect_stats_only=True, returns only running/pupil values for percentile computation", but the code does not short-circuit the expensive neural work; CONVERSION_NOTES never notes the duplication, and Pass 1 (261.6 s) in fact costs slightly *more* than Pass 2 (246.6 s).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Discarded work:
- **All Pass-1 neural processing** (see 9-c): `np.cumsum` over the entire dF/F matrix and a `(n_neurons, n_bins)` average for every trial of every experiment — ~half the total 510 s runtime — is computed and thrown away.
- **Pass-1 image identity, change flags and trial outcomes** likewise computed and discarded.
- **`interpolate_nans` over the full-session pupil trace** runs in both passes, and is then followed by NaN-aware bin averaging whose `pup_count_cumsum` machinery is redundant because interpolation has already removed the NaNs.
- **`input_trials`**: a `np.zeros((0, n_bins))` array is allocated per trial even though the task specifies no decoder inputs and `input_names` is empty — the decoder never reads it.
- **`running_speed_raw` / `pupil_diameter_raw`** are stored per trial and dropped after discretisation.
- **`discretize_values` return value `binned`** is discarded at both call sites that only want `bin_edges`; the same function recomputes `bin_edges[1:-1]` per trial.
- `session_info` (202 dicts) is written into metadata but unused by the decoder; `plot_processing` builds a 6×2 figure per session but only under `--show-processing`.

ii.
```python
        if collect_stats_only:
            continue
...
    if collect_stats_only:
        return {
            'running_values': running_values,
            'pupil_values': pupil_values,
        }
...
input_trials.append(np.zeros((0, n_bins), dtype=np.float32))
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)   # `binned` discarded
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)  # edges discarded
```

iii. CONVERSION_NOTES Step 6 presents the script as optimised ("Cumulative sum-based bin averaging", "np.searchsorted", "Sample 10 experiments for image name collection (not all 202)") and Step 10 concludes "No issues found", so none of this discarded work is acknowledged. The empty `input` arrays are required by the target format spec (`'input'` must be present per session/trial) even though `input_names` is empty.
