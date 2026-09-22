# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK cache API. It reads the released NWB files directly with `h5py` from the local data directory, discovering every `behavior_ophys_experiment_*.nwb` file on disk (284 files) and joining them against `project_metadata/ophys_experiment_table.csv` by `ophys_experiment_id`. It then keeps only **active behavior** experiments (`passive == False`), giving **202 experiments from 38 mice**. No `project_code` filter is applied, so both `VisualBehavior` (168 active) and `VisualBehaviorMultiscope` (34 active) experiments are included. Each experiment file is opened twice over the run (once in a statistics pass, once in the conversion pass), and each open reads dF/F traces + timestamps, the cell specimen table, the stimulus-presentations interval table, the trials table, running speed, and eye tracking.

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
```python
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]      # (timepoints, neurons)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    ...
    trials = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][:]
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
```

iii. From CONVERSION_NOTES Step 1/Step 4 and the trajectory: the AI first mapped the SDK loaders (`BehaviorOphysExperiment.from_nwb_path`, `DFFTraces.from_nwb`, `Trials.from_nwb`, `Presentations.from_nwb`, `RunningSpeed.from_nwb`, `EyeTrackingTable.from_nwb`) onto their NWB paths, then read the same HDF5 fields directly "h5py direct NWB read ≡ BehaviorOphysExperiment.from_nwb_path() — Equivalent" (Step 10, Check 3). Passive experiments were dropped because "passive sessions lack the trial outcomes (Hit/Miss/FA/CR) that the task requires" (trajectory step 31; Step 4 table: "Include only active sessions (need trial outcomes)"). Mesoscope experiments were kept deliberately — the AI enumerated the 34 MESO vs 168 CAM2P split and chose a binning scheme (750 ms) that is "consistent across all equipment types" instead of dropping a rig.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained (active) experiments, cast to `str` and sorted; this list becomes `data['subjects']` (38 mice). Each session stores an index into this list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])   # result['mouse_id'] = str(exp_info['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. `mouse_id` in `ophys_experiment_table.csv` is the canonical animal identifier (CONVERSION_NOTES Step 2 counts 107 unique mice in the full metadata, 38 in the available NWB subset). The AI cross-checked the resulting count against the whitepaper (82 mice in the full release) and documented the difference as expected because only a 284-file subset is on disk (Step 4 / Step 9 consistency tables).

## 1-c. How are the data split into sessions?

i. **Each NWB file (i.e. each `ophys_experiment_id` = one imaging plane) is treated as one "session"** in the output. Experiments are sorted by `ophys_experiment_id` and iterated one at a time; planes belonging to the same `ophys_session_id` are *not* merged. This yields 202 output sessions from 174 distinct physical ophys sessions: mesoscope sessions contribute up to 7–8 identical-trial "sessions" each (visible in the verification log as repeated trial counts `209 ×7, 287 ×7, 309 ×7, …`, and as mouse 457841 having "34 sessions").

ii.
```python
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
    ...
    session_info.append({'experiment_id': eid, 'mouse_id': result['mouse_id'], ...})
```

iii. Trajectory step 31: "Each NWB file is one 'experiment' (one imaging plane) — treat each as a session for the decoder"; "Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session". The AI later noticed the consequence ("Subject 457841 has 34 sessions … MESO sessions with multiple imaging planes get split into separate experiments — so 5 actual MESO sessions could generate 34 'sessions'", trajectory step 79) but decided it did not need further action.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial's time span is realised as the set of **stimulus presentations whose `trials_id` equals that trial's `id`**; each presentation becomes one 750 ms time bin. (I verified independently that these presentations start at the trial's `start_time` and end at its `stop_time`, so the window is effectively `start_time → stop_time`, mean ≈ 7.9 s, 10–17 bins, mean 11.6 bins.) Trials with no associated presentations are dropped.

ii.
```python
trials = f['intervals']['trials']
trial_ids = trials['id'][:]
trial_go, trial_catch = trials['go'][:].astype(bool), trials['catch'][:].astype(bool)
trial_aborted, trial_auto = trials['aborted'][:].astype(bool), trials['auto_rewarded'][:].astype(bool)
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto

for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. CONVERSION_NOTES Step 5: the natural task unit is the stimulus flash, so a trial is "the stimulus presentations within that trial"; "Each bin = 250 ms image + 500 ms grey". The trials table's `trials_id` column on the presentations table is the SDK's own flash→trial assignment, so the AI reused it rather than re-deriving trial boundaries by searchsorted on times.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied: (a) keep only `go | catch` and drop `aborted` and `auto_rewarded` trials (per the task instructions); (b) drop trials with zero stimulus presentations; (c) drop whole experiments with zero valid ROIs or no stimulus-presentation table; (d) drop experiments yielding `< 2` trials (the format requires ≥ 2 trials/session). Passive experiments were already removed at load. No engagement/performance (d′) filtering, no trial-level behavioural QC beyond this. In the full run 0 sessions were skipped and the minimum session had 39 trials.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if len(trial_stim_indices) == 0:
    continue
...
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping"); return None
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping"); return None
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules — Per task instructions: Include Go and Catch trials, exclude Aborted and Auto-rewarded") and Step 5. Aborted trials have no change stimulus (the mouse licked early) and auto-rewarded trials are free-reward trials whose outcome is not behaviourally informative. The ≥ 2 trial rule comes straight from the target-format requirement "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the pre-computed **dF/F** trace, read from `processing/ophys/dff/traces/data` (shape `(timepoints, ROIs)`) with its companion `timestamps`. Detected calcium **events** (which the reference paper uses) were deliberately not used.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. CONVERSION_NOTES Step 1: "dF/F is pre-computed in NWB files — no need to compute from raw fluorescence"; Step 4 discrepancy table: "Neural data type … Paper uses events for analysis → Use dF/F – standard for decoding, pre-computed"; Step 5 key decision 2: "dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs." Step 3 also records the SDK dF/F recipe (noise estimation → 600 s median-filter baseline → (F−F0)/F0 → detrending) to confirm no further computation is needed.

## 2-b. How is the `neural` data processed?

i. Two operations only: (1) select valid ROIs (`valid_roi == True`); (2) **average dF/F over each 750 ms stimulus-presentation window** `[flash_onset, flash_onset + 0.75 s)`, producing an `(n_neurons, n_bins)` float32 matrix per trial. No z-scoring, normalisation, baseline subtraction, smoothing, or cross-plane merging is performed. Averaging is done with a pre-computed cumulative sum so each bin mean is an O(1) difference.

ii.
```python
dff_valid = dff_data[:, valid_roi]                 # (timepoints, n_valid_neurons)
...
bin_duration = TIME_BIN_MS / 1000.0                # 0.75 s
all_stim_ends = stim_start + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, stim_start, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends, side='left')

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

iii. CONVERSION_NOTES Step 5 key decision 1 and Step 6: averaging within one stimulus flash is "the natural task unit", "consistent across all equipment types (MESO/CAM2P), aligned to image presentations", and the cumsum formulation was introduced in Step 7 optimisation to avoid per-bin boolean masking. No normalisation was applied because "the dF/F traces are already processed by the Allen SDK pipeline".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered with the NWB `cell_specimen_table['valid_roi']` boolean, reproducing the SDK's default `exclude_invalid_rois=True`. Experiments with 0 valid neurons are skipped entirely. No additional SNR/activity/event-rate criteria are applied. (In practice the released NWB files already contain only valid ROIs, so this filter is a no-op — I confirmed `valid_roi.all() == True` in 12 randomly sampled files; it is nevertheless the correct guard.)

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

iii. CONVERSION_NOTES Step 1 lists `CellSpecimens.__init__ with exclude_invalid_rois=True` as the SDK's CURATION step, and Step 3 enumerates what the flag removes ("unions of cells, duplicates (>70 % overlap), edge ROIs, apical dendrites, too small/narrow/dim, ghost cells (crosstalk), negative/zero traces"). Step 5 key decision 4: "Use only cells marked valid_roi=True in NWB, matching SDK default behavior." Session-level QC (z-drift, d′ ≥ 1, etc.) is noted as already applied upstream by the Allen release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **stimulus-presentation onset**: bin *k* of a trial covers `[start_time of the k-th flash of that trial, +750 ms)`, and frames are selected on the **ophys timestamps** with `np.searchsorted` (as the instructions require: "Temporally align based on ophys timestamp"). Because the flashes carrying a trial's `trials_id` run from the trial's `start_time` to its `stop_time`, each trial's neural window is effectively `trial start → trial stop` (variable length, 10–17 bins). `metadata['temporal_alignment_event'] = 'Stimulus presentation onset (each 750ms image flash)'`, with `off_start = off_end = None` because trials are variable-length.

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')
...
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
'off_start': None,
'off_end': None,
```

iii. Trajectory step 41: "For each trial, I'll bin the data into 750 ms windows aligned with stimulus presentations. Within each window, I'll average the neural activity, record the image identity, and compute the mean running speed and pupil diameter." Every stream is indexed by `searchsorted` against its own native timestamps into the *same* stimulus-onset grid, so all streams share one alignment. The AI verified alignment visually in the `--show-processing` plots and numerically in Step 10, Check 2 (`np.allclose`, atol = 1e-5, on 3 sessions).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **750 ms per bin** (`TIME_BIN_MS = 750.0`), i.e. one bin per stimulus presentation (250 ms image + 500 ms grey). This is an explicit rebinning: native ophys frames (30.94 Hz on CAM2P rigs, 10.73 Hz on MESO.1) are averaged down, ≈ 23 frames per bin for single-plane and ≈ 8 frames per bin for mesoscope. All behavioural streams are averaged into the same bins, so every session/trial has an identical bin width and `metadata['time_bin_size'] = 750.0`.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
bin_duration = TIME_BIN_MS / 1000.0
all_stim_ends = all_stim_starts + bin_duration
...
'time_bin_size': TIME_BIN_MS,
```

iii. The trajectory (steps 35, 39, 41) contains an extended deliberation. The AI measured the two rig frame rates, noted that the target format demands "Time bins should be the same size for all trials and sessions" while its session set mixes 11 Hz and 31 Hz, and rejected (a) upsampling MESO to 30 Hz ("artificially upsample… interpolation artifacts") and (b) downsampling CAM2P to 11 Hz, in favour of 750 ms bins because they are "Consistent across all equipment types; Aligned to the natural task structure; Each time bin corresponds to one image presentation; The task variables (image identity, change) naturally align to this". It explicitly acknowledged the cost ("too coarse for capturing the temporal dynamics… onset transients, sustained responses, offset effects all get averaged together") and accepted it. CONVERSION_NOTES Step 5 key decision 1 records the final rationale.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the stimulus-presentations interval table's `image_name` column (the per-flash ground truth), selected for the flashes belonging to each trial via `trials_id`. The presentations table is located by name at runtime (any `intervals` key that is not `trials`, `spontaneous*`, or `natural_movie*`, e.g. `Natural_Images_Lum_Matched_set_TRAINING_2017_presentations`). The trials table's `initial_image_name` / `change_image_name` were **not** used.

ii.
```python
def find_stim_key(f):
    for key in f['intervals']:
        if key == 'trials' or key.startswith('spontaneous') or key.startswith('natural_movie'):
            continue
        return key
    return None
...
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
stim_trials_id  = stim['trials_id'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. CONVERSION_NOTES Step 5 variable mapping: `image_name → output[0]`, source `stimulus_presentations`. Using the presentations table gives the identity actually on the screen at every flash, including the flashes after the change, without assuming the image is constant before `change_time`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. (1) `'omitted'` flashes (the 5 % omission trials) are **forward-filled** with the preceding image (or back-filled if the trial begins with an omission); (2) names are mapped to a **global** integer code via a sorted list of all image names; the global set spans both image sets A and B, so there are **16 classes** (`im000 … im106`), and `output_values[0]` stores the names. The global name list is gathered from only the **first 10 experiments** (an optimisation), and any name not in the list falls back to code 0.

ii.
```python
# Forward-fill omitted presentations
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]; break

img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```
```python
def collect_all_image_names(active_exps, nwb_map):
    """Collect all unique image names across experiments (sample a few to be fast)."""
    all_images = set()
    sample_exps = active_exps.head(min(10, len(active_exps)))   # 10 experiments only
    ...
    return sorted(all_images)
```

iii. Trajectory step 41: the AI considered dropping omitted flashes, rejected that because it "still violates the uniformity requirement because I'd have gaps", and settled on "including all presentations and forward-filling the image identity for omissions, setting image change to 0". Trajectory step 73 justifies the global 16-class code: "the key question is whether to encode image identity per-session (0–7) or globally (0–15), and global encoding is correct since the decoder needs consistent output dimensions" — the decoder weights absent classes by their per-session frequency, so unused classes are harmless. The 10-experiment sampling was introduced purely for speed and was checked post-hoc ("From just 10 experiments, I found all 16 unique images").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Perfectly by construction: the image code for bin *k* is the `image_name` of the same stimulus presentation whose `[onset, onset+750 ms)` window defined bin *k* of the neural matrix. Identity is constant within a bin, so no interpolation or boundary rounding is needed. The row is stacked with the other outputs into an `(5, n_bins)` array with the same `n_bins` as `neural`.

ii.
```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)  # (5, n_bins)
```

iii. Both the neural bins and the image labels are indexed by the same `trial_stim_indices` array, so alignment is exact. The AI's Step 10 sanity check re-derived image identity for 3 sessions directly from the NWB files and confirmed a match; I independently reproduced the image sequence for session 0 / trial 5 (`im061 ×7 → im066 ×6`) from the raw NWB.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the presentations table's boolean `is_change` column for the flashes of the trial. Because `is_change` is computed by the SDK as "image differs from the previous flash", it is `True` only on genuine changes — i.e. on go trials — and `False` on catch (sham-change) trials. The trials table's `change_time`/`go` columns were not used for this output.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 1 lists `is_change_event()` in `stimulus_processing.py` as the SDK function that "Identifies change events in stimulus presentations", and Step 5 maps `is_change → output[1]`, "1 at change flash only". The AI verified the semantics against the trial table in Step 10 Check 4: "Go trials all have change: 45,477 with change, 0 without; Catch trials no change: 0 with change, 6,515 without."

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none: the boolean is cast to int64, and `NaN` (which occurs for omitted presentations) is mapped to 0. The result is 1 in exactly **one** 750 ms bin per go trial (the change flash + its following grey period) and 0 everywhere else, including for all catch trials. Overall 7.5 % of bins are 1 (45,477 of 606,328).

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
...
change_value_names = ['no_change', 'change']
```

iii. CONVERSION_NOTES Step 4/5: omitted flashes carry no change event, so `NaN → 0`. Step 10 Check 4 verifies the 87.5 % / 12.5 % go/catch split implied by the change counts ("the expected 7/8 and 1/8 split").

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is already binary. Two classes are declared, `['no_change', 'change']`, matching the instruction "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0."

ii.
```python
change_value_names = ['no_change', 'change']
...
'output_names': ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome'],
'output_values': [image_value_names, change_value_names, running_value_names, pupil_value_names, outcome_value_names],
```

iii. Direct from the Decoder Output specification; no discretisation decision was needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin grid as the neural data: the change flag sits in the bin whose window starts at the onset of the changed image, i.e. it covers the change flash and the immediately following grey period — "right after a change in image identity". Verified against the raw NWB for session 0 / trial 5: `[0 0 0 0 0 0 0 1 0 0 0 0 0]`, with the 1 at the flash where `im061 → im066`.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
# indexed by the same trial_stim_indices used for neural_matrix
```

iii. As for image identity, alignment is exact by construction because outputs and neural bins are built from the same list of stimulus presentations. The processing plots (`--show-processing`) plot the change trace against the dF/F heatmap on a common time axis to make this visually checkable.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` (data + timestamps) — the SDK's **filtered** running speed (10 Hz low-pass Butterworth, transients with z ≥ 10 already set to NaN upstream), not `speed_unfiltered`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts    = f['processing']['running']['speed']['timestamps'][:]
```

iii. CONVERSION_NOTES Step 1 lists `RunningSpeed.from_nwb()` as the loader, and Step 3 records the filtering the Allen pipeline already applies ("10 Hz lowpass Butterworth filter, z-score ≥ 10 transients set to NaN"). Step 10 Check 3 records "Running speed: Filtered speed from NWB ≡ RunningSpeed.from_nwb() — Same source".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The ~60 Hz speed trace is **averaged within each 750 ms stimulus bin** (again via a cumulative sum and `searchsorted` bin boundaries on the running timestamps), producing one mean speed per bin. Bins containing no running samples are left at 0.0. The binned values are then discretised (see 5-c).

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
```

iii. Averaging (rather than interpolating) is the natural resampling for a 60 Hz signal into 750 ms bins and keeps running speed on exactly the same grid as the neural data (trajectory step 41: "compute running speed and pupil area averages" per bin). CONVERSION_NOTES Step 5 maps "running speed → output[2]: Average in 750 ms bins, discretize to 5 percentile bins".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. **Five equal-percentile bins** with edges computed **globally over every binned value in the whole dataset** in a dedicated first pass (percentiles 0/20/40/60/80/100, outer edges replaced by ±inf), then applied with `np.digitize`. Full-run edges: `[-inf, 0.0041, 0.788, 15.82, 33.13, inf]` cm/s. The resulting distribution is exactly 20 % per class (verification log).

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
    return np.clip(binned, 0, n_bins - 1), bin_edges
...
_, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)   # pass 1, all sessions
...
running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)  # pass 2
```

iii. Required by the Decoder Task ("discretized into five equal percentile bins"). CONVERSION_NOTES Step 5 key decision 7: "Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins" — global rather than per-session edges keep the class definitions comparable across sessions and mice. The edges are stored in `metadata['running_speed_bin_edges']` for interpretability.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Identically to the neural data: the same stimulus-onset windows `[onset, onset+750 ms)` are used, with bin boundaries found by `searchsorted` on the running timestamps, so bin *k* of the running row covers the same wall-clock window as bin *k* of the neural matrix.

ii.
```python
all_stim_ends = all_stim_starts + bin_duration          # shared window definition
ophys_bin_starts = np.searchsorted(ophys_ts,    all_stim_starts, side='left')   # neural
run_bin_starts   = np.searchsorted(running_ts,  all_stim_starts, side='left')   # running
```

iii. All NWB streams share a hardware-synchronised clock (CONVERSION_NOTES Step 3: "All data streams synchronized via NI PCI-6612 at 100 kHz"), so indexing each stream's own timestamps into a common window grid yields exact alignment. The `--show-processing` plots display the binned speed on the same time axis as the neural heatmap.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/area` (the fitted pupil-ellipse area) together with `acquisition/EyeTracking/likely_blink/data`. Pupil **diameter** is then computed as the equivalent-circle diameter, `d = 2·sqrt(area/π)`. `pupil_width`/`height` (the ellipse axes) were available but not used. Experiments without an `EyeTracking` group (3 of 284 files) are handled by producing all-NaN pupil values.

ii.
```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
if has_eye_tracking:
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
    pupil_ts       = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
    likely_blink   = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
    pupil_area = pupil_area_raw.copy().astype(float)
    pupil_area[likely_blink] = np.nan
    pupil_area[pupil_area <= 0] = np.nan
    pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. CONVERSION_NOTES Step 1/3 record the SDK's blink detector (`determine_likely_blinks()`, z-score threshold 3.0 with 2-frame dilation) and Step 5 maps "pupil area → diameter: 2*sqrt(area/pi), interpolate NaN, avg in bins, 5 percentile bins". The trajectory (step 41) shows the AI first noted "the whitepaper specifies it as the major axis of the ellipse fit, so I should use the width field directly", inspected the width/height/area ranges, and then chose the area-equivalent diameter instead: "For simplicity, I'll just compute diameter as 2*sqrt(area/pi) which gives the diameter of a circle with the same area. This is the most common definition."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) blink frames and non-positive areas → NaN; (2) area → equivalent diameter; (3) **linear interpolation over remaining NaNs across the whole session** (`np.interp`, constant extrapolation at the edges); (4) averaging of the valid samples inside each 750 ms bin (NaN-aware cumsum with a parallel count cumsum); (5) a second, trial-level NaN interpolation if a trial still has gaps; (6) global 5-percentile discretisation. If a trial's pupil signal is entirely missing, the whole trial is assigned the **middle bin (2)**.

ii.
```python
pupil_diameter = interpolate_nans(pupil_diameter)          # session-level blink interpolation
...
pup_valid  = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum       = np.concatenate([[0], np.cumsum(pup_filled)])
pup_count_cumsum = np.concatenate([[0], np.cumsum(pup_valid.astype(np.float64))])
...
n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
if n_valid > 0:
    pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
...
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. CONVERSION_NOTES Step 5 key decision 6: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging." Blinks are short so linear interpolation across them is standard; the middle-bin default for completely missing pupil data is a neutral placeholder that keeps the trial (and its neural data, image labels and outcome) in the dataset rather than discarding it.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-percentile bins, using the **same global two-pass procedure** as running speed, with NaNs excluded from the percentile computation. Full-run edges: `[-inf, 73.88, 83.84, 92.87, 105.38, inf]` px. Realised distribution: `q1 0.196, q2 0.196, q3 0.214, q4 0.196, q5 0.196` — the ~1.8 % excess in q3 is the all-missing trials forced into the middle bin.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
else:
    pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
...
pupil_value_names = [f'pupil_q{i+1}' for i in range(N_PERCENTILE_BINS)]
'pupil_diameter_bin_edges': pupil_bin_edges.tolist(),
```

iii. Required by the Decoder Task; global edges chosen for the same reason as running speed (CONVERSION_NOTES Step 5 key decision 7). Pupil size differs substantially between animals/rigs, so global edges make the five classes a dataset-wide arousal scale rather than a per-session one.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 750 ms stimulus-onset windows as the neural data; bin boundaries are found with `searchsorted` on the ~30 Hz eye-tracking timestamps (≈ 22 samples per bin), so pupil bin *k* covers the same window as neural bin *k*.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends   = np.searchsorted(pupil_ts, all_stim_ends,   side='left')
...
pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Same rationale as running speed: all streams are hardware-synchronised, and every stream is projected onto one shared window grid rather than onto each other, so no cross-stream interpolation error is introduced.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that priority order. Classes are `['hit', 'miss', 'false_alarm', 'correct_rejection']` → codes 0–3. Full-run distribution: hit 30.7 %, miss 56.8 %, FA 1.8 %, CR 10.7 %.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
...
if   trial_hit[trial_idx]:  outcome = 0  # Hit
elif trial_miss[trial_idx]: outcome = 1  # Miss
elif trial_fa[trial_idx]:   outcome = 2  # False Alarm
elif trial_cr[trial_idx]:   outcome = 3  # Correct Rejection
else:                       outcome = 1  # Default to Miss
```

iii. These are the SDK's canonical change-detection outcome labels (CONVERSION_NOTES Step 1/Step 4 data-flow listing of the trials table). Because aborted and auto-rewarded trials were already removed, go trials can only be hit/miss and catch trials only FA/CR — the AI verified this ("Trial outcomes: Hit+Miss for Go, FA+CR for Catch — Confirmed", Step 10 Check 4).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is **broadcast across every time bin of the trial** so that the output array is a uniform `(5, n_bins)` matrix (the variable is conceptually static per trial; the decoder reads it as a constant row). No other processing.

ii.
```python
output_arr = np.stack([
    ...,
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)  # (5, n_bins)
...
outcome_value_names = ['hit', 'miss', 'false_alarm', 'correct_rejection']
```

iii. The target format permits `(n_output, n_timepoints)` or `(n_output,)`, but mixing shapes within one output matrix is not possible, so the per-trial label is replicated to match the time-varying rows. The AI noted the statistical consequence in Step 12: "Trial outcome is static per trial (repeated across time bins), so there are effectively fewer independent samples", which it used to explain the larger train/validation gap for this output (0.483 vs 0.316).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing eye tracking group** (3/284 files) → pupil row is all-NaN → whole trial assigned the middle pupil bin.
- **Blinks / non-positive pupil area** → NaN → linearly interpolated (session level, then trial level).
- **Omitted stimulus flashes** → image identity forward-filled (back-filled at trial start); `is_change = NaN → 0`.
- **Experiments with no valid ROIs or no stimulus-presentation table** → experiment skipped with a warning.
- **Trials with no stimulus presentations** → trial skipped.
- **Sessions with < 2 usable trials** → session skipped.
- **Empty bins** (no ophys/running samples in a 750 ms window) → value left at 0.0 (silently), rather than NaN.
- **Unknown image name** (not in the 10-experiment sample) → silently coded 0.
- There is **no** try/except around per-experiment processing, so an unreadable file would abort the whole run.

ii.
```python
def interpolate_nans(values):
    valid = ~np.isnan(values)
    if valid.sum() == 0:  return values
    if valid.sum() == len(values): return values
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result
```
```python
if n_neurons == 0: ... return None
if stim_key is None: ... return None
if len(trial_stim_indices) == 0: continue
if result is None or result['n_trials'] < 2: skipped += 1; continue
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 key decisions 5 and 6 and Step 10 Check 5 ("Check for edge cases": minimum trials per session 39, "No NaN/Inf in neural data", "All output values are valid integers"). The AI's stated principle is to keep trials whenever the neural data are intact and substitute a neutral value for a missing behavioural covariate, rather than discarding neural data. The verification run reports no errors and no warnings, and 0 sessions were skipped in the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Reading each NWB file — above all the full-session dF/F matrix (`(≈140 000, n_neurons)`, up to 666 neurons) — plus the full-session `np.cumsum` over that matrix. This work is performed **twice per experiment**, once in the statistics pass and once in the conversion pass. Measured on the full run: Pass 1 = 261.6 s, Pass 2 = 246.6 s, pickling = 1.1 s, total = 510.4 s (~8.5 min). Timing is printed per pass and as a running ETA, as the instructions required.

ii.
```python
t0 = time.time()
for idx, (_, row) in enumerate(active_exps.iterrows()):
    stats = process_experiment(nwb_path, row, collect_stats_only=True)   # Pass 1: reads full dF/F
...
print(f"  Pass 1 completed in {time.time()-t0:.1f}s")
...
    result = process_experiment(nwb_path, row, collect_stats_only=False) # Pass 2: reads it again
...
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        remaining = (len(active_exps) - idx - 1) / rate
        print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
```

iii. CONVERSION_NOTES Step 7 projected ~24 min from the 2-session sample, above the 15-minute guideline, prompting the Step 7 optimisation round (cumsum + searchsorted binning, sampled image-name collection) that brought the full run down to 8.5 min. The AI identified "the per-bin loop that searches for ophys frames within each 750 ms window" as the bottleneck it could fix (trajectory step 61).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops that are straightforwardly vectorisable:
- the three per-bin loops (`neural`, `running`, `pupil`) — with the cumsum arrays already built, all bins of a trial can be computed as a single fancy-indexed difference, e.g. `(dff_cumsum[e_arr] - dff_cumsum[s_arr]).T / (e_arr - s_arr)`;
- the omitted-flash forward-fill loop — replaceable by `np.maximum.accumulate` over the indices of non-omitted flashes;
- the outer per-trial loop itself — trial boundaries are contiguous runs of `stim_trials_id`, so all trials of a session could be segmented with one `np.split`;
- `discretize_values` is called once per trial per variable; it could be applied once per session (or once for the whole dataset).

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
...
for bi, si in enumerate(trial_stim_indices):     # running
    s, e = run_bin_starts[si], run_bin_ends[si]
...
for bi, si in enumerate(trial_stim_indices):     # pupil
    s, e = pup_bin_starts[si], pup_bin_ends[si]
...
for bi in range(len(images)):                    # omitted forward-fill
    if images[bi] == 'omitted':
        ...
```

iii. The AI did perform one round of vectorisation (Step 6/7 notes: "Cumulative sum-based bin averaging (avoids per-bin boolean masking)", "np.searchsorted for efficient bin boundary finding"), replacing the original `(ophys_ts >= t_start) & (ophys_ts < t_end)` boolean-mask scan of the whole session per bin, and also replaced `IMAGE_NAMES_GLOBAL.index(img)` (O(n) list scan) with a dict lookup. It stopped once the projected runtime fell inside the 15-minute budget and did not revisit the residual loops.

## 9-c. What processing does the code repeat multiple times?

i. The **entire per-experiment pipeline is executed twice**. `process_experiment(..., collect_stats_only=True)` in Pass 1 and `process_experiment(..., collect_stats_only=False)` in Pass 2 are the *same function*: both re-open the NWB file, re-read dF/F, timestamps, cell table, stimulus table, trials table, running and eye data, rebuild all cumulative sums and searchsorted bin boundaries, and re-loop over every trial. Only the last few lines differ. This duplication accounts for ~260 s of the 510 s run. Smaller repetitions: `find_stim_key` and the image-name scan re-open files already opened elsewhere; `discretize_values` recomputes the digitize call per trial; `interpolate_nans` runs at both session and trial level on the pupil signal.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2 — same function, same file, same work
result = process_experiment(nwb_path, row, collect_stats_only=False)
```
```python
if collect_stats_only:
    return {'running_values': running_values, 'pupil_values': pupil_values}
```

iii. The two-pass design is documented in CONVERSION_NOTES Step 6 ("1. Pass 1: Collect running speed and pupil statistics for percentile bin computation; 2. Pass 2: Full conversion with discretization using global percentile bins"), and is genuinely needed to get *global* percentile edges before writing the discretised outputs. What is not justified is re-reading the files: the binned running/pupil values produced in Pass 2 could simply have been buffered in memory (which is what the human reference does — it stores per-trial raw values, computes edges from them, then discretises) so the data would be touched only once.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In **Pass 1** (`collect_stats_only=True`) the function still does all of the following before returning only the running/pupil lists: loads the full dF/F matrix from disk, applies the `valid_roi` selection, builds the full `(T+1, n_neurons)` cumulative sum, and, for every trial, allocates and fills `neural_matrix`, runs the omitted-image forward-fill, builds `image_indices` and `change_flags`, and computes the trial outcome — all thrown away at `if collect_stats_only: continue`. That is the single largest piece of wasted work in the script.
Also wasted: the cumulative sums span the whole session (~140 000 frames) although only the in-trial frames are ever read; the `valid_roi` filter is a no-op on released NWB files (all ROIs are already valid); per-trial `input_trials.append(np.zeros((0, n_bins)))` allocates empty arrays for an empty input space; `pup_count_cumsum` tracks NaNs that have already been interpolated away; and the `--show-processing` figures are rendered during the conversion run.

ii.
```python
for trial_idx in np.where(trial_mask)[0]:
    ...
    neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)      # computed in BOTH passes
    for bi, si in enumerate(trial_stim_indices):
        ...
    images = stim_image_name[trial_stim_indices].copy()                  # forward-fill in BOTH passes
    ...
    image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
    change_flags  = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
    ...
    if collect_stats_only:
        continue                                                          # <-- everything above discarded
```
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # loaded even when only stats are needed
dff_cumsum = np.cumsum(dff_valid, axis=0)                         # built even when only stats are needed
```

iii. The AI does not document this waste; CONVERSION_NOTES Step 6 lists only the optimisations it did make, and Step 9 records the accepted 8.5-minute runtime. The `collect_stats_only` flag was clearly intended as a cheap statistics mode (its docstring says "returns only running/pupil values for percentile computation"), but it was implemented as an early-exit inside the full pipeline rather than as a separate lightweight path, so the intended saving was never realised.
