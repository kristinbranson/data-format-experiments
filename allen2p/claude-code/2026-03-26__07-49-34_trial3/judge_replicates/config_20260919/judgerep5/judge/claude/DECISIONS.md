# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK API. It reads the local release tree directly:
`data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv` supplies the metadata
(mouse_id, ophys_session_id, session_type, targeted_structure, project_code, …), and each experiment's
data is read straight out of its NWB/HDF5 file with `h5py`
(`behavior_ophys_experiments/behavior_ophys_experiment_<eid>.nwb`).
The experiment list is the intersection of (a) the NWB files actually present on disk and (b) rows of
the experiment table whose `session_type` is one of the four *active* behavior sessions
(`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`). No `project_code`
filter is applied, so both `VisualBehavior` (single-plane, 168 experiments) and
`VisualBehaviorMultiscope` (multi-plane, 34 experiments) are included → 202 experiments, 38 mice.
From each NWB file it pulls: the cell-specimen table (`valid_roi`), the dF/F timestamps (used as the
ophys clock), the `event_detection` traces, the `trials` table, the natural-image
`stimulus_presentations` table, running speed, and eye tracking. Conversion runs in three passes over
the files (image names → running/pupil percentile pools → full processing).

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR  = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]

def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        eid = int(f.stem.split('_')[-1])
        nwb_ids.add(eid)
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()
```
```python
with h5py.File(nwb_path, 'r') as f:
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    valid_roi  = cell_table['valid_roi'][()].astype(bool)
    ophys_ts   = f['processing']['ophys']['dff']['traces']['timestamps'][()]
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    events_valid = events_data[:, valid_roi]
    trials_grp = f['intervals']['trials']
    ...
    running_speed = f['processing']['running']['speed']['data'][()]
    running_ts    = f['processing']['running']['speed']['timestamps'][()]
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. From CONVERSION_NOTES.md Step 1/2: "dF/F is PRE-COMPUTED in NWB files", "Events are PRE-COMPUTED
via FastLZeroSpikeInference", and the SDK loaders (`CellSpecimens.from_nwb`, `Trials.from_nwb`,
`Presentations.from_nwb`, …) simply read these HDF5 fields. The AI therefore concluded that reading the
NWB files with `h5py` gives exactly the same arrays as the SDK while being much faster, and it verified
this in Step 10 Check 3 ("h5py reads same data as AllenSDK NWB reader") and Check 2 (independent
re-load of an NWB file, `np.allclose` on the neural data → PASS). Passive sessions were excluded
because "OPHYS_2, OPHYS_5 are passive viewing (no lick spout, satiated mice). No meaningful trial
outcomes."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiment table, sorted and cast to `str`.
Each output session's `subject_idx` is the index of its experiment's `mouse_id`. Result: 38 mice.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
...
'subjects': [str(s) for s in subjects],
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5, decision 12: "Subject IDs: Use mouse_id from experiment table." The AI
cross-checked the count against the papers (82 mice in the full release vs. 38 on disk) and recorded
the difference as expected for a partial download (Step 4 discrepancy table).

## 1-c. How are the data split into sessions?

i. **One NWB experiment (i.e. one imaging plane) = one output "session"**. `ophys_session_id` is never
used for grouping. For the 168 single-plane `VisualBehavior` experiments this is a 1:1 mapping, but the
34 `VisualBehaviorMultiscope` experiments come from only 6 real recording sessions of a single mouse
(457841), so those sessions are emitted 3–7 times each, once per plane, with the same trials and the
same behavioural outputs but disjoint neuron sets. Final output: 202 "sessions" (174 real ophys
sessions), and mouse 457841 appears with 34 sessions while every other mouse has 2–9.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    raw_data = load_experiment_data(nwb_path, eid)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. CONVERSION_NOTES.md Step 5, decision 13: "Each NWB experiment = one 'session' in output format
(one imaging plane with its own neurons)." The trajectory (step 31) makes the reasoning explicit:
"Each NWB file represents one imaging plane (experiment), and for multiscope setups multiple
experiments share a session. Since different planes have different neurons, I should treat each
experiment as a separate session in the decoder output format." Decision 2 adds the session-type
filter: passive sessions are dropped because there is no lick spout and therefore no meaningful
Go/Catch outcomes.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if it is a Go **or** Catch trial
and is neither aborted nor auto-rewarded. The trial window is the full `start_time` → `stop_time`
interval (variable length; 210–377 bins at 30 Hz, mean ≈ 254 bins ≈ 8.5 s), taken as all samples of the
regular 30 Hz grid falling in `[start_time, stop_time)`. 51,992 trials in total.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop  = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
    trial_ts = regular_ts[trial_time_indices]
    n_tp = len(trial_ts)
    neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5, decision 5: "Trial window: Use trial start_time to stop_time from
trials table. Variable length across trials." Decision 3: "Exclude aborted and auto-rewarded trials:
Per task instructions." In the trajectory the AI first considered a fixed window around `change_time`,
then rejected it because the format allows variable trial lengths as long as the bin size is constant,
and the full trial window preserves both the pre-change flashes and the post-change response window
(needed for time-varying image identity / image change). It verified the resulting Go/Catch split
(87.5 %/12.5 %) against the whitepaper.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (1) session level — only active session types (passive OPHYS_2/5 dropped)
and only experiments whose NWB file exists; (2) experiment level — skipped if `valid_roi.sum() == 0`,
if no natural-image stimulus table is found, if the file fails to load, if fewer than 2 trials pass the
trial filter, or if fewer than 2 trials survive processing; (3) trial level — aborted and
auto-rewarded excluded, only Go|Catch kept, trials with fewer than 3 time bins dropped, and trials that
match none of hit/miss/false_alarm/correct_reject dropped. No neuron- or session-level d-prime /
engagement filtering is applied.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) < 2:
    print(f"  WARNING: Only {len(valid_indices)} valid trials in experiment ..., skipping")
    return None
...
    if len(trial_time_indices) < 3:
        continue
...
    if trials['hit'][trial_idx]:      outcome = 0
    elif trials['miss'][trial_idx]:   outcome = 1
    elif trials['false_alarm'][trial_idx]: outcome = 2
    elif trials['correct_reject'][trial_idx]: outcome = 3
    else:
        continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials in experiment ..., skipping")
    return None
```

iii. CONVERSION_NOTES.md Step 3 records the whitepaper's curation rules ("Aborted trials: premature
lick before change → excluded from performance calculations"; "Auto-rewarded: 5 free rewards at session
start + after 10 consecutive misses") and Step 5 decision 3 adopts exactly the instruction's rule. The
≥2-trial requirement is driven by the target-format requirement "There needs to be at least two trials
within each session in order to evaluate the decoder performance". Step 10 Check 5 confirms "All
sessions have >= 2 trials (min: 39)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The **deconvolved calcium events**, `processing/ophys/event_detection/data`
(FastLZeroSpikeInference output, shape `(n_timepoints, n_cells)`), restricted to columns with
`valid_roi == True`. dF/F is *not* used for the neural output — only its `timestamps` field is used as
the ophys clock. 29,444 neurons total, 4–666 per session.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]   # (n_timepoints, n_valid_cells)
```

iii. CONVERSION_NOTES.md Step 5, decision 1: "**Neural signal: events (not dF/F)**: Paper says 'We
performed our analyses on discrete calcium events.' Use raw events from event_detection." The
trajectory (step 35) shows the AI explicitly weighing dF/F against events and choosing events because
"the reference paper explicitly uses deconvolved calcium events detected via FastLZeroSpikeInference,
which are much closer to spike-like activity than raw dF/F. The instructions emphasize matching the
reference processing". It also chose the *raw* events rather than the SDK's causally half-Gaussian
filtered `events.filtered_events`, again citing the paper.

## 2-b. How is the `neural` data processed?

i. Three operations: (1) keep only `valid_roi` columns; (2) linearly interpolate every neuron's event
trace from the native ophys timestamps onto a regular 30 Hz grid spanning the session
(`np.arange(ophys_ts[0], ophys_ts[-1], 1/30)`), done with a Python loop over neurons; (3) clip to
`>= 0`. No normalisation, no smoothing, no z-scoring, no merging of planes from the same session. The
per-trial matrix is the transposed slice `(n_neurons, n_timepoints)`, float32.

ii.
```python
def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    if data.ndim == 1:
        return np.interp(target_timestamps, timestamps, data)
    else:
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
        return result
```
```python
t_start = ophys_ts[0]; t_end = ophys_ts[-1]
dt = 1.0 / target_rate            # target_rate = 30.0
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5, decision 4: "Resample to 30 Hz: Paper interpolates to 30 Hz. Needed
for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms",
quoting the paper's methods: "linearly interpolating onto a consistent set of 30hz timestamps relative
to the triggering behavioral event". Step 6 notes "Events are clipped to >=0 after interpolation".
Step 10 Check 1 defends the resulting 2,602 all-zero-trial warnings as intrinsic to sparse calcium
events (~0.25 % of samples non-zero), not a bug.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the SDK's ROI-validity flag: neurons with `valid_roi == False` are dropped, and an experiment
with zero valid ROIs is skipped. No activity-, SNR-, or event-rate-based neuron filtering, and no
removal of the very small populations (some sessions retain only 4–6 neurons).

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. CONVERSION_NOTES.md Step 1: "Cell filtering: Only automatic filter is `valid_roi` boolean (SVM
binary classifier output). `exclude_invalid_rois=True` by default." Step 5 decision 10: "valid_roi
filtering: Only include neurons with valid_roi=True." Step 3 lists the whitepaper's ROI-exclusion
reasons (union of cells, duplicate, edge/motion affected, apical dendrite, too small/narrow/dim). In
the released NWB files on disk every ROI already has `valid_roi == True`, so in practice the filter is
a no-op and the neuron count (29,444) equals the raw ROI count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything lives on one clock — the ophys timestamps (taken from the dF/F trace timestamps, which
are identical to the event-detection timestamps). A single regular 30 Hz grid is built per experiment,
anchored at the first ophys timestamp; neural, running and pupil signals are all interpolated onto that
grid, and a trial is the set of grid samples in `[trial.start_time, trial.stop_time)`. So the
alignment event is **trial start** (`metadata['temporal_alignment_event'] = 'Trial start time (first
stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` because trials have variable length).
Neural and all output streams share the same index vector, so they cannot drift relative to each other.

ii.
```python
regular_ts = np.arange(ophys_ts[0], ophys_ts[-1], 1.0 / target_rate)
events_resampled  = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
pupil_resampled   = np.interp(regular_ts, pupil_ts, pupil_area)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial   = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial  = running_resampled[trial_time_indices]
pupil_trial    = pupil_resampled[trial_time_indices]
image_idx      = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal  = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```
```python
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The instructions say "Temporally align based on ophys timestamp", and CONVERSION_NOTES.md Step 5
decision 6 records "Alignment event: Trial start time (stimulus onset). off_start=0, off_end=None
(variable)". The trajectory shows the AI considered aligning to `change_time` with a fixed ±750 ms
window, but rejected it because `off_start`/`off_end` must be single values and the full trial window
is needed for time-varying image identity/image-change outputs. Step 10 Check 5 verified
"Neural/output length alignment at trial boundaries", and the `--show-processing` plots overlay neural
and output traces per trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 33.33 ms (30 Hz) for every trial and session; `metadata['time_bin_size'] = 1000/30`. Yes — explicit
rebinning by **linear interpolation** onto a regular 30 Hz grid. This slightly *down*-samples the
single-plane Scientifica data (native ≈ 30.94 Hz) and nearly *triples* the Multiscope data (native
≈ 10.7 Hz). No boxcar/bin-summation is used, so event amplitude is not mass-preserving.

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
regular_ts = np.arange(t_start, t_end, dt)          # dt = 1/30
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. Two justifications given (Step 5 decision 4 and the trajectory): the paper's own method —
"linearly interpolating onto a consistent set of 30hz timestamps" — and the target-format requirement
that "Time bins should be the same size for all trials and sessions", which cannot be satisfied while
mixing 31 Hz Scientifica and 11 Hz Multiscope recordings at their native rates. The AI acknowledged in
the trajectory that "the interpolation won't add real information" for the 11 Hz data but accepted it
for uniformity.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The natural-image `stimulus_presentations` interval table inside the NWB file: `start_time`,
`image_name` and `omitted`. Omitted flashes are removed from the lookup, so their 750 ms slot inherits
the preceding image. The trials table's `initial_image_name`/`change_image_name` are loaded but *not*
used for this output.

ii.
```python
stim_key = None
for k in f['intervals'].keys():
    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
        stim_key = k
        break
stim = f['intervals'][stim_key]
stim_data = {'start_time': stim['start_time'][()], 'stop_time': stim['stop_time'][()],
             'image_name': stim['image_name'][()], 'is_change': stim['is_change'][()].astype(bool),
             'omitted': stim['omitted'][()]}
```
```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names  = stim_data['image_name'][non_omitted]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`stimulus_presentations.image_name` → output[0]:
image_identity — Map to categorical int, time-varying per ophys frame — 8 natural images". Decisions 7
and 8: "Image identity during gray screen: Use the identity of the image that was just shown (last
presented image)" and "Image identity for omitted flashes: Continue with previous image identity". The
trajectory reasons that the 500 ms grey inter-stimulus interval should carry the most recent image
forward "rather than treating it as undefined or null".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per trial time bin, `searchsorted` finds the most recent non-omitted flash onset and its image name
is looked up. Names are first mapped to a per-experiment sorted index, then remapped to a **global**
sorted index over all image names seen in the dataset (16 codes: 8 for image set A + 8 for image set
B). Bins before the first flash get code 0. The result is an int64 row in the output matrix, and
`output_values[0]` holds the 16 image names.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    n_tp = len(timepoints)
    image_idx = np.zeros(n_tp, dtype=np.int64)
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names  = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0  # Before first stimulus
    return image_idx
```
```python
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    local_to_global[local_idx] = global_image_names.index(name) if name in global_image_names else 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']],
                         dtype=np.int64)
```

iii. A global mapping was used so that image codes are comparable across sessions and image sets
(Pass 1 of the conversion exists solely to collect `global_image_names`). CONVERSION_NOTES.md Step 9
records the outcome as "Image identity classes: 16 (8 per image set, 2 image sets)", and the planned
sanity check "Image identity has 8 unique values per session" is confirmed by the per-session output
ranges in `verification_full_out.txt` ([5,14] for image set A sessions, [0,15] for image set B).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the same 30 Hz grid timestamps (`trial_ts`) used to slice the neural
matrix, so the image row has the same length as the trial's neural matrix by construction. The switch
from the pre-change to the post-change image happens at the first grid bin at or after the change
flash's onset. (I independently re-derived image identity for 60 trials of experiment 1007107386 using
the reference's `initial_image_name`/`change_image_name`/`change_time` rule and got **0 mismatches**
out of 15,374 bins.)

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx    = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. Sharing a single grid and a single index vector is the AI's stated guarantee against
misalignment ("Temporal alignment: 30 Hz interpolation matches paper", Step 10 Check 3; "Neural/output
length alignment at trial boundaries", Step 10 Check 5). Step 10 Check 2 spot-checked image identity
directly from the NWB file: "Image identity (session 0, trial 5, timepoint 10): expected 'im063', got
'im063'. PASS".

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` and `omitted` columns plus `start_time` of the natural-image
`stimulus_presentations` table. Because `is_change` is only set for genuine image changes, catch
(sham-change) trials automatically get an all-zero row. The trials table's `change_time`/`go` columns
are loaded but not used here.

ii.
```python
change_mask   = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops  = stim_data['stop_time'][change_mask]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`trials.is_change` + stimulus timing → output[1]:
image_change — Binary 1 at change timepoint, 0 otherwise, time-varying — 1 for one 750 ms window at
change". Step 12 verified the catch-trial consequence explicitly: "Catch trials: 0/6,515 have
image_change signal (correct: catch = sham change); Go trials: 0/45,477 missing image_change signal".

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector the length of the trial is set to 1 for every grid bin in `[change_onset,
change_onset + 0.75 s)`, i.e. the changed image's 250 ms flash plus the following 500 ms grey period.
The loop runs over all change flashes of the session, but only the trial's own change falls inside the
trial window. Across the dataset 7.7 % of bins are labelled `change`, matching the expected ~1/13 of
flashes.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops  = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. CONVERSION_NOTES.md Step 3 records the stimulus structure from the paper/whitepaper ("Image
presentation: 250 ms stimulus + 500 ms gray = 750 ms"), and Step 6 states "Image change signal: 1
during 750 ms window starting at change onset". Step 10 Check 4 uses this as a consistency check:
"Image change fraction: 7.7 % — matches expected 1/13 flashes."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — it is already binary. Two categories are declared,
`['no_change', 'change']`, with values 0/1 held in an int64 row. Observed distribution:
no_change 0.923 / change 0.077.

ii.
```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
output_values = [
    global_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
]
```

iii. The Decoder Output spec calls image change a "binary variable. Have value of 1 right after a
change in image identity, otherwise 0. Time-varying", which the AI implemented literally; the only
free parameter is the width of the "right after" window, set to one full 750 ms image interval to match
the paper's image-interval analysis window.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: computed on `trial_ts`, the very grid samples used to slice the
neural matrix, so the rows are index-for-index aligned. (I re-derived image change for all 365 valid
trials of experiment 1007107386 using the reference's `change_time`+`go` rule: **0 mismatches** out of
93,359 bins.)

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. Same justification as 2-d/3-c: one clock, one grid, one index vector. The `--show-processing`
plots were produced specifically to show "no temporal misalignments" (Step 7).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` (cm/s, ~60 Hz) with its `timestamps`, i.e. the same array the SDK
exposes as `dataset.running_speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts    = f['processing']['running']['speed']['timestamps'][()]
```

iii. CONVERSION_NOTES.md Step 1 lists `RunningSpeed.from_nwb()` as the SDK loader for "running speed
(cm/s)"; Step 5's mapping table maps `running/speed/data` → `output[2]: running_speed`. The instruction
list names running speed as a discretized, time-varying decoder output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Linear interpolation from the ~60 Hz encoder timestamps to the experiment's 30 Hz grid; (2)
discretization into 5 bins whose edges are the 0/20/40/60/80/100th percentiles of the **pooled raw
running samples of every included experiment** (collected in Pass 2, before any trial segmentation);
(3) degenerate edges are nudged apart by 1e-10 and NaNs are mapped to bin 0. Note the pool is built
from full-session raw samples (including inter-trial and aborted-trial periods) rather than from the
retained trial data, and Multiscope sessions contribute their behaviour once per plane. Observed bin
occupancy 0.199/0.203/0.184/0.209/0.205.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        all_running_values.append(running.astype(np.float32))
all_running_cat  = np.concatenate(all_running_values)
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges

def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The Decoder Output spec requires "Running speed, discretized into five equal percentile bins.
Time-varying." CONVERSION_NOTES.md Step 5 records "Interpolate to 30 Hz, discretize into 5 percentile
bins, time-varying", and the planned sanity check "Running speed and pupil bins are roughly equal-sized
(by definition of percentile bins)" was confirmed in Step 9 ("Running speed bins 18–21 % each").
Computing the edges globally (rather than per session) keeps the class definition identical across
sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five quintile bins with globally-computed edges, labelled `bin_0` … `bin_4`; `np.digitize` on the
interior edges, clipped to [0, 4], NaN → bin 0. Per-session ranges in `verification_full_out.txt` show
that some (low-locomotion) sessions only occupy bins 0–2, which is the expected consequence of global
rather than per-session edges.

ii.
```python
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
...
[f'bin_{i}' for i in range(5)],  # running speed bins
...
'running_speed_bin_edges': running_bin_edges.tolist(),
```

iii. Directly mandated by the Decoder Output spec ("five equal percentile bins"). The AI stored the
edges in metadata so the discretization can be inverted, and plotted the running-speed histogram with
the bin edges overlaid in the `--show-processing` figure to show the discretization is correct.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated onto the same session-wide 30 Hz grid as the neural data before trial
segmentation and then sliced with the same `trial_time_indices`, so the alignment is exact by
construction.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial  = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial = running_resampled[trial_time_indices]
```

iii. Step 3 of CONVERSION_NOTES.md notes that all data streams share a hardware-synchronised clock
("Temporal sync: NI PCI-6612 board at 100 kHz, all clocks synchronized"), so interpolating the running
trace onto the ophys-derived grid is valid; using one shared index vector then guarantees per-bin
correspondence.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area**, ~30 Hz) with its `timestamps`, plus
`acquisition/EyeTracking/likely_blink/data` to mask blinks. Pupil *width*/*height* are not used. If the
`EyeTracking` group is absent the experiment keeps a NaN pupil trace (3 of 202 experiments:
795953296, 833631914, 806456687).

ii.
```python
try:
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
    pupil_area = pupil_tracking['area'][()]
    pupil_ts   = pupil_tracking['timestamps'][()]
    likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`EyeTracking/pupil_tracking/area` → output[3]:
pupil_diameter — Interpolate to 30 Hz, handle blinks (NaN→interpolate), discretize into 5 percentile
bins — **Use pupil area as proxy for diameter**". Step 4 flags the blink issue ("Pupil: area with NaNs
during blinks, ~9 % blinks ... Need to handle NaN/blinks"), and Step 1 identifies
`EyeTrackingTable.from_nwb()` as the corresponding SDK loader.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink samples are set to NaN; (2) all NaNs are filled by linear interpolation **in the native
eye-tracking time base**; (3) the filled trace is linearly interpolated onto the 30 Hz grid; (4) it is
discretized with globally-pooled quintile edges computed in Pass 2 from blink-excluded pupil samples of
every experiment; (5) any remaining NaN (the 3 experiments without eye tracking) maps to bin 0.
Observed bin occupancy 0.228/0.185/0.191/0.190/0.206.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```
```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if not nans.any():  return arr.copy()
    if nans.all():      return np.zeros_like(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. CONVERSION_NOTES.md Step 5 decision 9: "Pupil NaN handling: During blinks (likely_blink=True or
NaN), linearly interpolate. Compute percentile bins from non-blink data." The rationale is that blink
artefacts would otherwise corrupt both the percentile edges and the neighbouring samples, while
interpolating keeps the output defined at every bin (so no trial has to be dropped).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: five globally-computed equal-percentile bins `bin_0` … `bin_4`, edges
stored in `metadata['pupil_area_bin_edges']`, NaN → bin 0. Consequence: the 3 experiments without eye
tracking are emitted with a constant `bin_0` pupil row (visible in `verification_full_out.txt` as
sessions with output range [0.0, 0.0]).

ii.
```python
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
...
[f'bin_{i}' for i in range(5)],  # pupil diameter bins
...
'pupil_area_bin_edges': pupil_bin_edges.tolist(),
```

iii. Mandated by the Decoder Output spec ("Pupil diameter, discretized into five equal percentile
bins. Time-varying."). Step 9's consistency table reports "Pupil diameter bins 19–23 % each — Yes
(roughly equal)", and the `--show-processing` figure overlays the edges on the pupil histogram.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed: resampled onto the shared 30 Hz grid before trial segmentation, then
sliced with the same `trial_time_indices`.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. Same reasoning as 5-d — the eye-tracking camera is hardware-synchronised to the same 100 kHz sync
board, so its timestamps are directly comparable with the ophys timestamps and a single interpolation
onto the common grid is sufficient.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually-exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`,
`correct_reject`. Observed distribution: hit 0.302, miss 0.572, false_alarm 0.017,
correct_reject 0.108.

ii.
```python
trials = {
    ...
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
    ...
}
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`trials.hit/miss/false_alarm/correct_reject` →
output[4]: trial_outcome — Static per-trial categorical — 4 classes"; Step 4 verified "Each valid trial
has exactly 1 outcome — Confirmed". The planned sanity check "Go trials → hit or miss; Catch trials →
FA or CR" comes from the same table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A fixed mapping hit=0, miss=1, false_alarm=2, correct_reject=3 via an if/elif chain; trials matching
none of the four are dropped. Although the variable is conceptually static per trial, it is broadcast
to a constant row of length `n_tp` so that the output matrix is a single `(5, n_timepoints)` array.

ii.
```python
if trials['hit'][trial_idx]:              outcome = 0  # hit
elif trials['miss'][trial_idx]:           outcome = 1  # miss
elif trials['false_alarm'][trial_idx]:    outcome = 2  # false_alarm
elif trials['correct_reject'][trial_idx]: outcome = 3  # correct_reject
else:
    continue  # Unknown outcome, skip
```
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
...
['hit', 'miss', 'false_alarm', 'correct_reject'],  # trial outcomes
```

iii. The format spec says outputs should be time-varying "if at all possible", and all five outputs
must share one array, so the per-trial label is repeated across bins. Step 10 Check 5 verified "Trial
outcome constant within trials", and Step 12 checked that outcomes vary across sessions ("some sessions
have high hit rate, others mostly miss").

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **File/field failures**: the whole NWB read is wrapped in `try/except`; on failure an ERROR is printed
  and the experiment is skipped. Missing eye tracking is caught separately and yields a NaN pupil trace
  rather than dropping the experiment (3 experiments).
- **Missing stimulus table / no valid ROIs**: experiment skipped with a warning.
- **Too few trials**: experiments with < 2 valid or < 2 processed trials are skipped; trials with < 3
  time bins are skipped.
- **Unclassifiable trials**: trials with none of the four outcome flags are skipped.
- **Blinks / NaN pupil**: blinks → NaN → linearly interpolated; any residual NaN → bin 0 in
  `digitize_to_bins`.
- **Degenerate percentile edges** (constant signal): edges nudged apart by 1e-10 so `np.digitize` stays
  monotone; empty valid pool falls back to `np.linspace(0, 1, 6)`.
- **Bytes vs. str and float vs. bool HDF5 columns**: decoded/cast on load (`image_name`, `omitted`).
- **Interpolation edges**: `np.interp` clamps to the first/last sample rather than extrapolating;
  events are clipped to ≥ 0 afterwards.

ii.
```python
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None
```
```python
if isinstance(trials['initial_image_name'][0], bytes):
    trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
    trials['change_image_name']  = np.array([x.decode() for x in trials['change_image_name']])
...
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
```
```python
for i in range(1, len(edges)):
    if edges[i] <= edges[i-1]:
        edges[i] = edges[i-1] + 1e-10
...
binned = np.clip(binned, 0, n_bins - 1)
binned[np.isnan(values)] = 0
```
```python
if raw_data['pupil_area'] is not None:
    ...
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. CONVERSION_NOTES.md Step 5 decision 9 covers blinks; Step 6 states "Pupil blinks
(likely_blink=True) set to NaN and interpolated before resampling. Events are clipped to >=0 after
interpolation"; Step 10 Check 5 enumerates the edge-case checks that were run ("No NaN in neural data",
"All events non-negative", "All output values in valid range", "All sessions have >= 2 trials"). The
choice to map missing pupil to bin 0 rather than dropping the session is not separately justified in
the notes — it is inherited from the generic NaN rule in `digitize_to_bins`.

## 9-a. What are the most time-consuming steps of the code?

i. Conversion took 358 s (6.0 min) in total for 202 experiments, and per-experiment timing is printed
(`Loaded in 0.3s, processed in 0.5s`). The dominant costs are:
1. **Reading the NWB files three times** — Pass 1 (image names), Pass 2 (running + pupil for the
   percentile pools), Pass 3 (full load). Pass 3 reads the entire `(n_timepoints, n_cells)` event
   array (up to 140 k × 666 floats) plus running (~270 k samples) and eye tracking (~136 k samples).
2. **Resampling the neural data** — `interpolate_to_regular_grid` calls `np.interp` once **per neuron**
   in a Python loop, i.e. up to 666 interpolations of ~140 k → ~136 k points per experiment; this is
   the bulk of the "process" time.
3. **Per-trial output construction** — `get_image_at_timepoints` (Python loop over ~255 bins) and
   `get_image_change_at_timepoints` (Python loop over *all* ~350 session change flashes) are called
   once per trial, i.e. ~52 k times each.
4. **Pickling 8.3 GB** (9.1 s) — small in wall-clock but the dominant disk/memory cost.

ii.
```python
        t0 = time.time()
        print(f"\n[{idx+1}/{len(exp_table)}] Processing experiment {eid}...", flush=True)
        raw_data = load_experiment_data(nwb_path, eid)
        t_load = time.time() - t0
        ...
        t_process = time.time() - t0 - t_load
        print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Sequential processing of experiments
(could parallelize); Multiple passes over NWB files." Step 7 estimated ~0.5 s/session → ~2–3 min for
the full run, "well under 15 min limit", so no further optimisation was pursued; the actual run (6 min)
was within the same order. `ProcessPoolExecutor` is imported but never used.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four remain un-vectorized:
1. `interpolate_to_regular_grid`: `for i in range(data.shape[1])` over neurons — could be replaced by a
   single `searchsorted`-based interpolation or by `scipy.interpolate.interp1d(..., axis=0)`.
2. `get_image_at_timepoints`: `for i in range(n_tp)` over time bins — the `searchsorted` result can be
   used directly as a fancy index (`codes[np.clip(insert_idx, 0, None)]` with a mask for `< 0`).
3. `get_image_change_at_timepoints`: `for cs, ce in zip(change_starts, change_stops)` scans every
   change flash of the whole session for every trial — one `searchsorted` on the trial's own change
   would suffice, or the whole session's change mask could be computed once and then sliced per trial.
4. `img_id_global = np.array([local_to_global.get(v, 0) for v in ...])` — a per-bin Python list
   comprehension where a lookup array (`lut[image_idx]`) would do; `global_image_names.index(name)` is
   also a linear search.
   Additionally, the per-trial `(regular_ts >= start) & (regular_ts < stop)` full-length boolean mask is
   built once per trial over the entire session grid (O(n_trials × n_session_bins)) where
   `np.searchsorted` would be O(log n).

ii.
```python
    for i in range(data.shape[1]):
        result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```
```python
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
```
```python
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```
```python
        trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
        trial_time_indices = np.where(trial_mask)[0]
```

iii. CONVERSION_NOTES.md Step 6 claims "Code speedups added: Vectorized interpolation using np.interp;
Efficient searchsorted for image identity assignment" — a partly optimistic self-assessment, since the
per-neuron and per-timepoint Python loops above survive. The AI's justification for stopping there is
in Step 7: the measured ~0.5 s/session gave an estimated 2–3 min full run, comfortably inside the
15-minute budget set by the instructions, so no further vectorization was required.

## 9-c. What processing does the code repeat multiple times?

i. Repeated work:
- **Each NWB file is opened and read three times** (Pass 1 image names, Pass 2 running/pupil, Pass 3
  everything). Running speed and the pupil area/blink arrays are therefore decompressed twice, and the
  stimulus `image_name` column twice.
- The **global image-name set** could have been assembled during Pass 3 (the reference builds its
  global mapping after extraction, in one pass); instead Pass 1 exists only to pre-compute it.
- `get_image_change_at_timepoints` re-scans the **whole session's change flashes** for every trial, and
  `get_image_at_timepoints` rebuilds the `name_to_idx` dict and the non-omitted stimulus arrays on every
  trial call.
- `local_to_global` performs a linear `list.index` per local image name per experiment.

ii.
```python
    # First pass: determine global image names
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f: ...
    # --- Collect all running speed and pupil values for global percentile computation ---
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        with h5py.File(nwb_path, 'r') as f:
            running = f['processing']['running']['speed']['data'][()]
            ...
    # --- Main processing pass ---
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        raw_data = load_experiment_data(nwb_path, eid)
```
```python
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}   # rebuilt every trial
```

iii. The three-pass design is a deliberate trade-off documented in Step 6 ("3-pass approach: (1)
collect global image names, (2) compute global percentile bins for running/pupil, (3) process
experiments") and acknowledged as an inefficiency in the same section ("Multiple passes over NWB
files"). The motivation is that the percentile bin edges and the image-code mapping must be global
across the dataset, which requires knowing all values before any trial is encoded; the AI accepted the
extra I/O because the total runtime stayed within budget.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work whose result is never used, or that adds no information:
- `change_stops` is extracted and iterated (`for cs, ce in zip(...)`) but `ce` is never used — the
  window is always `cs + 0.75`.
- `cell_specimen_ids[valid_roi]` is loaded and returned but never consumed; `trials['is_change']` and
  `stim['stop_time']` are loaded but unused (change is taken from the stimulus table, not the trials
  table).
- The `valid_roi` filter is a **no-op** on this release — every ROI in the on-disk NWB files already has
  `valid_roi == True` (checked across a random sample of 25 files: 3,764 ROIs, 0 invalid) — so the
  boolean indexing copies the whole event array for nothing.
- `np.maximum(events_resampled, 0)` cannot change anything: linear interpolation of a non-negative
  signal is non-negative.
- **Up-sampling the Multiscope experiments from ~10.7 Hz to 30 Hz** creates ~3× as many samples as
  there is information in (34 of 202 experiments); the single-plane data is resampled 30.94 → 30 Hz,
  which also buys nothing beyond nominal uniformity while smearing impulse-like events across bins.
- An empty `np.zeros((0, n_tp))` input array is allocated per trial (51,992 allocations) although
  `input_names` is empty.
- `ProcessPoolExecutor`/`as_completed` and `os`-level imports for parallelism are imported and never
  used.
- The pickle stores the neural data densely as float32 even though only ~0.25 % of the event samples
  are non-zero, giving an 8.3 GB file.

ii.
```python
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)   # `ce` unused
```
```python
            return {
                'experiment_id': experiment_id,
                'valid_roi': valid_roi,
                'cell_specimen_ids': cell_specimen_ids[valid_roi],   # never used
                ...
                'is_change': trials_grp['is_change'][()].astype(bool),   # never used
```
```python
    events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
    # Ensure non-negative (events should be >= 0)
    events_resampled = np.maximum(events_resampled, 0)
```
```python
            session_input.append(np.zeros((0, n_tp), dtype=np.float32))
```
```python
from concurrent.futures import ProcessPoolExecutor, as_completed   # never used
```

iii. CONVERSION_NOTES.md does not flag any of these as unnecessary; the closest statements are Step 6's
"Code inefficiencies identified" (parallelism, multiple passes) and the trajectory's acceptance that
up-sampling the 11 Hz Multiscope data "won't add real information" but is required by the constraint
that time bins be identical across all trials and sessions. The `valid_roi` filter is justified in
Step 5 decision 10 as matching the SDK default (`exclude_invalid_rois=True`) even though it removes
nothing here.
