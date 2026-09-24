# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object model. It reads the project metadata CSV
`project_metadata/ophys_experiment_table.csv`, intersects it with the set of NWB files actually
present on disk (`behavior_ophys_experiments/*.nwb`, 284 files), and then keeps only **active
(non-passive)** experiments (`passive == False`), giving 202 experiments / 38 mice / 174 unique
ophys sessions. It does **not** filter on `project_code`, so both `VisualBehavior` (168 single-plane
CAM2P experiments) and `VisualBehaviorMultiscope` (34 MESO imaging planes from 6 sessions of a
single mouse) are included. Each NWB file is then opened directly with `h5py` and only the needed
datasets are read (event traces, ophys timestamps, valid_roi, running speed, pupil area, trials
table, stimulus presentations). Two passes are made over the files: one to gather global statistics
(running/pupil percentiles and the image-name vocabulary) and one to convert.

ii.
```python
DATA_DIR = 'data/visual-behavior-ophys-1.1.0'
EXPT_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')

def get_experiment_list(sample=False):
    expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    downloaded_ids = set()
    for f in os.listdir(EXPT_DIR):
        if f.endswith('.nwb'):
            eid = int(f.split('_')[-1].replace('.nwb', ''))
            downloaded_ids.add(eid)
    expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
    expt_table = expt_table[expt_table.passive == False]
    ...
    return expt_table
```
```python
def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    data['events'] = f['processing/ophys/event_detection/data'][:].T
    data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
    data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
    data['running_speed'] = f['processing/running/speed/data'][:]
    ...
```

iii. From CONVERSION_NOTES.md and the trajectory: "Efficient NWB loading with h5py (no SDK
overhead)"; the AI verified against the SDK that `exclude_invalid_rois=True` is the SDK default and
reimplemented that filter manually. Passive sessions were excluded because "those don't have the
Visual Behavior task" (no licking/reward, so trial outcome is not meaningful) — recorded in the
Step-4 discrepancy table as "Exclude passive (no behavioral task)". Multiscope data was kept because
"the task description doesn't specify this filtering… The task says to collect data under the
'Visual Behavior' task. I should include all active sessions."

## 1-b. How are the data split into subjects?

i. Subjects are the `subject_id` string stored inside each NWB file
(`general/subject/subject_id`). Unique subject ids are accumulated in order of first appearance into
`subjects`, and `subject_idx` records, for each output session, the index of its mouse. 38 mice
result.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
if isinstance(data['subject_id'], bytes):
    data['subject_id'] = data['subject_id'].decode()
...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The AI took the mouse id from the NWB file itself rather than the metadata table so that the
subject label is guaranteed to come from the same source as the neural data ("Subjects (mice)" is
reported in Step 2 as 38, which it cross-checked against the experiment table).

## 1-c. How are the data split into sessions?

i. **Each ophys experiment (i.e. each imaging plane / each NWB file) is emitted as one "session"** in
the output. Single-plane (CAM2P) recordings have one experiment per ophys session, so there the two
coincide. For the Multiscope recordings, the 6 real ophys sessions of mouse 457841 are split into 34
separate output sessions that share identical trials, behaviour and outputs but hold disjoint neuron
sets. The verification log confirms this: "Subject 457841: 34 sessions", with trial counts repeating
in blocks (209 ×7, 309 ×7, 287 ×7, 239 ×5, 196 ×5, 265 ×3).

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    ...
    result = process_experiment(expt_id, expt_info, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])      # one session per experiment
    all_output.append(result['output'])
    subject_idx_list.append(unique_subjects.index(subj))
```

iii. Trajectory step 22: "MESO sessions have multiple experiments (imaging planes) per session —
they share the same behavioral data but have different neurons… For the decoder, each experiment
should be treated as a separate 'session' since each has different neurons." The AI also noted the
paper performs decoding "per imaging plane".

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. Trials are selected with the mask
`go | catch`, and each trial spans `start_time` → `stop_time` (the full variable-length trial window,
which contains several pre-change image flashes plus the post-change response window; mean 91.5 bins
≈ 8.5 s). Within the window a uniform grid `np.arange(start_time, stop_time, 93.23 ms)` defines the
trial's time bins.

ii.
```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop  = trial_data['stop_time'][trial_idx]
    trial_neural, common_ts = resample_events_to_common_bins(
        events, ophys_ts, trial_start, trial_stop)
```
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. "Trial curation: Include Go and Catch trials, exclude Aborted and Auto-rewarded" — taken
directly from the Decoder Task specification. The AI verified on one file that 323 go + 42 catch =
365 of 503 trials survive, with 133 aborted and 5 auto-rewarded removed. Trial window = trial
start→stop was chosen so that both the pre-change flashes and the response window are inside the
trial (needed for time-varying image identity / image change).

## 1-e. How are trials filtered based on quality controls?

i. Filters applied: (a) passive experiments dropped entirely; (b) experiments with zero valid
neurons or with no stimulus-presentation table dropped; (c) experiments with fewer than 2 go/catch
trials dropped, and after conversion experiments retaining <2 usable trials dropped; (d) trials whose
window yields fewer than 2 common bins dropped; (e) trials whose outcome is none of
hit/miss/correct_reject/false_alarm dropped. No trial is dropped for behavioural/engagement reasons.

ii.
```python
if n_neurons == 0:
    print(f'  WARNING: No valid neurons in experiment {expt_id}'); return None
if raw['stimulus_presentations'] is None:
    print(f'  WARNING: No stimulus presentations in experiment {expt_id}'); return None
if len(include_indices) < 2:
    print(f'  WARNING: Too few trials in experiment {expt_id}'); return None
...
    if trial_neural is None:      # <2 common bins
        continue
...
    outcome = get_trial_outcome(trial_data, trial_idx)
    if outcome == -1:
        continue
...
if len(neural_trials) < 2:
    print(f'  WARNING: Too few valid trials in experiment {expt_id}'); return None
```

iii. The ≥2-trial rule comes straight from the format spec ("There needs to be at least two trials
within each session"). The AI documented under Step 10 Check 5 that it explicitly looked for these
edge cases and found none triggered in practice ("Trials with < 2 timepoints: skipped (none found);
Experiments with 0 valid neurons: skipped (none found)").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the **detected (deconvolved) calcium events**, stored in
the NWB as (n_timepoints, n_neurons) and transposed to (n_neurons, n_timepoints). The unsmoothed
events are used, not the SDK's half-gaussian `filtered_events`, and not dF/F. Timestamps are taken
from `processing/ophys/dff/traces/timestamps` (the shared ophys frame clock).

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. From methods.txt the AI extracted that the paper analysed "discrete calcium events that were
regressed from the raw fluorescence traces", and from the SDK (`events.py`, `event_detection.py`)
that the half-gaussian filter (scale 2/31 s, 20 steps) is applied only "to smooth it for
visualization". Its conclusion (trajectory step 84): "the raw events in the NWB ARE the detected
calcium events… so using raw events is correct."

## 2-b. How is the `neural` data processed?

i. Two operations: (1) restrict to valid ROIs; (2) re-bin from the native ophys frame rate onto a
common 93.23 ms grid anchored at each trial's start, **summing** the events of all ophys frames whose
timestamp falls in each bin. No smoothing, normalisation, z-scoring or neuron-level standardisation
is applied. Bin membership is computed with `np.digitize` against bin edges centred on the common
timestamps (so the first/last bin extend half a bin beyond the trial window).

ii.
```python
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    n_common = len(common_ts)
    if n_common < 2:
        return None, None
    trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
    trial_ophys_ts = ophys_ts[trial_mask]
    trial_events = events[:, trial_mask]
    bin_edges = np.concatenate([[common_ts[0] - COMMON_DT/2],
                                (common_ts[:-1] + common_ts[1:]) / 2,
                                [common_ts[-1] + COMMON_DT/2]])
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    return resampled, common_ts
```

iii. "Neural data: Raw detected events (not filtered), summed within each common bin" (Step 5, key
decision 2). Summing rather than averaging was chosen to preserve the total event magnitude, and the
AI verified this in Step 10 Check 2 by re-reading the NWB for experiment 879332693 and comparing
per-trial event sums (exact match for interior trials; 2–6 % differences for trials whose common bins
extend half a bin past the trial boundary, which it reproduced exactly against the expanded window).

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs with `valid_roi == False` in the segmentation `cell_specimen_table` are dropped; nothing
else (no event-rate, SNR, or activity thresholding). In these released NWB files every ROI is already
flagged valid, so the filter is a no-op in practice (29,444 neurons retained over 202 experiments).

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
n_neurons = events.shape[0]
```

iii. Step 1/Step 10 notes: the SDK loads experiments with `exclude_invalid_rois=True` by default, so
the AI replicated that behaviour explicitly since it bypasses the SDK ("ROI filtering: valid_roi flag
from segmentation pipeline… matching SDK default"). It listed "Neuron count matches NWB valid_roi
count" as a planned sanity check and marked it passed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`trials.start_time`): the common time grid for a trial begins at
`start_time` and runs to `stop_time`, and every ophys frame is assigned to that grid by its ophys
timestamp. `metadata['temporal_alignment_event'] = 'Trial start time'`, `off_start = 0.0`,
`off_end = None` (variable-length trials). All other streams are placed on the identical grid, so
neural/behavioural/stimulus data are aligned by construction.

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
...
'temporal_alignment_event': 'Trial start time',
'off_start': 0.0,
'off_end': None,
```

iii. The task requires "Temporally align based on ophys timestamp" and "Segment each recording
session into individual trials based on how they are defined in the experiment". The AI used the
ophys timestamps as the master clock and confirmed in step 11 of the trajectory that running (60 Hz)
and pupil (~30 Hz) streams are on the same session clock and must be resampled onto it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **93.23 ms (10.73 Hz) for every trial and every session**, and yes — explicit rebinning is
applied. The native rate differs by rig (CAM2P single-plane ≈ 32.3 ms / 30.94 Hz; MESO multiplane ≈
93.23 ms / 10.73 Hz). Because the AI kept both rigs it adopted the slower MESO period as the common
bin and downsampled all CAM2P data ~3× (summing ~3 frames per bin); MESO data is essentially
re-gridded 1:1 onto bins of the same width anchored at each trial start.

ii.
```python
# Common time bin: use ~93ms (MESO rate) to accommodate all experiments
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
'time_bin_size': COMMON_DT * 1000,
```

iii. Step 4 discrepancy table: "Frame rates — CAM2P ~31 Hz, MESO ~10.7 Hz → Resample to common
93.23 ms bins (MESO rate)". Trajectory step 49: "The task says 'Time bins should be the same size for
all trials and sessions.' This is a hard requirement… we can't meaningfully upsample MESO data;
downsampling CAM2P data to ~93 ms is straightforward (sum events in each bin)."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentation table** (`intervals/Natural_Images_Lum_Matched_set_training_2017_
presentations`), using its `image_name` and `start_time` columns — not the trials table's
`initial_image_name`/`change_image_name`.

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    vals = stim[key][:]
    ...
data['stimulus_presentations'] = stim_data
```

iii. Step 5 mapping table: `stimulus_presentations.image_name → output[0] (image_identity)`, "Map to
categorical index, 16 unique images". Using the flash table gives the image actually on screen at
every moment of the trial, independent of how many flashes precede the change.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For every common time bin, the most recent stimulus presentation (by `start_time`) is found with
`np.searchsorted`; its image name is mapped to a global integer code. During the 500 ms grey
inter-stimulus interval and during *omitted* flashes the previously shown image is carried forward, so
the variable is piecewise-constant and never encodes "grey". The code vocabulary is built globally
across all experiments (16 images = image sets A and B pooled, sorted alphabetically), so codes are
consistent across sessions.

ii.
```python
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    last_valid = 0
    for t in range(n_tp):
        idx = indices[t]
        if idx < 0:
            image_id[t] = last_valid; continue
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]; last_valid = image_to_idx[img]
        else:
            image_id[t] = last_valid
    return image_id
```
```python
all_images = sorted(all_images)          # 'omitted' excluded when building vocabulary
image_to_idx = {img: i for i, img in enumerate(all_image_names)}
```

iii. Step 5, key decision 6: "Image identity during gray screen: Assign most recently shown
non-omitted image", justified by the instruction "Image identity (of the image presented during the
non-grey screen)". Step 10 Check 2 verified the resulting codes against the raw stimulus table for
three trials (im085, im077, im061 all matched).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `common_ts` grid used to bin the neural events for that trial,
so row 0 of the output matrix is sample-for-sample aligned with the neural matrix.

ii.
```python
trial_neural, common_ts = resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop)
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
...
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned,
                         np.full(n_common, outcome, dtype=np.int64)], axis=0)
```

iii. All NWB time columns share one session clock, so evaluating the stimulus table at the neural bin
centres guarantees alignment. `--show-processing` plots overlay image identity, image change and
neural activity on a common time axis with trial boundaries and change times marked, which the AI
used to confirm "proper alignment" (Step 7).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus presentation table's `is_change` flag together with the flash `start_time` /
`stop_time` (i.e. the flash that the Allen pipeline marks as an image change). The trials table's
`change_time` / `go` columns are loaded but not used for this output.

ii.
```python
is_change   = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
start_times = stim_data['start_time']
stop_times  = stim_data['stop_time']
change_indices = np.where(np.array(is_change) == 1)[0]
```

iii. Step 5 mapping: `stimulus_presentations.is_change → output[1] (image_change)`, "Binary time
series, 1 during change stimulus". Because `is_change` is only set on flashes whose image differs
from the previous flash, catch (sham-change) trials correctly get an all-zero trace.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is initialised to zero and set to 1 for every common bin whose timestamp lies
inside a change flash's presentation interval, i.e. a **250 ms window starting at the change**
(≈2–3 bins). Over the full dataset this yields 2.8 % change bins.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    change = np.zeros(n_tp, dtype=np.int64)
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    return change
```

iii. "1 during change stimulus" (README / Step 5). The instruction was "Have value of 1 right after a
change in image identity, otherwise 0", and the AI took the change stimulus presentation itself as
that window. It cross-checked the resulting rate against the task design: "Image change rate ~2.8 % of
timepoints — ✓" (Step 9 consistency table).

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required: the variable is already binary, encoded as {0, 1} with
`output_values[1] = ['no_change', 'change']`.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'correct_reject', 'false_alarm'],
]
```

iii. The Decoder Task specifies "Image change, binary variable", so the AI encoded it directly as two
categories with named values.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: evaluated on the trial's `common_ts` grid, so it is bin-for-bin
aligned with the neural matrix; it occupies row 1 of the (5, n_timepoints) output.

ii.
```python
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
...
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned, ...], axis=0)
```

iii. Step 10 Check 2: "Image change detection verified: bins within change stimulus window correctly
marked as 1"; the processing plots also draw a dashed red line at each trial's `change_time` over the
image-change trace.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the running-wheel
encoder speed on its native (~60 Hz) clock.

ii.
```python
data['running_speed']      = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. Step 5 mapping table: `running_speed → output[2]`, "Interpolate + 5 percentile bins, global
percentiles". This is the same signal the SDK exposes as `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Linear interpolation from the running clock onto the trial's common time grid (NaNs removed
before interpolating; out-of-range → NaN). (2) Discretisation into 5 bins using **global** percentile
edges. The edges are computed in a separate first pass over all 202 NWB files, sub-sampling up to
2000 non-NaN speed samples per experiment (seeded per experiment for determinism) from the whole
session — not just the trial windows. Resulting edges:
`[-22.15, -0.0037, 0.337, 14.12, 33.15, 97.87]` cm/s.

ii.
```python
def interpolate_to_common(data, source_ts, common_ts):
    valid = ~np.isnan(data)
    f_interp = interpolate.interp1d(source_ts[valid], data[valid],
                                    kind='linear', bounds_error=False, fill_value=np.nan)
    return f_interp(common_ts)
```
```python
run_data = f['processing/running/speed/data'][:]
valid_run = run_data[~np.isnan(run_data)]
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
all_running.append(valid_run)
...
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
```

iii. Step 5, key decision 5: "Percentile binning: Computed across all sessions globally (sampled 2000
values per session)" — global edges keep the five categories comparable across sessions, and
sub-sampling keeps the extra pass cheap. The AI checked the resulting marginal distribution:
"Running speed bins 5 equal percentile → ~20 % each ✓" (measured 0.200/0.202/0.183/0.210/0.205).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count percentile bins (0–20–40–60–80–100 %), applied with `np.digitize` on the four
interior edges and clipped to [0, 4]; bins that come out NaN (no running data at that time) are set
to bin 2, the middle bin.

ii.
```python
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. Directly follows the Decoder Task requirement "Running speed, discretized into five equal
percentile bins". Equal-occupancy bins also give a balanced decoding target (chance = 0.2).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly onto the trial's `common_ts` grid (the same grid on which the neural events
were binned), so it is bin-for-bin aligned; it is row 2 of the output matrix.

ii.
```python
run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
...
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned, ...], axis=0)
```

iii. The running and ophys streams are on the same hardware-synced session clock, so linear
interpolation onto the ophys-derived bin centres is a valid resampling. The `--show-processing`
figure plots raw running speed above the binned running trace to visually confirm no time shift.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area**, not width/diameter) with its own
timestamps. Note this NWB column already carries NaN on every `likely_blink` frame (verified: the NaN
fraction equals the blink fraction exactly), and `area_raw` is the un-masked version, which the AI did
not use. Three of the 284 files have no EyeTracking group at all.

ii.
```python
try:
    data['pupil_area']       = f['acquisition/EyeTracking/pupil_tracking/area'][:]
    data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
    data['has_pupil'] = True
except KeyError:
    data['has_pupil'] = False
    data['pupil_area'] = None
    data['pupil_timestamps'] = None
```

iii. Step 5 mapping: `pupil_tracking.area → output[3] (pupil_diameter)`, "Interpolate + 5 percentile
bins". Because the output is rank-discretised into percentile bins, the AI treated pupil area as an
equivalent proxy for pupil size; the README labels the variable "Pupil area in equal percentile bins".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: NaN samples (i.e. blinks and tracking failures) are dropped,
the remaining samples are linearly interpolated onto the trial's common grid (which effectively
interpolates across blinks), then discretised with global 5-percentile edges computed in the same
first pass (up to 2000 samples/experiment). Edges: `[228.6, 4303.4, 5455.9, 6709.9, 8652.3, 107624.5]`
px².

ii.
```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(raw['pupil_area'], raw['pupil_timestamps'], common_ts)
else:
    pupil_interp = np.full(n_common, np.nan)
```
```python
pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
valid_pupil = pupil_data[~np.isnan(pupil_data)]
...
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
```

iii. Same rationale as running speed (global edges for cross-session comparability). Blink handling is
implicit: `interpolate_to_common` starts by masking `~np.isnan(data)`, and the NWB `area` column is
already NaN at `likely_blink` frames, so blinks are excluded from both the percentile estimate and the
interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-count global percentile bins via `np.digitize`, clipped to [0, 4]; NaN (no pupil sample
in range, or an experiment with no eye tracking at all) → bin 2. Measured marginal distribution:
0.199 / 0.182 / 0.216 / 0.200 / 0.202.

ii.
```python
if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
    pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
    pupil_binned[np.isnan(pupil_interp)] = 2
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. Required by the Decoder Task ("Pupil diameter, discretized into five equal percentile bins").
Step 5, key decision 7: "Missing pupil data: Fill with NaN, assign to middle bin (bin 2)" — the middle
bin was chosen as the least-committal label. The AI noticed the consequence in Step 9/trajectory 66:
"pupil data shows some sessions with min > 0 … likely because those sessions have no pupil data and
the NaN fill is being used."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same per-trial `common_ts` grid as the neural data; row 3 of the output
matrix.

ii.
```python
pupil_interp = interpolate_to_common(raw['pupil_area'], raw['pupil_timestamps'], common_ts)
...
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned,
                         np.full(n_common, outcome, dtype=np.int64)], axis=0)
```

iii. Same justification as running speed — eye-tracking timestamps share the session clock with the
ophys frames, so resampling onto the neural bin centres aligns the two streams by construction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `correct_reject`,
`false_alarm`.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if   trial_data['hit'][trial_idx]:            return 0
    elif trial_data['miss'][trial_idx]:           return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]:    return 3
    else: return -1
```

iii. Step 5 mapping: `trials.hit/miss/CR/FA → output[4] (trial_outcome)`, "Categorical static, 4
categories". These are the canonical change-detection outcome labels for go/catch trials. Step 10
Check 2: "Verified trial outcomes for 5 trials against original NWB data — all outcomes match".

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to an integer 0–3 (hit, miss, correct_reject, false_alarm) and **broadcast across all time
bins of the trial** so that it occupies row 4 of the time-varying output matrix (constant within a
trial). Trials matching none of the four (outcome −1) are dropped. `output_values[4]` lists the labels
in the same order as the integer codes. Measured distribution: hit 30.2 %, miss 57.2 %,
correct_reject 10.8 %, false_alarm 1.7 %.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
trial_output = np.stack([img_id, img_change, run_binned, pupil_binned,
                         np.full(n_common, outcome, dtype=np.int64)], axis=0)
```
```python
output_values = [..., ['hit', 'miss', 'correct_reject', 'false_alarm']]
```

iii. The format spec says "Can be time-varying or discrete values per trial. If at all possible, make
it time-varying", so a per-trial constant was tiled to the full trial length to keep all five outputs
on one (5, n_timepoints) array. The AI compared the resulting distribution with the reference texts:
"Trial outcome dist ~30 % hit, ~57 % miss → 30.2 % hit, 57.2 % miss ✓" (Step 9).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Whole experiment fails to load / process**: wrapped in `try/except`, logged with traceback, and
  skipped without aborting the run.
- **No EyeTracking group** (3 of 284 files): `has_pupil=False`, pupil filled with NaN and therefore
  labelled bin 2 for the entire session; the run still proceeds.
- **NaN running/pupil samples** (blinks, tracking dropouts): removed before interpolation; NaNs that
  remain after interpolation (out-of-range extrapolation) are assigned the middle bin (2).
- **Omitted stimulus flashes**: image identity carries the last valid image forward.
- **Missing stimulus table / zero valid neurons / <2 trials / <2 time bins / unrecognised outcome**:
  the experiment or trial is skipped with a printed warning.
- **Trial windows extending past the recording**: only ophys frames inside the trial window are
  binned; bins with no frames stay zero.

ii.
```python
try:
    result = process_experiment(expt_id, expt_info, ...)
except Exception as e:
    print(f'  ERROR: {e}')
    import traceback; traceback.print_exc()
    continue
if result is None:
    continue
```
```python
run_binned[np.isnan(run_interp)] = 2
pupil_binned[np.isnan(pupil_interp)] = 2
...
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```
```python
if img == 'omitted':
    image_id[t] = last_valid
```

iii. Step 10 Check 5 ("Check for edge cases") enumerates exactly these: "Missing pupil data: handled
with NaN fill + middle bin assignment; Omitted stimuli: use last valid image for identity; Trials with
< 2 timepoints: skipped (none found); Experiments with 0 valid neurons: skipped (none found)". The
2602 "all neural data is zero" warnings in `verification_full_out.txt` were examined and attributed to
genuinely sparse calcium events in low-neuron-count planes rather than to a conversion fault.

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented per-experiment load and total time (`load=…s, total=…s`). The full run took
518 s for 202 experiments. Per the logs, NWB loading is 0.1–1.8 s per experiment while the total is
2.3–6.6 s, so the dominant cost is the **per-trial Python processing** (event re-binning plus the
image-identity and image-change loops), followed by NWB I/O — and there are two I/O passes
(`collect_global_stats`, then conversion). Pickling the 3.0 GB output is also non-trivial.

ii.
```python
t0 = time.time()
raw = load_experiment_data(expt_id)
t_load = time.time() - t0
...
t_process = time.time() - t0
print(f'  Experiment {expt_id}: {n_neurons} neurons, {len(neural_trials)} trials, '
      f'load={t_load:.1f}s, total={t_process:.1f}s')
```

iii. Step 7 documents the estimate ("Load NWB 0.1–1.8 s; Resample + process 0.5–5 s; ~600 s total,
within the 15-minute limit") and trajectory step 50 records that the AI first wrote an
O(n_timepoints × n_ophys_timepoints) resampler, recognised it "will be extremely slow", and replaced
it with the `np.digitize` version before running the full conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- `resample_events_to_common_bins` loops over every output bin and builds a boolean mask over the
  trial's frames for each (`for b in range(n_common)`); `np.add.reduceat` or
  `np.bincount`/matrix-multiply would do this in one call.
- `get_image_at_timepoints` loops over every timepoint in Python even though `np.searchsorted`
  already produced the indices; the omitted/carry-forward logic is a `np.maximum.accumulate` on a
  forward-fill mask.
- `get_image_change_at_timepoints` loops over **all** change flashes of the entire session for every
  trial and builds a full-length mask each time, i.e. O(n_changes_session × n_bins) per trial — this
  is the main reason each experiment takes ~2.3 s even when loading takes 0.1 s. Restricting to the
  flashes inside the trial (`searchsorted`) or vectorising over flash intervals removes it.
Also, per-experiment processing is independent and could have been parallelised across processes.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```
```python
for t in range(n_tp):
    idx = indices[t]
    ...
```
```python
change_indices = np.where(np.array(is_change) == 1)[0]
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. CONVERSION_NOTES.md Step 6 claims "Vectorized event binning using `np.digitize`" and Step 7
concludes the estimated ~10 min runtime is acceptable, so the AI stopped optimising once it was under
the 15-minute budget rather than removing the remaining per-bin/per-flash loops.

## 9-c. What processing does the code repeat multiple times?

i. **A whole extra read pass over every NWB file.** `collect_global_stats` opens all 202 files to read
running speed, pupil area and the stimulus `image_name` column; `load_experiment_data` then re-opens
each file and re-reads running speed, pupil area and the stimulus table. The image-change flash list
(`np.where(is_change)`) is recomputed for each trial from the whole-session table rather than once per
experiment, and the interpolator objects for running and pupil are rebuilt for every trial from the
full-session arrays instead of being constructed once and evaluated per trial.

ii.
```python
def collect_global_stats(expt_table):          # pass 1: reads running/pupil/images from every file
    with h5py.File(fname, 'r') as f:
        run_data   = f['processing/running/speed/data'][:]
        pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
        imgs       = f[f'intervals/{stim_keys[0]}/image_name'][:]
```
```python
for trial_idx in include_indices:              # pass 2: per-trial, from full-session arrays
    run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
    img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. The two-pass design is a deliberate consequence of the AI's decision to use **global** percentile
edges and a global image vocabulary — they must be known before any trial is written. The AI kept the
scan cheap by reading only three datasets per file and sub-sampling 2000 values per experiment, and
did not note the redundancy with the conversion pass in CONVERSION_NOTES.md.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor items only:
- `valid_roi` filtering is a no-op in this release (every ROI is flagged valid), yet the full mask is
  read and applied for all 202 files.
- Several loaded fields are never used to build the outputs: `cell_specimen_ids`, and the trials
  columns `change_time`, `is_change`, `change_image_name`, `initial_image_name`, `aborted`,
  `auto_rewarded` (only `go`/`catch`/outcome/`start_time`/`stop_time` are used; the rest are used only
  by the optional plots), plus the stimulus `omitted` column.
- `input` is populated with per-trial zero-row arrays `np.zeros((0, T))` for all 51,992 trials even
  though `input_names` is empty and the decoder takes no inputs.
- In `--show-processing` mode the entire raw experiment dict is retained in the result object.
- Storing 31 Hz data re-binned to 10.7 Hz as float32 events yields a 3.0 GB pickle, most of which is
  zeros (calcium events are sparse); a sparse representation was not considered.

ii.
```python
all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])
```
```python
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
```
```python
if show_processing:
    result['raw'] = raw
    result['ophys_ts'] = ophys_ts
```

iii. These extra reads are cheap scalar/1-D columns and were pulled in so the same `raw` dict could
serve both the conversion and the `--show-processing` verification plots, which the instructions
required. The zero-width `input` entries exist to satisfy the target format ("input: list of sessions
… for each trial") while signalling that this task has no decoder inputs.
