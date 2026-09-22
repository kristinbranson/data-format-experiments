# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK cache API. It enumerates experiments from the project metadata CSV (`project_metadata/ophys_experiment_table.csv`), intersects that table with the `*.nwb` files actually present on disk in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`, and then keeps only the four *active* behavior session types (`OPHYS_1_images_A`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_6_images_B`). No `project_code` filter is applied, so both `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` (multi-plane) experiments are included. This yields 202 experiments from 38 mice (of the 284 NWB files on disk, 247 sessions, 38 mice).

Each selected NWB file is then opened directly with `h5py` and the needed arrays are read out by HDF5 path: `processing/ophys/event_detection/data`, `processing/ophys/dff/traces/timestamps`, `processing/ophys/image_segmentation/cell_specimen_table`, `intervals/trials`, the natural-image `intervals/*_presentations` table, `processing/running/speed`, and `acquisition/EyeTracking`.

The file is read three separate times over three passes: Pass 1 collects the global image-name vocabulary, Pass 2 collects running-speed/pupil samples for global percentile bin edges, Pass 3 does the actual conversion.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
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
        try:
            eid = int(f.stem.split('_')[-1])
            nwb_ids.add(eid)
        except ValueError:
            pass
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()
    ...
    return active_exps
```

```python
def load_experiment_data(nwb_path, experiment_id):
    try:
        with h5py.File(nwb_path, 'r') as f:
            cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
            valid_roi = cell_table['valid_roi'][()].astype(bool)
            ...
            ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
            events_data = f['processing']['ophys']['event_detection']['data'][()]
            events_valid = events_data[:, valid_roi]
            trials_grp = f['intervals']['trials']
            ...
            running_speed = f['processing']['running']['speed']['data'][()]
            running_ts = f['processing']['running']['speed']['timestamps'][()]
            pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. From CONVERSION_NOTES.md Step 1/Step 4: the AI verified that "dF/F is PRE-COMPUTED" and "Events are PRE-COMPUTED" in the NWB files, so the SDK object layer is not needed and reading HDF5 directly gives the same arrays with less overhead (it explicitly claims in Step 10 Check 3 that "h5py reads same data as AllenSDK NWB reader"). Only the 284 files on disk are a subset of the 1,936 released experiments; the AI documented this and reconciled 38 mice / 174 active sessions against the whitepaper's 82 mice / 551 sessions as "a subset - OK". Passive session types (OPHYS_2, OPHYS_5) were dropped because they are "passive viewing (no lick spout, satiated mice). No meaningful trial outcomes."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values in the filtered experiment table, sorted and cast to `str`. A `subject_to_idx` dict maps each mouse id to its index, and one `subject_idx` entry is appended per emitted session. Result: 38 subjects.

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

iii. Step 5 Key Decision 12: "Subject IDs: Use mouse_id from experiment table." `mouse_id` is the canonical per-animal identifier in the Allen metadata; the AI cross-checked the count (38 on disk vs 82 in the full release) in Step 4.

## 1-c. How are the data split into sessions?

i. **One NWB experiment (= one imaging plane) is emitted as one "session"** in the output structure. The AI does not group experiments by `ophys_session_id`. Passive session types are excluded entirely. This gives 202 output "sessions" that actually come from 174 distinct `ophys_session_id`s: the 168 single-plane `VisualBehavior` experiments are 1:1 with sessions, but the 34 `VisualBehaviorMultiscope` experiments (all from one mouse, 457841) come from only 6 real sessions, so that mouse appears with 34 "sessions" that repeat the same behavior/trials/outputs across 4–7 imaging planes each.

ii.
```python
# one iteration of this loop == one output "session"
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    raw_data = load_experiment_data(nwb_path, eid)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'],
                         region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. Step 5 Key Decision 13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." The reasoning recorded in the trajectory (step 32/36) is: "Each NWB file represents one imaging plane (experiment), and for multiscope setups multiple experiments share a session. Since different planes have different neurons, I should treat each experiment as a separate session in the decoder output format." Step 5 Key Decision 2 covers the passive exclusion: "OPHYS_2, OPHYS_5 are passive viewing (no lick spout, satiated mice). No meaningful trial outcomes." The AI noticed the consequence in the trajectory ("Subject 457841 stands out with 34 sessions, which makes sense since it's a Multiscope mouse with multiple imaging planes per session") but did not change the design.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. The trial window is the full `[start_time, stop_time)` interval, selected on the resampled 30 Hz grid, so trials are variable length (210–377 bins, mean ~254, i.e. ~8.5 s).

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
    trial_ts = regular_ts[trial_time_indices]
    n_tp = len(trial_ts)
    neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Step 5 Key Decisions 3 and 5: "Exclude aborted and auto-rewarded trials: Per task instructions"; "Trial window: Use trial start_time to stop_time from trials table. Variable length across trials." The AI verified the resulting split against the whitepaper: 87.4–87.5 % go (hit+miss) and 12.5–12.6 % catch (FA+CR), matching "GO trials comprise 87.5 % of all trials".

## 1-e. How are trials filtered based on quality controls?

i. Four filters, in order: (1) `(go|catch) & ~aborted & ~auto_rewarded`; (2) an experiment is skipped up front if fewer than 2 trials survive that mask; (3) a trial is skipped if it covers fewer than 3 bins on the 30 Hz grid (this also clips/drops trials whose `stop_time` runs past the end of the recording, since `regular_ts` only spans the ophys recording); (4) a trial is skipped if none of `hit/miss/false_alarm/correct_reject` is True. Finally the experiment is skipped again if fewer than 2 trials survive processing. Minimum trials per emitted session in the full run was 39.

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
    print(f"  WARNING: Only {len(neural_trials)} processed trials ..., skipping")
    return None
```

iii. Step 3 "Trial curation rules": "Aborted trials: premature lick before change → excluded from performance calculations; Auto-rewarded: 5 free rewards at session start + after 10 consecutive misses; Per task instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded." The ≥2-trial requirement follows the format spec ("There needs to be at least two trials within each session"). No `d'`/engagement-based session QC was applied even though the AI catalogued the whitepaper's 10 session-QC criteria in Step 3; it noted the released data already passed those.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the pre-computed **deconvolved calcium events** (FastLZeroSpikeInference / L0 regularized), shape `(n_timepoints, n_cells)`, columns subset to `valid_roi == True`. dF/F traces are *not* used for the neural signal; only `processing/ophys/dff/traces/timestamps` is read, as the ophys time base.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
# events_data shape: (n_timepoints, n_all_cells)
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
```

iii. Step 5 Key Decision 1: "**Neural signal: events (not dF/F)**: Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The trajectory shows the AI explicitly weighing the two and switching: "I'm deciding between dF/F traces and detected calcium events… Actually, wait—the reference paper explicitly uses deconvolved calcium events detected via FastLZeroSpikeInference, which are much closer to spike-like activity than raw dF/F. The instructions emphasize matching the reference processing, so I should use events instead of dF/F." The paper text it quotes is: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."

## 2-b. How is the `neural` data processed?

i. Three operations: (1) the full-session event matrix is **linearly interpolated column-by-column onto a uniform 30 Hz grid** spanning `ophys_ts[0]` → `ophys_ts[-1]`; (2) the interpolated values are clipped at 0 (`np.maximum(..., 0)`); (3) each trial is sliced out and transposed to `(n_neurons, n_timepoints)` and cast to `float32`. **No smoothing/event filtering is applied** — the SDK's causal half-gaussian `filtered_events` is not used, even though the AI catalogued `filter_events_array()` in Step 1. No z-scoring, baseline subtraction, or per-neuron normalization.

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
...
t_start = ophys_ts[0]; t_end = ophys_ts[-1]; dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Step 6: "Resamples all data streams to 30 Hz via linear interpolation… Events are clipped to >=0 after interpolation." Step 5 Key Decision 4 justifies 30 Hz from the paper ("linearly interpolating onto a consistent set of 30hz timestamps") and from the need for a consistent bin size across the 31 Hz Scientifica and 11 Hz Multiscope rigs. The decision to use raw rather than filtered events is justified only by the paper quote about "discrete calcium events". The AI observed the consequence — 2,602 trials in the full dataset are entirely zero, and calcium events are "~0.25 % of timepoints nonzero" — and in Step 10/12 explicitly declined to treat it as a bug: "This is NOT a bug: calcium events are sparse… Sessions with few neurons (e.g., 4-6) will have many trials with no detected events."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `valid_roi == True` from the `cell_specimen_table`. If zero ROIs are valid, the whole experiment is skipped. No other neuron-level QC (no SNR, event-rate, or trace-quality threshold). In practice this filter is a no-op on this release — every ROI in the on-disk NWB files already has `valid_roi == True`, so all 29,444 ROIs in the 202 active experiments are kept.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. Step 1/Step 3 "Neuron curation rules": "`valid_roi == True` (SVM classifier output)… Exclusion reasons: union of cells, duplicate, edge/motion affected, apical dendrite, too small/narrow/dim"; Step 1 notes "`exclude_invalid_rois=True` by default" in `CellSpecimens`, so the AI reproduced the SDK default. Step 10 Check 3(b): "we filter neurons by valid_roi (same as CellSpecimens with exclude_invalid_rois=True)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`trials.start_time`). All streams are first placed on one common 30 Hz grid `regular_ts` derived from the ophys timestamps; a trial is then the set of grid bins with `start_time <= t < stop_time`. The same index array `trial_time_indices` slices neural, running, pupil, image identity and image change, so all streams are aligned by construction. Metadata records `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial   = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial  = running_resampled[trial_time_indices]
pupil_trial    = pupil_resampled[trial_time_indices]
image_idx      = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal  = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. Step 5 Key Decisions 5 & 6: "Trial window: Use trial start_time to stop_time from trials table. Variable length across trials"; "Alignment event: Trial start time (stimulus onset). off_start=0, off_end=None (variable)." The AI justified the full-window choice because the trial must contain both the pre-change flashes and the post-change response window so that image identity and image change can be time-varying. Step 3 notes the hardware sync ("all experimental clocks on a single NI PCI-6612 digital IO board at 100 kHz"), which is why cross-stream interpolation onto one grid is considered valid. Step 10 Check 5 verified "Neural/output length alignment at trial boundaries".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **33.33 ms (30 Hz), uniform across every trial and every session.** Yes — rebinning is applied: the native ophys rate (~31 Hz on the Scientifica single-plane rigs, ~10.7–11 Hz per plane on the Multiscope) is resampled by linear interpolation onto `np.arange(t_start, t_end, 1/30)`. Running speed (~60 Hz) and pupil (~30 Hz) are interpolated onto the same grid. `metadata['time_bin_size'] = 33.33`.

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled  = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
pupil_resampled   = np.interp(regular_ts, pupil_ts, pupil_area)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. Step 5 Key Decision 4: "**Resample to 30 Hz**: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms." Step 4 discrepancy table resolves "Frame rate: 31 Hz (CAM2P), 11 Hz (MESO) … 30 Hz interpolated (paper) → Will resample all to 30 Hz." Step 10 Check 3(c/d) lists this as matching the paper. It is also required by the target format ("Time bins should be the same size for all trials and sessions"), which would otherwise be violated by mixing 31 Hz and 11 Hz rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The stimulus presentation table (`intervals/Natural_Images_*_presentations`): the `image_name`, `start_time` and `omitted` columns. The trials table's `initial_image_name`/`change_image_name` are read but not used to build the output. The AI selects the stimulus table by skipping `trials`, anything containing "spontaneous", and anything containing "movie".

ii.
```python
stim_key = None
for k in f['intervals'].keys():
    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
        stim_key = k
        break
stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. Step 5 Variable Mapping: "`stimulus_presentations.image_name` → output[0]: image_identity — Map to categorical int, time-varying per ophys frame | 8 natural images". Using the presentation table rather than the trials table means the actual flash onset times drive the label, and the AI verified in Step 10 Check 2 by re-reading the NWB independently: "Image identity (session 0, trial 5, timepoint 10): expected 'im063', got 'im063'. PASS".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per 30 Hz bin, the label is the most recent **non-omitted** flash at or before that bin (`searchsorted(..., side='right') - 1`), so the identity is carried through the 500 ms grey interval and through 5 %-probability omitted flashes. Names are first mapped to a per-experiment sorted list, then remapped to a **global, sorted 16-image vocabulary** built in Pass 1 over all experiments (8 images from set A + 8 from set B). Anything not found defaults to index 0. The result is stored as `int64`.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    n_tp = len(timepoints)
    image_idx = np.zeros(n_tp, dtype=np.int64)
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
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
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']))
...
global_image_names = sorted(all_image_names_set)
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
    else:
        local_to_global[local_idx] = 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']],
                         dtype=np.int64)
```

iii. Step 5 Key Decisions 7 and 8: "Image identity during gray screen: Use the identity of the image that was just shown (last presented image)"; "Image identity for omitted flashes: Continue with previous image identity." A global vocabulary is used because "different sessions use different [image] sets" (trajectory step 73), so the decoder needs a session-independent label space; the full-run distribution is roughly uniform over the 16 codes (0.058–0.067 each).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated directly at `trial_ts = regular_ts[trial_time_indices]` — exactly the same 30 Hz bin centres used to slice the neural matrix — so it is aligned bin-for-bin by construction and has identical length.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
n_tp = len(trial_ts)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. Step 6/Step 10: alignment is guaranteed because every stream is placed on the single `regular_ts` grid before trial segmentation and then indexed with the same `trial_time_indices`. Step 10 Check 5 explicitly verified "Neural/output length alignment at trial boundaries", and Step 10 Check 2 spot-checked the image label at (session 0, trial 5, timepoint 10) against the raw NWB.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus presentation table's `is_change` flag combined with `omitted` and the flash `start_time`. Because `is_change` is True only for genuine image changes (catch trials are marked `is_sham_change`, not `is_change`), catch trials automatically get an all-zero change signal. The trials table's `is_change`/`change_time` are loaded but not used for this output.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. Step 5 Variable Mapping: "`trials.is_change` + stimulus timing → output[1]: image_change — Binary 1 at change timepoint, 0 otherwise, time-varying | 1 for one 750ms window at change" (the implementation uses the stimulus-table `is_change`). Step 12 verified the catch-trial consequence directly: "Catch trials: 0/6,515 have image_change signal (correct: catch = sham change). Go trials: 0/45,477 missing image_change signal (all have change)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length `n_tp` is created; for every change flash in the session, all trial bins falling in `[change_start, change_start + 0.75 s)` are set to 1. The 750 ms width is one full image interval (250 ms stimulus + 500 ms grey). The result is a binary `int64` time series; 7.7 % of all bins in the full dataset are 1, matching the ~1-in-13 flash change rate.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        # Mark the full image interval (stimulus + gray) as change
        # Use 750ms window from change start
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. Step 3 notes "Image presentation | 250 ms stimulus + 500 ms gray = 750 ms", and Step 6: "Image change signal: 1 during 750ms window starting at change onset." The trajectory (step 95) records the AI re-checking this choice: "the 750ms window I mark as 'change' after change onset includes both the stimulus and gray". Step 9 consistency table: "Image change fraction | ~1/13 flashes | 7.7% | Yes (~1/13=7.7%)."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — it is constructed directly as a two-class categorical variable (0 = `no_change`, 1 = `change`), declared in `output_values`.

ii.
```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
output_values = [
    global_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The decoder-output spec calls for "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0. Time-varying," so the variable is categorical by definition; the only free parameter is the width of the "right after" window, which the AI set to one 750 ms image interval (see 4-b).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: the 750 ms windows are evaluated against `trial_ts = regular_ts[trial_time_indices]`, the identical 30 Hz bins used for the neural slice, so the change flag is bin-aligned with the neural data and has the same length.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
```

iii. Same rationale as 3-c — one common 30 Hz grid and one shared index array for all streams. Verified in Step 10 Check 5 (alignment at trial boundaries) and Step 12 (change flag present in exactly the go trials, absent in exactly the catch trials).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` (cm/s, ~60 Hz) and its `timestamps`, read directly from the NWB — the same stream the SDK exposes as `dataset.running_speed`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. Step 5 Variable Mapping: "`running/speed/data` → output[2]: running_speed — Interpolate to 30 Hz, discretize into 5 percentile bins, time-varying." Step 1 identified `RunningSpeed.from_nwb()` as the SDK loader for the same HDF5 path.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. One step before discretization: `np.interp` from the ~60 Hz running timestamps onto the 30 Hz `regular_ts`. `np.interp` clamps (rather than NaN-extrapolates) outside the running timestamp range, so no NaNs are produced. No smoothing, no absolute value, no sign clipping — negative speeds (backwards wheel motion) are preserved (global bin edges run from −24.1 cm/s to +99.9 cm/s).

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
output_trials.append({..., 'running_speed': running_trial.astype(np.float32), ...})
```

iii. Step 6: "Resamples all data streams to 30 Hz via linear interpolation." Step 3 notes hardware clock synchronisation across all streams, which makes direct interpolation onto the ophys-derived grid valid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins with **globally shared edges**. Critically, the edges are computed in Pass 2 from the **raw ~60 Hz full-session running traces of all 202 experiments concatenated** — not from the trial-restricted, 30 Hz-resampled values that are actually binned. `np.digitize` against the interior edges, clipped to [0, 4], NaN → bin 0. Because the edge-fitting population differs from the binned population (whole session incl. aborted/inter-trial time, native sample weighting, and multiscope sessions counted once per plane), the realised distribution is 19.9 / 20.3 / 18.4 / 20.9 / 20.5 % rather than exactly 20 % per bin.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
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
# Pass 2 — edges from raw full-session traces
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Task spec: "Running speed, discretized into five equal percentile bins." Step 6 describes the 3-pass design explicitly so that bin edges are global: "(2) compute global percentile bins for running/pupil". Step 7 acknowledges the side-effect: "Running speed bins slightly unequal within sessions (expected, bins computed globally)", and Step 9's consistency table records "Running speed bins | 5 equal percentile | 18-21% each | Yes (equal percentile)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled onto `regular_ts` for the whole session before trial segmentation, then sliced with the same `trial_time_indices` as the neural matrix, so it is bin-aligned with the neural data by construction.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
neural_trial   = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial  = running_resampled[trial_time_indices]
```

iii. As in 2-d/3-c — a single 30 Hz grid plus a single index array guarantees alignment; Step 3 documents that all clocks are hardware-synchronised at 100 kHz, so resampling across streams is legitimate.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area**, ~30 Hz) together with `acquisition/EyeTracking/likely_blink/data`. The AI did not use the `width`/`height` datasets that are also present in the same group. If the eye-tracking group is missing, pupil is set to all-NaN for that experiment and a warning is printed.

ii.
```python
try:
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
    pupil_area = pupil_tracking['area'][()]
    pupil_ts = pupil_tracking['timestamps'][()]
    likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
```

iii. Step 5 Variable Mapping: "`EyeTracking/pupil_tracking/area` → output[3]: pupil_diameter — … | **Use pupil area as proxy for diameter**". Step 4 flagged the blink issue ahead of time: "Pupil | area with NaNs during blinks | ~9% blinks | 30 Hz camera | Need to handle NaN/blinks."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Frames flagged `likely_blink` are set to NaN (the NWB `area` field already carries NaN at exactly those frames, so this is belt-and-braces); (2) remaining NaNs are **linearly interpolated over in the native 30 Hz eye-tracking time base** by `interpolate_nans` (edge NaNs are filled with the nearest valid value; an all-NaN trace becomes all zeros); (3) `np.interp` onto the 30 Hz `regular_ts`; (4) percentile discretization (see 6-c). No smoothing and no conversion of area to a diameter/width scale.

ii.
```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if not nans.any():
        return arr.copy()
    if nans.all():
        return np.zeros_like(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

iii. Step 5 Key Decision 9: "**Pupil NaN handling**: During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data." Step 6: "Pupil blinks (likely_blink=True) set to NaN and interpolated before resampling." Interpolating in the native time base before the 30 Hz resample prevents blink artefacts from leaking into neighbouring bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same machinery as running speed: five equal-percentile bins with global edges, `np.digitize` + clip, NaN → bin 0. The edges are computed in Pass 2 from the concatenated **raw, blink-excluded, native-rate** pupil-area samples of all 202 experiments, again not from the trial-restricted resampled values that are binned. Realised distribution across the full dataset: 22.8 / 18.5 / 19.1 / 19.0 / 20.6 % (bin 0 is inflated partly because experiments with no eye tracking contribute all-NaN → bin 0). Edges in area units: [125.6, 4374, 5528, 6747, 8599, 323783].

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Task spec: "Pupil diameter, discretized into five equal percentile bins." Step 5 Key Decision 9 ("Compute percentile bins from non-blink data") and Step 9's table: "Pupil diameter bins | 5 equal percentile | 19-23% each | Yes (roughly equal)." Because percentile binning is rank-based, the AI treats area and diameter as interchangeable for this purpose.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed: resampled onto the session-wide `regular_ts` and then sliced with the same `trial_time_indices` used for the neural matrix. When pupil data are absent the trial gets an all-NaN vector of the correct length (→ all bin 0), so shapes always match.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. Same rationale as 5-d — single common grid, single index array, clocks hardware-synchronised (Step 3).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order.

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

iii. Step 5 Variable Mapping: "`trials.hit/miss/false_alarm/correct_reject` → output[4]: trial_outcome — Static per-trial categorical | 4 classes." Step 4 recorded the verification "Trial outcomes | Each valid trial has exactly 1 outcome | Confirmed", and Step 5 planned the sanity check "Trial outcome distribution: Go trials → hit or miss; Catch trials → FA or CR", which was confirmed (87.4 % go vs 12.6 % catch).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to integer codes 0 = hit, 1 = miss, 2 = false_alarm, 3 = correct_reject by an if/elif chain; a trial matching none of the four is dropped. Although the variable is static per trial, it is **broadcast to a constant row across all `n_tp` bins** so that the output block for every trial is a single `(5, n_timepoints)` array. `output_values[4] = ['hit','miss','false_alarm','correct_reject']`. Full-dataset distribution: hit 30.2 %, miss 57.2 %, FA 1.7 %, CR 10.8 %.

ii.
```python
if trials['hit'][trial_idx]:
    outcome = 0  # hit
elif trials['miss'][trial_idx]:
    outcome = 1  # miss
elif trials['false_alarm'][trial_idx]:
    outcome = 2  # false_alarm
elif trials['correct_reject'][trial_idx]:
    outcome = 3  # correct_reject
else:
    continue  # Unknown outcome, skip
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. The target format allows "(n_output, n_timepoints) or (n_output)" but the AI chose a single homogeneous `(5, n_timepoints)` block per trial (code comment: "For mixed time-varying and static, we need to handle carefully… Combined: (n_output, n_timepoints) where static is broadcast"). Step 10 Check 5 verified "Trial outcome constant within trials".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is:
- **Unreadable / malformed NWB file**: the whole `load_experiment_data` body is wrapped in `try/except Exception`; on failure it prints an error and returns `None`, and the experiment is skipped.
- **No valid ROIs**: experiment skipped with a warning.
- **No stimulus-presentation table**: experiment skipped with a warning.
- **Missing eye tracking**: caught by `except (KeyError, Exception)`; pupil becomes an all-NaN vector, which `digitize_to_bins` maps to bin 0.
- **Blink frames / NaNs in the pupil trace**: set to NaN and linearly interpolated; an all-NaN trace becomes all zeros.
- **Image names absent from the local/global vocabulary**: silently default to index 0.
- **Trials running past the end of the recording, or too short**: the `regular_ts` mask naturally truncates them, and trials with fewer than 3 bins are dropped.
- **Trials with no outcome flag**: dropped.
- **Degenerate experiments**: fewer than 2 trials before or after processing → experiment dropped.
- **Numerical**: interpolated events clipped to ≥ 0; bin indices clipped to [0, n_bins−1]; percentile edges nudged by 1e-10 if non-increasing (constant-valued streams).

ii.
```python
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None
...
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping"); return None
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found in {experiment_id}, skipping"); return None
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
...
if nans.all():
    return np.zeros_like(arr)
...
binned = np.clip(binned, 0, n_bins - 1)
binned[np.isnan(values)] = 0
...
for i in range(1, len(edges)):
    if edges[i] <= edges[i-1]:
        edges[i] = edges[i-1] + 1e-10
```

iii. Step 6: "Handle missing data appropriately… Pupil blinks (likely_blink=True) set to NaN and interpolated before resampling; Events are clipped to >=0 after interpolation." Step 5 Key Decision 9 covers the blink policy. Step 10 Check 5 reports eight edge-case checks passing, including "No NaN in neural data", "All events non-negative", "All output values in valid range", "All sessions have >= 2 trials (min: 39)". In the full run no experiment actually hit the error paths (202/202 processed).

## 9-a. What are the most time-consuming steps of the code?

i. Total full-run time was 358 s for 202 experiments. Of that, the 202 per-experiment reports in `conversion_full_out.txt` sum to only ~130 s (load ≈ 0.2–0.4 s, process ≈ 0.2–0.5 s each), so roughly **220 s — about 60 % of the runtime — is spent in Pass 1 and Pass 2**, which open and re-read every NWB file purely to collect the image-name vocabulary and the running/pupil samples for the percentile edges. Within Pass 3 the dominant costs are the full-array HDF5 reads (`event_detection/data[()]` for ~140 k × n_cells, the ~270 k-sample running trace, the ~136 k-sample pupil trace) and the per-column interpolation of the whole session's event matrix to 30 Hz. Writing the 8.3 GB pickle took 9.1 s. The AI instrumented only Pass 3 (`t_load` / `t_process` per experiment); Passes 1 and 2 are untimed, which is why its Step 7 estimate ("~2-3 minutes") undershot the actual 6 minutes by ~2×.

ii.
```python
t0 = time.time()
print(f"\n[{idx+1}/{len(exp_table)}] Processing experiment {eid}...", flush=True)
raw_data = load_experiment_data(nwb_path, eid)
t_load = time.time() - t0
result = process_single_experiment(raw_data, exp_meta)
t_process = time.time() - t0 - t_load
...
print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
```

iii. Step 6 "Code inefficiencies identified: Sequential processing of experiments (could parallelize); Multiple passes over NWB files." Step 7 gives the (under-)estimate: "Load ~0.25s, Process ~0.25s → Total ~0.5s/session → ~2 min", and Step 9 reports the actual "Conversion time: 6 minutes (358s)". Since 6 min is well under the 15-min budget set by the instructions, the AI did not optimise further, and the `ProcessPoolExecutor` it imported was never used.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four clear cases:
1. **`get_image_at_timepoints`**: after the vectorized `np.searchsorted`, a pure-Python `for i in range(n_tp)` loop does the name→index lookup. With ~52 k trials × ~254 bins this is ~13 M Python iterations. It could be one `np.where(insert_idx >= 0, codes[insert_idx], 0)` after pre-encoding `stim_names` as an integer array once per experiment.
2. **`get_image_change_at_timepoints`**: for every trial it loops over *all* change flashes of the whole session (~300–500) and builds a full boolean mask over the trial's bins each time — O(n_trials × n_changes). The change indicator could be computed once per session over `regular_ts` and then sliced with `trial_time_indices`, like every other stream.
3. **Per-trial window selection**: `trial_mask = (regular_ts >= t0) & (regular_ts < t1)` scans the entire session grid (~140 k bins) once per trial; `np.searchsorted(regular_ts, [t0, t1])` plus `slice` would be O(log n).
4. **Image-code remapping and the local→global lookup**: `local_to_global` is built with `global_image_names.index(name)` (linear scan) and applied with a Python list comprehension per trial (`[local_to_global.get(v, 0) for v in ...]`); both should be a single `np.array(lut)[image_identity]` fancy-index.

Also `interpolate_to_regular_grid` loops `np.interp` over neurons; that one is harder to avoid with `np.interp` but `scipy.interpolate.interp1d(..., axis=0)` would do it in one call.

ii.
```python
    for i in range(n_tp):                      # (1)
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
```
```python
    for cs, ce in zip(change_starts, change_stops):   # (2)
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```
```python
        trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)   # (3)
        trial_time_indices = np.where(trial_mask)[0]
```
```python
        for local_idx, name in enumerate(result['all_image_names']):               # (4)
            if name in global_image_names:
                local_to_global[local_idx] = global_image_names.index(name)
        ...
        img_id_global = np.array([local_to_global.get(v, 0)
                                  for v in trial_data_out['image_identity']], dtype=np.int64)
```
```python
        for i in range(data.shape[1]):          # per-neuron interpolation
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. Step 6 lists under "Code speedups added": "Vectorized interpolation using np.interp; Efficient searchsorted for image identity assignment" — i.e. the AI vectorized the *search* but left the subsequent element-wise lookup, the per-trial change loop and the per-trial mask scans as Python loops. Because the total runtime (6 min) came in under the 15-minute threshold in the instructions, it did not revisit them.

## 9-c. What processing does the code repeat multiple times?

i. 
- **Every NWB file is opened and read three times** (Pass 1 for image names, Pass 2 for running/pupil, Pass 3 for the conversion). The stimulus `image_name` array is read in Pass 1 and again in Pass 3; the running trace and the pupil area/blink arrays are read and NaN-masked in Pass 2 and then read and NaN-masked again in Pass 3.
- **The blink→NaN masking and the `np.isnan` filtering of pupil are duplicated** between `convert_all`'s Pass 2 and `process_single_experiment`.
- **The change-flash mask** (`is_change & ~omitted`, and the `change_starts` array) is recomputed from the full session table for every single trial.
- **`name_to_idx`** is rebuilt from scratch on every call to `get_image_at_timepoints`, i.e. once per trial.
- **Image codes are assigned twice**: once to a per-experiment local index, then remapped to the global index.
- The per-trial `regular_ts` mask re-scans the whole session grid for each trial (see 9-b).

ii.
```python
# Pass 1
with h5py.File(nwb_path, 'r') as f:
    ...  img_names = f['intervals'][k]['image_name'][()]
# Pass 2
with h5py.File(nwb_path, 'r') as f:
    running = f['processing']['running']['speed']['data'][()]
    pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
    blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
    pupil_area[blink] = np.nan
# Pass 3 — reads all of the same arrays again
raw_data = load_experiment_data(nwb_path, eid)
...
    pupil_area[likely_blink] = np.nan
```
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    ...
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}   # rebuilt per trial
```

iii. Step 6 acknowledges the design: "3-pass approach: (1) collect global image names, (2) compute global percentile bins for running/pupil, (3) process experiments" and lists "Multiple passes over NWB files" under "Code inefficiencies identified". The multi-pass structure is genuinely needed for *global* percentile edges and a *global* image vocabulary, but the repeated I/O could have been avoided by caching the Pass-2/Pass-3 arrays (or by collecting them during a single pass and deferring only the discretization, which is what the reference does).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. 
- **`valid_roi` filtering is a no-op** on this data release: every ROI in all 284 on-disk NWB files has `valid_roi == True` (checked directly), so the mask, the `n_valid == 0` guard and the boolean column indexing never remove anything.
- **The entire session is resampled to 30 Hz before trial segmentation** — events, running and pupil — but only the go/catch trial windows are kept. Aborted trials make up a large fraction of the timeline in many sessions (e.g. 735 trials of which only 215 are go/catch), so a substantial part of the interpolated arrays is thrown away. The same applies to `interpolate_nans` over the full pupil trace.
- **Values read but never used**: `cell_specimen_ids` (loaded, masked and returned, never consumed), `trials['is_change']` (the change output is built from the stimulus table instead), `stim['stop_time']` (unpacked as `ce` in `get_image_change_at_timepoints` and then ignored in favour of `cs + 0.75`), and `trials['change_time']`/`initial_image_name`/`change_image_name` (loaded and byte-decoded but unused).
- **Dead imports / flags**: `sys`, `ProcessPoolExecutor`, `as_completed` are imported and never used; `--full` is defined as `default=True` and is only ever negated by `--sample`, so it has no effect.
- **Redundant NaN handling on pupil**: the NWB `area` field is already NaN at exactly the `likely_blink` frames, so `pupil_area[likely_blink] = np.nan` changes nothing.
- **The static trial outcome is broadcast to every one of the ~254 bins** of each trial, inflating the stored output by ~254× for that row (a deliberate format choice, but the extra copies carry no information).
- **`compute_percentile_bins` collects every raw running/pupil sample of every session into memory** (tens of millions of values) to compute six edge values, and those samples include the non-trial time that is never binned.

ii.
```python
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed   # never used
...
    'cell_specimen_ids': cell_specimen_ids[valid_roi],             # never consumed
    'is_change': trials_grp['is_change'][()].astype(bool),         # never consumed
...
    for cs, ce in zip(change_starts, change_stops):                # ce unused
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
...
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)  # whole session
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
...
parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
```

iii. The AI documented none of these as waste. It listed `valid_roi` as an active curation step (Step 5 Key Decision 10, Step 10 Check 3b: "we filter neurons by valid_roi (same as CellSpecimens with exclude_invalid_rois=True)") rather than noting it is inert on this release; whole-session resampling was chosen for simplicity so that all streams share one grid before segmentation; and the outcome broadcast is justified by the decision to emit a uniform `(5, n_timepoints)` output block per trial. Since the full conversion finished in 6 minutes — inside the instructions' 15-minute budget — the AI stopped optimising after Step 7.
