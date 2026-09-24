# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK `VisualBehaviorOphysProjectCache`. Instead it reads the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) directly, intersects it with the set of NWB files actually present on disk in `behavior_ophys_experiments/`, and loads each experiment individually with `BehaviorOphysExperiment.from_nwb_path()`. Two filters are applied at selection time: (1) an NWB file must exist, and (2) `passive == False` (active behavior sessions only). **No `project_code` filter is applied**, so both `VisualBehavior` (single-plane, 31 Hz; 168 experiments) and `VisualBehaviorMultiscope` (multi-plane, 11 Hz; 34 experiments) are included — 202 experiments, of which 200 were processed successfully (2 dropped for having < 5 neurons). The unit of iteration is the *experiment* (one imaging plane), and each experiment becomes one "session" in the output.

ii.
```python
NWB_DIR = '/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/project_metadata/'

def load_experiment_table():
    """Load and filter the experiment table to active behavior sessions."""
    exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get available NWB files
    nwb_ids = set()
    for f in os.listdir(NWB_DIR):
        if f.endswith('.nwb'):
            eid = int(f.replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
            nwb_ids.add(eid)

    # Filter: have NWB file + active behavior only
    exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
    exp = exp[exp['passive'] == False]
    return exp
```
```python
    for i, (_, row) in enumerate(exp_table.iterrows()):
        result = process_experiment(row, sample_mode=args.sample)
        if result is not None:
            results.append(result)
```
```python
    eid = exp_row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. From the trajectory (steps 24–38) and `CONVERSION_NOTES.md`: the AI first tried to reproduce the paper's exact cohort (familiar, active, multiscope sessions) and found that only 22 such experiments from a *single* mouse were available locally. It judged this insufficient for a decoder ("we need data from multiple mice"), and read the instruction "Collect and convert data under the 'Visual Behavior' task" as referring to the change-detection task generally rather than to the `VisualBehavior` project code: *"The task says 'Collect and convert data under the Visual Behavior task' - this likely means ALL Visual Behavior experiments."* It therefore included every active experiment from both project codes, planning to reconcile the 31 Hz / 11 Hz frame-rate difference by resampling to a common bin size (see 2-e). Passive sessions were excluded because the lick spout is retracted, so there is no behavioral report to decode. Loading directly from NWB paths (rather than via the S3 cache) was chosen because the data are present locally as bare NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the processed experiments, read from each NWB's `dataset.metadata['mouse_id']` and stored as strings. The sorted unique list becomes `subjects`; each session's index into it is stored in `subject_idx`. Result: 38 mice (37 single-plane + 1 multiscope mouse, 457841).

ii.
```python
    mouse_id = str(meta['mouse_id'])
...
    all_subjects = sorted(set(r['mouse_id'] for r in results))
...
        subject_idx.append(all_subjects.index(r['mouse_id']))
...
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. No explicit discussion in the trajectory beyond counting mice for cohort-size decisions ("168 active experiments ... 37 mice", "200 sessions from 38 mice across 3 cell types"). `mouse_id` is the canonical animal identifier in the SDK metadata, so the AI used it without further comment.

## 1-c. How are the data split into sessions?

i. **Each ophys *experiment* (i.e. each imaging plane / NWB file) is treated as one "session"** in the output structure. The AI never groups experiments by `ophys_session_id`. For single-plane `VisualBehavior` data this is equivalent to a session (1 plane per session), but for the 34 `VisualBehaviorMultiscope` experiments, the up-to-8 planes recorded simultaneously in one physical session are emitted as up-to-8 separate "sessions" that share identical behavior, stimuli and trial timing. This is visible in the output: mouse 457841 is credited with 32 sessions, in blocks of 7/7/3/7/4/4 sessions with identical trial counts (209, 209, 209, 209, 209, 209, 209, 309, 309, ...). Total: 200 sessions.

ii.
```python
    for i, (_, row) in enumerate(exp_table.iterrows()):
        print(f"\nProcessing experiment {i+1}/{len(exp_table)}: {row['ophys_experiment_id']}")
        result = process_experiment(row, sample_mode=args.sample)
        if result is not None:
            results.append(result)
...
    for r in results:
        session_neural = []
        ...
        neural_all.append(session_neural)
        output_all.append(session_output)
```
(`results` has exactly one entry per NWB file; `neural_all` therefore has one entry per imaging plane.)

iii. The AI was aware of the plane/session distinction — it read the whitepaper passage stating "For multi-plane imaging experiments, there can be up to 8 imaging planes (8 experiments) per session", and it recorded the consequence in `CONVERSION_NOTES.md` under **Known Limitations**: *"The multiplane experiments (mouse 457841) produce many sessions with few neurons per plane, since each imaging plane is a separate experiment."* Its reasoning (step 29) was that "Each NWB file represents a separate session with different neurons", i.e. it accepted plane-as-session as a workable unit rather than merging planes. No justification is given for why merging was rejected.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's built-in `dataset.trials` table. A trial spans `start_time` → `stop_time` (variable length, ~7–12.5 s, mean 8.5 s). Trials are kept if they are Go **or** Catch and are neither Aborted nor Auto-rewarded. Neural frames are those with `t_start <= ophys_ts < t_end`, and a regular 90.9 ms bin grid is laid down from `t_start` to `t_end` (see 2-e).

ii.
```python
    trials = dataset.trials

    # Filter trials: Go or Catch, not Aborted, not Auto-rewarded
    valid_trials = trials[
        ((trials['go'] == True) | (trials['catch'] == True)) &
        (trials['aborted'] == False) &
        (trials['auto_rewarded'] == False)
    ]
```
```python
    for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
        t_start = trial['start_time']
        t_end = trial['stop_time']
        ...
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
        ...
        trial_ophys_ts = ophys_ts[frame_mask]
        trial_neural_raw = neural_data[:, frame_mask]
```

iii. Directly from the instruction ("Segment each recording session into individual trials based on how they are defined in the experiment. Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"). `CONVERSION_NOTES.md`: *"Excluded: Aborted trials (premature licking before stimulus change) and Auto-rewarded trials (free reward trials). This matches the standard analysis approach described in the whitepaper where aborted trials are excluded from performance metrics."* The full `start_time`→`stop_time` window (rather than a fixed window around the change) was used so that the trial contains both pre-change flashes and the response window.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the Go/Catch/not-aborted/not-auto-rewarded selection, the following filters are applied:
- Trials with NaN `start_time`/`stop_time`, or `stop_time <= start_time`, are skipped.
- Trials with fewer than 3 ophys frames in the window (`MIN_TRIAL_FRAMES = 3`) are skipped, and again if fewer than 3 resampled bins result.
- Trials whose outcome is none of hit/miss/false_alarm/correct_reject (`outcome == -1`) are dropped.
- Experiments with fewer than 2 pre-filter valid trials, fewer than 2 post-processing trials, or fewer than `MIN_NEURONS = 5` neurons are dropped entirely (2 experiments dropped for the neuron criterion).
- Whole experiments that fail to load or lack dff/stimulus/trials tables are dropped.
- Passive sessions were never loaded (see 1-a).

ii.
```python
    if len(valid_trials) < 2:
        print(f"  SKIP {eid}: only {len(valid_trials)} valid trials")
        return None
...
        if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
            continue
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
        if frame_mask.sum() < MIN_TRIAL_FRAMES:
            continue
...
        if neural_resampled is None or len(bin_centers) < MIN_TRIAL_FRAMES:
            continue
...
        outcome = get_trial_outcome(trial)
        if outcome == -1:
            continue
...
    if trial_count < 2:
        print(f"  SKIP {eid}: only {trial_count} processed trials")
        return None
```

iii. The 2-trial minimum is required by the instructions ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). `MIN_NEURONS = 5` and `MIN_TRIAL_FRAMES = 3` are the AI's own guards; `CONVERSION_NOTES.md` lists "Some sessions have very few neurons (minimum 5), which may limit decoder performance" as a known limitation, indicating the threshold was set to avoid degenerate, near-empty sessions while keeping as much data as possible. Dropping outcome `-1` trials keeps the trial-outcome output variable strictly within its four declared categories.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.dff_traces` — the per-cell dF/F calcium traces from the AllenSDK pipeline, stacked into an `(n_neurons, n_frames)` array. The paper's `events` (detected calcium events) were tried first and then rejected.

ii.
```python
    # Get neural data (dF/F traces - continuous fluorescence measure)
    try:
        dff_df = dataset.dff_traces
        neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
    except Exception as e:
        print(f"  ERROR getting dff_traces for {eid}: {e}")
        return None
```
Metadata records the choice:
```python
            'neural_data_type': 'dF/F traces (normalized change in fluorescence)',
```

iii. The AI initially wrote the pipeline with `events` "matching the paper methodology" (step 38), ran it, and saw `train_decoder.py` warn about all-zero neural trials. It then measured sparsity (step 47): *"Events are 99.7% sparse! That's why we get all-zero trials. DFF is continuous and more suitable for decoding ... The decoder needs 'Neural activity data (e.g., firing rates, spike counts)'. DFF traces are neural activity. The paper used events. But events being 99.7% sparse causes decoder issues."* It invoked the instructions' escape clause ("Discrepancies are only allowed if required by the Decoder Input and Decoder Output specifications") and switched to dF/F. `CONVERSION_NOTES.md` repeats this rationale. Note the module docstring was never updated and still (incorrectly) claims "Neural data: detected calcium events (matching paper methodology)".

## 2-b. How is the `neural` data processed?

i. Two operations: (1) the per-cell dF/F arrays are stacked into one `(n_neurons, n_frames)` matrix (each session = one imaging plane, so there is no cross-plane merging); (2) the trial-windowed traces are **temporally rebinned** to a common 90.9 ms grid by averaging all ophys frames whose timestamps fall inside each bin (nearest-neighbour copy if a bin is empty). No normalisation, smoothing, z-scoring, baseline subtraction or neuron-level selection is applied. Output is cast to `float32`.

ii.
```python
def resample_to_common_bins(timestamps, data, trial_start, trial_end):
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
    if len(bin_centers) == 0:
        return None, None
    ...
    for b, tc in enumerate(bin_centers):
        t_lo = tc - COMMON_BIN_SIZE / 2
        t_hi = tc + COMMON_BIN_SIZE / 2
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
        else:
            # Nearest neighbor for sparse data
            idx = np.argmin(np.abs(timestamps - tc))
            resampled[:, b] = data[:, idx]
    return resampled, bin_centers
```
```python
        neural_resampled, bin_centers = resample_to_common_bins(
            trial_ophys_ts, trial_neural_raw, t_start, t_end
        )
...
            session_neural.append(neural.astype(np.float32))
```

iii. `CONVERSION_NOTES.md`: *"dF/F traces are already processed by the Allen pipeline: motion corrected, cell segmented, ROI filtered, demixed, neuropil subtracted, baseline normalized, and detrended (as described in whitepaper Section F). No additional neural processing was applied."* The averaging step exists only to reach a common bin size across the 31 Hz and 11 Hz rigs (see 2-e).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control is applied — every cell in `dff_traces` is kept, relying on the Allen pipeline's own segmentation/QC. The only neural-related curation is at the experiment level: experiments with fewer than 5 neurons are discarded (2 of 202).

ii.
```python
    n_neurons = neural_data.shape[0]
    if n_neurons < MIN_NEURONS:
        print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
        return None
```

iii. Same rationale as 2-b: the SDK's `dff_traces` are already the QC-passed, valid cell-specimen ROIs, so no extra filtering was considered necessary. The 5-neuron floor is an AI-added guard against sessions that would contribute almost no decodable signal (listed under Known Limitations in `CONVERSION_NOTES.md`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`trials.start_time`). The ophys frames of the trial are selected by `t_start <= ophys_ts < t_end`, and the resampling grid starts at `t_start + bin/2` and steps by 90.9 ms. Every stream (neural, image identity, image change, running, pupil) is evaluated on that same `bin_centers` vector, guaranteeing alignment across streams. Metadata declares `temporal_alignment_event = 'Trial start time (onset of first stimulus in trial)'`, `off_start = 0.0`, `off_end = None` (variable duration).

ii.
```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
        ...
        trial_ophys_ts = ophys_ts[frame_mask]
        trial_neural_raw = neural_data[:, frame_mask]
        neural_resampled, bin_centers = resample_to_common_bins(
            trial_ophys_ts, trial_neural_raw, t_start, t_end)
        ...
        img_identity = get_image_identity_at_times(stim_cd, bin_centers, {...})
        img_change   = get_image_change_at_times(stim_cd, bin_centers)
        run_at_bins  = interpolate_to_bins(running_ts, running_speed, bin_centers)
        pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```
```python
            'temporal_alignment_event': 'Trial start time (onset of first stimulus in trial)',
            'off_start': 0.0,
            'off_end': None,  # variable trial duration
```

iii. The instruction only says "Temporally align based on ophys timestamp" and to segment by experimentally defined trials; the AI used the trials table's own `start_time`/`stop_time` so that each trial contains the pre-change flashes and the post-change response window (needed for the time-varying image-identity and image-change outputs). Behavioural streams are brought onto the same bin centres by interpolation rather than the neural data being moved, keeping everything on the ophys clock (steps 33, 38).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — all data are rebinned to a common 1/11 s = 90.9 ms bin.** Because both project codes are included, the raw rates differ (single-plane VivoScope 31 Hz; Multiscope 11 Hz per plane). The AI chose the slower rate as the common grid, so the 168 single-plane experiments (83% of the data) are downsampled ~3× by averaging ~3 ophys frames per bin, while the 11 Hz data are essentially kept (one frame per bin, nearest-neighbour when a bin is empty). Resulting trials are 77–138 bins (mean 93).

ii.
```python
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
...
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
...
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
...
            'time_bin_size': COMMON_BIN_SIZE * 1000,  # in ms
```

iii. Step 35 reasoning: *"The practical approach is to resample everything to 11Hz to match the paper's resolution and ensure uniform bin sizes across all data ... Upsampling the 11Hz multiplane data would intro[duce artifacts]"*, i.e. downsampling was preferred over upsampling because upsampling would fabricate resolution. `CONVERSION_NOTES.md` states: *"Common Time Bin Size: 90.9 ms (~11 Hz) ... This matches the frame rate of the multiplane mesoscope (11 Hz per plane). For single-plane (31 Hz) experiments, neural data was averaged within each time bin"*, and lists as a limitation: *"The common bin size of 90.9 ms means some temporal resolution is lost for the 31 Hz single-plane data."* The driver is the format requirement that "Time bins should be the same size for all trials and sessions", which became binding only because both rigs were included.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `dataset.stimulus_presentations` — specifically the `image_name` and `start_time` columns of the change-detection stimulus block (rows with `image_name == 'omitted'` are excluded). The trials table's `initial_image_name`/`change_image_name` are **not** used.

ii.
```python
        stim = dataset.stimulus_presentations
        if 'stimulus_block_name' in stim.columns:
            stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
        else:
            stim_cd = stim[stim['image_name'] != 'spontaneous']
```
```python
def get_image_identity_at_times(stim_presentations, bin_centers, image_to_idx):
    stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
    ...
    stim_times = stim['start_time'].values
    stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
    result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
    return result.astype(int)
```

iii. `CONVERSION_NOTES.md`: *"At each time bin, the most recently presented non-omitted image is recorded. During grey screen periods (inter-stimulus intervals), the last shown image persists. The task description says 'of the image presented during the non-grey screen' — since trials only occur during change detection blocks, images are always being presented."* The stimulus table was preferred because it is the flash-by-flash ground truth and handles the 5% omitted flashes explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per experiment, a sorted local list of unique non-omitted image names is built and used to code each bin. At assembly time these local codes are remapped, via a vectorised lookup array, to a **global** sorted code space built from the union of image names across all 200 sessions (16 images, `im000 … im106`). The result is stored as row 0 of the `(5, n_bins)` int64 output array, and the global name list becomes `output_values[0]`.

ii.
```python
    image_names = sorted(stim_cd[
        (stim_cd['image_name'] != 'omitted') &
        (stim_cd['image_name'].notna())
    ]['image_name'].unique().tolist())
...
    all_image_names = set()
    for r in results:
        all_image_names.update(r['image_names'])
    all_image_names = sorted(all_image_names)
    global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}
...
        local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
...
            img_id_global = local_to_global_arr[out['image_identity']]
            output_data[0, :] = img_id_global        # image identity
```

iii. No explicit narrative beyond the need for consistent categorical codes across sessions; the sorted global mapping makes codes deterministic and lets `output_values` name each category. The local→global remap was also vectorised specifically for speed (step 136) after the assembly stage was found to be slow.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on exactly the same `bin_centers` vector used to rebin the neural data, using a `searchsorted` "most recent flash onset at or before this bin centre" rule, so image identity and neural activity share the same time axis by construction.

ii.
```python
        img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                                    {name: i for i, name in enumerate(image_names)})
...
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
```
```python
            assert out.shape[1] == neural_all[si][ti].shape[1], \
                f"Session {si}, trial {ti}: output time {out.shape[1]} != neural time {neural_all[si][ti].shape[1]}"
```

iii. Implicit: all output streams are functions of `bin_centers`, which is derived from the trial window and shared with the neural resampling, and a post-hoc assertion verifies identical time lengths. Bins that precede every stimulus onset fall back to code 0.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` boolean column and `start_time` of `stimulus_presentations` (change-detection block). The trials table's `change_time`/`go` columns are not used.

ii.
```python
def get_image_change_at_times(stim_presentations, bin_centers):
    result = np.zeros(len(bin_centers), dtype=int)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    if len(changes) == 0:
        return result
    change_times = changes['start_time'].values
    ...
```

iii. `CONVERSION_NOTES.md`: *"Changes identified from `is_change` flag in stimulus_presentations table."* `is_change` is the SDK's canonical per-flash marker of an actual image-identity change, so on catch trials (sham change, same image re-presented) it is False and the indicator stays 0 throughout — consistent with the instruction that the variable marks "a change in image identity".

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change flash onset `ct` found anywhere in the session's stimulus table, all bins whose centre falls in `[ct, ct + 2 × 90.9 ms)` are set to 1; all other bins are 0. In practice each trial window contains at most one change, so exactly two bins (~182 ms) are marked per Go trial and zero bins per Catch trial. The result is row 1 of the output array, with `output_values[1] = ['no_change', 'change']`. Across the dataset, 1.9% of bins are labelled `change`.

ii.
```python
    for ct in change_times:
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
        result[mask] = 1
    return result
```
```python
            output_data[1, :] = out['image_change']   # image change
```

iii. `CONVERSION_NOTES.md`: *"Value of 1 at time bins immediately after a change in image identity (within 2 bin widths of change onset). Value of 0 otherwise"*, and *"~2% of time bins contain image changes, consistent with the task design"*. This is a literal reading of the instruction "Have value of 1 right after a change in image identity, otherwise 0" — the marker is a brief transient rather than a sustained post-change epoch.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The variable is natively binary, so the only "threshold" is the temporal one: the 2-bin (~182 ms) window after each change onset defines class 1, everything else class 0. There is no amplitude threshold. Note the window (182 ms) is shorter than the 250 ms stimulus flash itself, and is defined in units of the chosen bin size rather than in units of the stimulus cycle.

ii.
```python
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
        result[mask] = 1
```
```python
        ['no_change', 'change'],                           # image change values
```
```python
            assert np.all((out[1] == 0) | (out[1] == 1)), \
                f"Session {si}, trial {ti}: image change not binary"
```

iii. See 4-b — "right after a change" was interpreted as the immediately following one-to-two samples. No alternative window lengths are discussed in the trajectory.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: the indicator is computed directly on the shared `bin_centers` vector, so it is aligned to the rebinned neural data by construction, with the change onset landing in the bin whose centre first exceeds `ct`.

ii.
```python
        img_change = get_image_change_at_times(stim_cd, bin_centers)
...
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
```

iii. As in 3-c — every output stream is a function of `bin_centers`, the same grid on which the neural data were averaged.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, using its `timestamps` and `speed` columns (the SDK's default 10 Hz low-pass-filtered wheel-encoder speed in cm/s, natively ~60 Hz). If the attribute raises, the session's running trace is treated as absent.

ii.
```python
    try:
        running = dataset.running_speed
        running_ts = running['timestamps'].values
        running_speed = running['speed'].values
    except Exception as e:
        print(f"  WARNING: no running speed for {eid}: {e}")
        running_ts = None
        running_speed = None
```

iii. `CONVERSION_NOTES.md`: *"Running speed from wheel encoder, already processed by Allen pipeline (10 Hz low-pass filtered)."* The AI explicitly noted during exploration that `running_speed` is the filtered stream and `raw_running_speed` the unfiltered one, and took the standard filtered stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the raw speed onto the trial's `bin_centers` (`fill_value='extrapolate'`, NaN samples dropped first), then discretisation into 5 percentile bins whose edges are computed **once, globally**, over all interpolated running values from all trials of all sessions. Sessions without running data get a vector of zeros. Edges found: `[-inf, -0.0021, 0.353, 16.35, 33.68, inf]` cm/s; the resulting class fractions are 0.200 each.

ii.
```python
def interpolate_to_bins(timestamps, values, bin_centers):
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.full(len(bin_centers), np.nan)
    f = interpolate.interp1d(
        timestamps[valid], values[valid],
        kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(bin_centers)
```
```python
    all_running_flat = np.concatenate(all_running)
    running_edges = discretize_continuous(all_running_flat, n_bins=5)
...
            run_disc = apply_discretization(out['running_speed_raw'], running_edges)
            output_data[2, :] = run_disc              # running speed (discretized)
```

iii. `CONVERSION_NOTES.md`: *"Interpolated to ophys time bins. Discretized into 5 equal percentile bins across all sessions."* Global (rather than per-session) percentiles were used so the five categories mean the same thing in every session, and percentile binning satisfies the instruction's "five equal percentile bins" while giving a balanced class distribution for the decoder.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. `np.percentile` at 0/20/40/60/80/100 over all non-NaN pooled values; the outer two edges are then replaced by `±inf` so every value falls inside the range, and `np.digitize` against the 4 interior edges yields labels 0–4 (clipped defensively). Category names encode the numeric edges, e.g. `speed_bin2_0.4_16.3`.

ii.
```python
def discretize_continuous(all_values, n_bins=5):
    valid = all_values[~np.isnan(all_values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    # Ensure unique edges
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_discretization(values, edges):
    result = np.digitize(values, edges[1:-1])  # bins 0 to n_bins-1
    result = np.clip(result, 0, len(edges) - 2)
    return result
```

iii. Straight implementation of "discretized into five equal percentile bins"; the `±inf` substitution and the `clip` are guards so that no value (including extrapolated extremes) can fall outside the 5 declared categories, which the end-of-run sanity check then asserts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the shared `bin_centers`, i.e. the same grid the neural data are averaged onto, so the two streams are aligned sample-for-sample. Discretisation is applied afterwards and does not change alignment.

ii.
```python
        if running_ts is not None and running_speed is not None:
            run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
        else:
            run_at_bins = np.zeros(n_bins)
```

iii. Same rationale as 2-d/3-c: all streams are evaluated as functions of `bin_centers`, and the eye/running/ophys clocks are hardware-synchronised in the SDK, so interpolation onto the ophys-derived grid is valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`: pupil diameter is defined as the **mean of `pupil_width` and `pupil_height`** (the two axes of the fitted pupil ellipse, in pixels). Frames flagged `likely_blink` are set to NaN. If `eye_tracking` is unavailable the session's pupil trace is absent.

ii.
```python
    try:
        eye = dataset.eye_tracking
        eye_ts = eye['timestamps'].values
        # Compute pupil diameter as mean of width and height
        pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
        # Set likely blinks to NaN
        likely_blink = eye['likely_blink'].values
        pupil_diam[likely_blink] = np.nan
    except Exception as e:
        print(f"  WARNING: no eye tracking for {eid}: {e}")
        eye_ts = None
        pupil_diam = None
```

iii. `CONVERSION_NOTES.md`: *"Computed as mean of pupil_width and pupil_height from eye tracking ellipse fits. Frames marked as likely_blink set to NaN and interpolated."* During exploration the AI noted that `likely_blink` "indicates outliers or failed fits", so those samples are removed rather than trusted; averaging the two ellipse axes is used as a more symmetric estimate of diameter than either axis alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples → NaN; linear interpolation onto `bin_centers` with the NaN samples excluded (so blinks are bridged by interpolation, not propagated); then any remaining NaN bins (sessions with no eye tracking at all, and bins outside the tracked range) are filled with the **global median pupil value** computed once over all sessions; then discretisation into 5 global percentile bins. Edges: `[-inf, 35.47, 40.32, 44.75, 50.74, inf]` px. The median-fill slightly inflates the middle class (fractions 0.196/0.196/0.214/0.196/0.196).

ii.
```python
    pupil_fill_value = np.nanmedian(all_pupil_flat)
...
            pupil_raw = out['pupil_diam_raw'].copy()
            nan_mask = np.isnan(pupil_raw)
            pupil_raw[nan_mask] = pupil_fill_value
            pupil_disc = apply_discretization(pupil_raw, pupil_edges)
            output_data[3, :] = pupil_disc            # pupil diameter (discretized)
```

iii. `CONVERSION_NOTES.md`: *"Sessions without eye tracking data filled with global median"* and, under Known Limitations, *"3 sessions had eye tracking errors - pupil data filled with global median for those sessions."* Median filling was chosen as the least-distorting constant. (The `nanmedian` call was originally inside the 51 k-iteration trial loop and was hoisted out after the AI diagnosed it as the assembly-stage bottleneck — steps 131–133.)

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: global 0/20/40/60/80/100 percentiles of all non-NaN pooled pupil values, outer edges replaced with `±inf`, `np.digitize` → labels 0–4, names encoding the edges (e.g. `pupil_bin3_44.7_50.7`).

ii.
```python
    pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)
...
        [f"pupil_{pupil_bin_labels[i]}" for i in range(5)],   # pupil diameter bin labels
```
```python
            assert np.all(out[3] >= 0) and np.all(out[3] < 5), \
                f"Session {si}, trial {ti}: pupil diameter bin out of range"
```

iii. As for running speed — the instruction specifies five equal percentile bins, and pooling across sessions keeps the categories comparable between sessions and mice.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same `bin_centers` as the neural data, so alignment is exact by construction; blink gaps are interpolated across and out-of-range bins are median-filled after the fact.

ii.
```python
        if eye_ts is not None and pupil_diam is not None:
            pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
        else:
            pupil_at_bins = np.full(n_bins, np.nan)
```

iii. Same as 5-d: eye-tracking timestamps live on the same synchronised clock as the ophys frames, so a single interpolation onto the trial bin grid aligns pupil with neural activity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order and mapped to codes 0–3. Anything matching none of them returns `-1` and the trial is dropped.

ii.
```python
def get_trial_outcome(trial):
    """Get trial outcome as categorical index."""
    if trial['hit']:
        return 0  # hit
    elif trial['miss']:
        return 1  # miss
    elif trial['false_alarm']:
        return 2  # false_alarm
    elif trial['correct_reject']:
        return 3  # correct_reject
    else:
        return -1  # unknown
```

iii. `CONVERSION_NOTES.md`: *"4 categories: hit (correct Go response), miss (no response to Go), false_alarm (response to Catch), correct_reject (no response to Catch). Assigned from trial flags in the trials table."* These are the SDK's canonical change-detection outcome labels and exhaust the Go/Catch space once aborted and auto-rewarded trials are removed.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast across every time bin of the trial (row 4 of the `(5, n_bins)` output), making it a constant time series rather than a scalar — satisfying the instruction's preference for time-varying outputs while remaining static per trial. `output_values[4] = ['hit','miss','false_alarm','correct_reject']`. Observed distribution over the full dataset: hit 0.303, miss 0.572, false_alarm 0.017, correct_reject 0.108.

ii.
```python
            output_data[4, :] = out['trial_outcome']  # trial outcome (static, same value each bin)
...
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
            assert np.all(out[4] >= 0) and np.all(out[4] < 4), \
                f"Session {si}, trial {ti}: trial outcome out of range"
```

iii. `CONVERSION_NOTES.md`: *"Broadcast to all time bins within the trial."* No further processing; dropping `-1` trials guarantees the four declared categories are exhaustive, which the sanity check asserts. (Note: the outcome percentages quoted in `CONVERSION_NOTES.md` — "Hit ~45%, Miss ~42%" — are stale sample-mode numbers and disagree with the full-run verification output.)

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered defensive handling:
- Whole-experiment failures: `BehaviorOphysExperiment.from_nwb_path`, `dff_traces`, `stimulus_presentations` and `trials` are each wrapped in `try/except`; failure returns `None` and the experiment is skipped rather than crashing the run.
- Missing running speed → zeros for the whole session; missing eye tracking → NaN, later filled with the global median pupil value (3 sessions).
- Blink frames → NaN, bridged by interpolation; interpolation with fewer than 2 valid samples returns all-NaN.
- Out-of-range interpolation: running extrapolates, pupil NaNs are median-filled; `np.digitize` results are `np.clip`ped so no out-of-range category can ever be emitted.
- NaN/invalid trial times, trials shorter than 3 frames/bins, and trials with no recognised outcome are skipped; experiments with <5 neurons or <2 usable trials are dropped.
- A `stimulus_block_name` fallback exists for NWBs lacking that column.
- Five assertions after assembly re-check neuron-count consistency, neural/output time-length equality, brain-region index lengths, category ranges and subject indices.

ii.
```python
    try:
        dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
    except Exception as e:
        print(f"  ERROR loading {eid}: {e}")
        return None
...
        if 'stimulus_block_name' in stim.columns:
            stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
        else:
            stim_cd = stim[stim['image_name'] != 'spontaneous']
...
        if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
            continue
...
            pupil_raw[nan_mask] = pupil_fill_value
...
    result = np.clip(result, 0, len(edges) - 2)
```

iii. The AI's stated aim (`CONVERSION_NOTES.md`, "Sanity Checks") was that a single bad experiment must not abort a 200-experiment run, and that every emitted categorical label must be inside its declared value list so that `train_decoder.py` reports no errors or warnings — which the final `verification_full_out.txt` confirms ("Data format is valid, no errors or warnings").

## 9-a. What are the most time-consuming steps of the code?

i. Measured at roughly 3–4 s per experiment (~12 min for 202 experiments). The dominant costs, in order:
1. `BehaviorOphysExperiment.from_nwb_path()` plus materialising `dff_traces`, `stimulus_presentations`, `eye_tracking` — I/O and HDF5 decoding.
2. `resample_to_common_bins`, a Python loop over every time bin of every trial (≈ 4.8 M bin iterations overall), each doing a boolean mask over the trial's frame timestamps.
3. The per-trial calls to `get_image_identity_at_times` / `get_image_change_at_times`, which each re-filter and re-sort the *whole session's* stimulus table (~4–5 k rows) for every one of the session's 200–400 trials.
4. Pickling and writing the 3.2 GB output file, and the final assertion sweep over all 51 557 trials.
Historically the single worst step was `np.nanmedian(all_pupil_flat)` being called inside the 51 k-trial assembly loop; the AI diagnosed and hoisted it out.

ii.
```python
    for b, tc in enumerate(bin_centers):
        t_lo = tc - COMMON_BIN_SIZE / 2
        t_hi = tc + COMMON_BIN_SIZE / 2
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
```
```python
    # Precompute median pupil value for NaN filling (avoid recomputing in loop)
    pupil_fill_value = np.nanmedian(all_pupil_flat)
```

iii. Step 85: *"each NWB file takes about 30s to load and process. With 202 files, that's about 100 minutes"* (before optimisation); step 131: *"I see the issue! The `np.nanmedian(all_pupil_flat)` is being called inside the per-trial loop. With 51k trials ... that's the bottleneck!"*; step 100 after the fixes: *"it's running and processing about 1 experiment every ~3-4 seconds."*

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain un-vectorised:
- `resample_to_common_bins`: the per-bin loop could be replaced with a single `np.searchsorted`/`np.add.reduceat` (or `scipy.stats.binned_statistic`) over the trial's frames, computing all bins at once for all neurons.
- `get_image_change_at_times`: loops over every change time in the session and rebuilds a full boolean mask over `bin_centers` for each; a `searchsorted` of the change times against `bin_centers` would mark the windows in one pass, and the change list should be restricted to the trial window first.
- `frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)` scans the full session timestamp vector for every trial; `np.searchsorted` on the sorted timestamps (as the reference does) would make this O(log T).
Additionally, `neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])` uses `iterrows()` where `np.vstack(dff_df.dff.values)` would be faster, and `all_subjects.index(...)` is a linear scan inside a loop.
The AI did vectorise two loops during development: the original per-bin `get_image_identity_at_times` (replaced by `searchsorted`) and the per-trial image-code remap (replaced by an index array).

ii. Current code:
```python
    for b, tc in enumerate(bin_centers):
        ...
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
```
```python
    for ct in change_times:
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
        result[mask] = 1
```
Already vectorised by the AI:
```python
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
    result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
...
        local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])
        ...
            img_id_global = local_to_global_arr[out['image_identity']]
```

iii. Step 94: *"The `get_image_identity_at_times` function iterates per bin which is very slow. Let me vectorize it."* Step 135: *"Also, let me vectorize the image identity remapping."* The remaining loops were never revisited once throughput reached ~3 s per experiment, which the AI judged acceptable.

## 9-c. What processing does the code repeat multiple times?

i.
- The change-detection stimulus table is re-filtered (`!= 'omitted'`) and re-sorted, and the `image_name → local index` dictionary is rebuilt, **once per trial** instead of once per experiment; likewise `stim_presentations['is_change'] == True` is re-evaluated per trial.
- `frame_mask` rescans the whole `ophys_timestamps` array for every trial.
- The `MIN_TRIAL_FRAMES` check is performed twice (on raw frames and again on resampled bins).
- Running/pupil interpolation objects are rebuilt per trial from the full-session series rather than interpolating the session once onto all bin centres.
- All per-trial arrays are traversed a second time in the post-save assertion sweep.
- Historically, the pupil median was recomputed on every one of the 51 557 trials (fixed).

ii.
```python
def get_image_identity_at_times(stim_presentations, bin_centers, image_to_idx):
    stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
```
```python
        img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                                    {name: i for i, name in enumerate(image_names)})
        img_change = get_image_change_at_times(stim_cd, bin_centers)
        ...
        run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
        pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
```
(both called inside `for trial_idx, (_, trial) in enumerate(valid_trials.iterrows())`)

iii. Not discussed in the trajectory; the AI's only stated concern about repeated work was the `nanmedian` call, which it found and fixed. The remaining repetition appears to be an accepted consequence of organising the code as pure per-trial helper functions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Output arrays are stored as `int64`** although every value is in 0–15; `int8` would have cut that part of the file ~8×, and the pickle is 3.2 GB.
- The 90.9 ms rebinning **throws away ~2/3 of the temporal resolution** of the 168 single-plane experiments; that discarded information is not recoverable downstream (and was only needed because multiscope data were included).
- Human-readable bin-label strings (`speed_bin2_0.4_16.3`) are built from `min`/`max` scans over the pooled arrays purely for naming.
- Rich `session_info` metadata (`n_go_trials`, `n_catch_trials`, `cre_line`, `ophys_frame_rate`, …) is assembled for all 200 sessions but is not used by the decoder.
- The full five-way assertion sweep over all 51 557 trials runs *after* the file is written, so it costs time without being able to prevent a bad save.
- `trial_neural_raw` is materialised as a copy before resampling, doubling peak memory for each trial's neural slice.
- The raw (undiscretised) running and pupil traces are retained for every trial through the whole first pass; only the discretised versions are saved.
- `dff_df.iterrows()` builds a Python list of per-cell arrays before stacking.

ii.
```python
            output_data = np.zeros((5, n_bins), dtype=np.int64)
```
```python
    running_bin_labels = []
    for i in range(5):
        lo = running_edges[i] if i > 0 else valid_running.min()
        hi = running_edges[i + 1] if i < 4 else valid_running.max()
        running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")
```
```python
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    ...
    # Sanity checks
    for si, session in enumerate(output_all):
        for ti, out in enumerate(session):
            assert out.shape[0] == 5, ...
```

iii. The AI was memory-conscious in one place — it added `del all_running, all_pupil, all_running_flat, all_pupil_flat` after computing the bin edges (step 138) and observed 6.6 GB peak RSS — but it did not revisit dtype choice or the post-save verification cost. The rebinning trade-off is acknowledged explicitly in `CONVERSION_NOTES.md`: *"The common bin size of 90.9 ms means some temporal resolution is lost for the 31 Hz single-plane data."*
