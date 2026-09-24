# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the project metadata CSV `project_metadata/ophys_experiment_table.csv` directly (rather than the SDK's `get_ophys_experiment_table()`), intersects it with the experiment IDs of the NWB files actually present in `data/behavior_ophys_experiments/` (284 files), and then filters that set to `behavior_type == 'active_behavior'` and `experience_level == 'Familiar'`. This yields 110 experiments / 92 unique `ophys_session_id`s / 38 mice. Passive-viewing sessions and all Novel / Novel>1 sessions are dropped; no `project_code` filter is applied, so both `VisualBehavior` (88 experiments, Scientifica single-plane) and `VisualBehaviorMultiscope` (22 experiments, Mesoscope multi-plane) data are kept. Each experiment is then loaded individually through the SDK: `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='data')` → `cache.get_behavior_ophys_experiment(exp_id)`, inside a `try/except` so a failed load only skips that experiment.

ii.
```python
def get_experiment_table(data_dir='data'):
    exp_table = pd.read_csv(
        os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    )
    nwb_dir = os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/behavior_ophys_experiments/')
    nwb_files = os.listdir(nwb_dir)
    exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files]

    available = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
    filtered = available[
        (available['behavior_type'] == 'active_behavior') &
        (available['experience_level'] == 'Familiar')
    ].copy()
    filtered = filtered.sort_values(['mouse_id', 'ophys_session_id', 'ophys_experiment_id'])
    return filtered
```
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
exp_table = get_experiment_table(data_dir)
...
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(cache, exp_info, all_image_names, image_to_idx,
                                target_dt, all_running_speeds, all_pupil_diameters)
```
```python
def load_experiment(cache, ophys_experiment_id):
    return cache.get_behavior_ophys_experiment(ophys_experiment_id)
```

iii. From `CONVERSION_NOTES.md` and the trajectory: passive sessions are excluded because "mice are not performing the task"; Familiar-only is justified by an explicit quote from the paper methods — "For neural analysis we used neurons recorded during familiar image set presentations on the multi-plane imaging rig" and "we restricted our analysis to familiar stimuli". The AI initially reasoned (step 32) that restricting to the paper's exact subset — familiar *and* multi-plane — left only "22 such experiments from 1 mouse. That's too restrictive", so it kept both rigs "for a larger dataset, with appropriate resampling to a common time bin", and all three cre lines because "the paper analyzed all three". Intersecting with the on-disk NWB list avoids the cache trying to fetch experiments that are not present locally.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the experiments that survived processing, sorted, and converted to strings. `subject_idx` maps each output "session" (see 1-c) to its mouse. Result: 38 mice.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
...
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. Not discussed explicitly; `mouse_id` is the SDK/metadata-table identifier for the animal, and the AI reports mouse counts throughout (`Unique mice: 38`) as a sanity check against the metadata table.

## 1-c. How are the data split into sessions?

i. **Each ophys *experiment* (single imaging plane) is emitted as one "session" of the output format.** Experiments are never grouped by `ophys_session_id`. For the 88 single-plane (Scientifica) experiments this is equivalent to a session; for the 22 Multiscope experiments it is not — the 4–8 simultaneously recorded planes of one physical session each become a separate output "session" with the same trials repeated. This is visible in the output: mouse 457841 contributes 22 "sessions", in groups that share identical trial counts (209×7, 309×7, 265×3, 239×5) and have only 4–22 neurons each. Total: 110 output sessions from 92 physical sessions.

ii.
```python
    for i, (_, exp_info) in enumerate(exp_table.iterrows()):
        exp_id = int(exp_info['ophys_experiment_id'])
        ...
        result = process_experiment(cache, exp_info, ...)
        ...
        experiment_results.append({'exp_info': exp_info, 'neural_trials': neural_trials, ...})
...
    for result in experiment_results:
        ...
        neural_all.append(session_neural)   # one output "session" per experiment
        input_all.append(session_input)
        output_all.append(session_output)
```
```python
        # Brain region index for each neuron (all same region for one experiment)
        region_idx = region_to_idx[brain_region]
        brain_region_idx_all.append(np.full(n_neurons, region_idx, dtype=np.int64))
```

iii. From the trajectory (step 32): "Each imaging plane corresponds to one experiment, and since different planes have different neurons, each experiment should be treated as a separate 'session' in the output format." The AI had read the whitepaper passage defining a session as a single continuous recording containing up to 8 experiments, but chose the plane-level unit anyway so that each output session has a single homogeneous neuron population / brain region.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's built-in `dataset.trials` table. Trials are kept if `(go | catch) & ~aborted & ~auto_rewarded`. Each trial's window is the full `start_time` → `stop_time` interval (variable length, mean ≈ 8.2 s), taken as the set of ophys frames inside that closed interval. No fixed window around the change is used, so a trial contains the pre-change flashes and the post-change response period.

ii.
```python
    trials = ds.trials
    valid_trials = trials[
        ((trials['go'] == True) | (trials['catch'] == True)) &
        (trials['aborted'] == False) &
        (trials['auto_rewarded'] == False)
    ]
```
```python
    for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
        t_start = trial['start_time']
        t_stop = trial['stop_time']
        frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
        frame_indices = np.where(frame_mask)[0]
```

iii. `CONVERSION_NOTES.md`: "Go trials: Trials where the image changed... Catch trials: Trials where no change occurred (sham change)... Excluded: Aborted trials (premature licking before change), Auto-rewarded trials (free rewards)", matching the instruction to include Go and Catch and exclude Aborted and Auto-rewarded. The AI cross-checked the resulting trial length ("Mean trial length of ~8.2 seconds is consistent with the expected range from the truncated exponential change-time distribution (2.25–8.25 s, mean ~4.2 s, plus post-change time)") against the whitepaper.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (1) session/experiment level — only locally available NWBs, active behavior, familiar images; (2) trial level — aborted and auto-rewarded trials dropped, only go/catch kept; (3) trials whose ophys-frame window is shorter than the downsample factor (`len(frame_indices) < ds_factor`, i.e. fewer than 3 frames for single-plane data) are skipped so downsampling cannot produce an empty trial; (4) experiments with fewer than 2 valid trials before processing, or fewer than 2 usable trials after processing, are skipped entirely. No engagement/d-prime/reward-rate filtering and no per-trial neural QC is applied — trials whose event traces are entirely zero are kept (the verifier emits many "all neural data is zero" warnings for the low-neuron-count Multiscope planes).

ii.
```python
    if len(valid_trials) < 2:
        return None, None, None, None, None, f"Only {len(valid_trials)} valid trials"
```
```python
        if len(frame_indices) < ds_factor:
            continue
```
```python
    if len(neural_trials) < 2:
        return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. `CONVERSION_NOTES.md`: "Minimum trials: Sessions with fewer than 2 valid trials were excluded" (the format spec requires ≥2 trials per session to evaluate the decoder). The AI relies on the Allen pipeline's own session-level QC described in the whitepaper ("Assessment of the following quality metrics was performed after each imaging session…") rather than adding its own.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.events`, the SDK's detected-calcium-event table; the per-cell `events` column (unfiltered event magnitudes) is stacked into an `(n_neurons, T)` matrix. dF/F is not used. Neurons come only from the single experiment/plane being processed.

ii.
```python
    # Get neural data (calcium events as used in paper)
    try:
        events = ds.events
        neural_full = np.vstack(events['events'].values).astype(np.float32)
    except Exception as e:
        return None, None, None, None, None, f"Failed to get events: {e}"
```

iii. `CONVERSION_NOTES.md`: "Used **detected calcium events** (not raw dF/F), matching the paper: 'we used the detected calcium events as described in Garrett et al.'... Events are deconvolved from fluorescence traces... providing cleaner signals with ~200 ms resolution. Events exclude prolonged calcium transients that could contaminate responses to subsequent stimuli." This mirrors the methods text: "For all analysis of neural data we used the detected calcium events".

## 2-b. How is the `neural` data processed?

i. Minimal processing: the event traces are stacked, sliced to the trial's frame indices, and — for recordings faster than the target bin (single-plane) — averaged in non-overlapping blocks of `ds_factor` frames. No normalization, smoothing, z-scoring or neuron-level selection. No merging of neurons across imaging planes (see 1-c). Each neuron's brain region is the experiment's `targeted_structure` (`VISp` or `VISl`); imaging depth is not encoded in the region label.

ii.
```python
def downsample_by_factor(data, factor):
    ...
    elif data.ndim == 2:
        n = data.shape[1]
        n_new = n // factor
        if n_new == 0:
            return data[:, :1]
        return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)
```
```python
        neural_trial = neural_full[:, frame_indices]
        ...
        if ds_factor > 1:
            neural_trial = downsample_by_factor(neural_trial, ds_factor)
```

iii. `CONVERSION_NOTES.md`: "ROIs are pre-filtered by the AllenSDK processing pipeline (see whitepaper Section F: ROI Filtering). Only valid cell bodies are included (excludes dendrites, duplicates, edge artifacts, unions)"; "Neural data: averaged over 3 frames for downsampling" to reach a common bin size across rigs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural QC. Every ROI returned by the SDK for an experiment is kept; there is no minimum-event-rate, SNR, or activity filter, and no minimum neuron count (sessions with as few as 4 neurons are retained). The only implicit criterion is that the `events` table must load.

ii.
```python
    n_neurons = neural_full.shape[0]
    if n_neurons == 0:
        return None, None, None, None, None, "No neurons"
```

iii. `CONVERSION_NOTES.md` (ROI Filtering section): the AI relies entirely on the Allen pipeline's classifier-based ROI filtering and session-level QC described in the whitepaper, considering additional filtering unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All streams are placed on the ophys timebase (`ds.ophys_timestamps`) and each trial is the contiguous block of ophys frames with `t_start <= ophys_ts <= t_stop`, i.e. alignment is to **trial start**, with variable trial length. Metadata records `temporal_alignment_event = 'Trial start (aligned to ophys timestamps)'`, `off_start = 0.0`, `off_end = None`. The same `frame_indices` are used for neural, running, pupil, image identity and image change, so all streams are aligned by construction. Neural data are never interpolated — the imaging frames define the grid, and the behavioral streams are interpolated onto it.

ii.
```python
    ophys_ts = ds.ophys_timestamps
    ...
        frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
        frame_indices = np.where(frame_mask)[0]
        neural_trial = neural_full[:, frame_indices]
        trial_ophys_ts = ophys_ts[frame_indices]
        ...
        running_trial = running_at_ophys[frame_indices]
        pupil_trial = pupil_at_ophys[frame_indices]
```

iii. Instruction-driven ("Temporally align based on ophys timestamp"); `CONVERSION_NOTES.md`: "All signals aligned to ophys timestamps". The trajectory notes the ophys timestamps are the 2P frame times and that only the slower streams need interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is applied. A single target bin `target_dt = 0.0932 s` is chosen to match the Multiscope frame period. For each experiment the integer factor `ds_factor = round(target_dt / dt)` is computed from the median inter-frame interval and, when > 1, neural data / running / pupil are block-averaged, image identity is block-moded, and the change signal is block-averaged then re-binarized. In practice Multiscope data (dt = 93.21 ms) get `ds_factor = 1` and single-plane data (dt = 32.31 ms) get `ds_factor = 3`. Note the resulting bins are **not** identical across sessions: 93.21 ms for the 22 Multiscope sessions vs 3 × 32.31 = **96.93 ms** for the 88 single-plane sessions, while `metadata['time_bin_size']` is hard-coded to 93.2 ms for all of them. Mean trial length is 87.6 bins (~8.2–8.5 s). (The reference solution keeps the native single-plane resolution, 32.3 ms / ~264 bins per trial.)

ii.
```python
    # Target time bin (matching multiscope ~10.7 Hz)
    target_dt = 0.0932  # seconds
```
```python
    ophys_ts = ds.ophys_timestamps
    dt = np.median(np.diff(ophys_ts))
    ds_factor = max(1, int(round(target_dt / dt)))
    actual_dt = dt * ds_factor          # computed but never used
```
```python
        if ds_factor > 1:
            neural_trial = downsample_by_factor(neural_trial, ds_factor)
            image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
            change_signal = downsample_by_factor(change_signal, ds_factor)
            change_signal = (change_signal > 0.0).astype(np.float32)
            running_trial = downsample_by_factor(running_trial, ds_factor)
            pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```
```python
            'time_bin_size': target_dt * 1000,  # in ms
```

iii. From the trajectory (step 32): "The challenge is that different experiments have different frame rates, but the decoder format requires consistent time bins across all trials and sessions. Multiscope gives roughly 11 Hz (93 ms bins) while single-plane gives 31 Hz (32 ms bins). I could downsample everything to 11 Hz to work across both". `CONVERSION_NOTES.md`: "All data resampled to 93.2 ms bins (~10.7 Hz), matching the Multiscope frame rate. Single-plane Scientifica data (~31 Hz, ~32.3 ms bins) downsampled by factor of 3 via averaging." The rebinning is a direct consequence of the decision (1-a) to mix the two imaging rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `dataset.stimulus_presentations`, restricted to the change-detection block, using `image_name` and `start_time`; rows with `image_name == 'omitted'` are removed before computing identity. The trials table's `initial_image_name` / `change_image_name` are not used. The set of image categories is taken from the **first loaded experiment only** (8 images, image set A: im061…im085).

ii.
```python
    stim = ds.stimulus_presentations
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
    ...
        trial_stim = stim_cd[
            (stim_cd['start_time'] >= t_start - 0.5) &
            (stim_cd['start_time'] <= t_stop + 0.5)
        ]
        trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
        image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```
```python
    first_exp = cache.get_behavior_ophys_experiment(int(exp_table.iloc[0]['ophys_experiment_id']))
    stim = first_exp.stimulus_presentations
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
    all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                              if n != 'omitted' and isinstance(n, str)])
    image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
```

iii. `CONVERSION_NOTES.md`: "8 categories corresponding to the 8 natural scene images", cross-checked against the whitepaper ("Each session included 8 images... We confirmed 8 unique image names"). Using the stimulus-presentation table gives the image actually on screen at each flash rather than inferring it from the trial's initial/change image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, a per-ophys-frame integer code is produced by walking the trial's stimulus presentations in time and assigning, at each frame, the code of the most recently *started* image presentation. During the 500 ms grey inter-stimulus interval the previous image persists; omitted flashes also retain the previous image (they were removed from the table). Frames before the first presentation in the trial keep the initialization value 0 (i.e. they are labelled as `im061`). Any image name not in the mapping would be silently skipped (the previous identity carries forward) — harmless here only because every selected session uses image set A. When `ds_factor > 1` the per-frame codes are downsampled by taking the mode within each 3-frame block.

ii.
```python
def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    image_indices = np.zeros(len(ophys_timestamps), dtype=np.int64)
    stim_starts = stim_presentations['start_time'].values
    stim_images = stim_presentations['image_name'].values
    current_image_idx = -1
    stim_idx = 0
    for t_idx, t in enumerate(ophys_timestamps):
        while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
            stim_idx += 1
        if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
            img = stim_images[stim_idx]
            if img in image_to_idx and img != 'omitted':
                current_image_idx = image_to_idx[img]
        if current_image_idx >= 0:
            image_indices[t_idx] = current_image_idx
    return image_indices
```
```python
def downsample_categorical_by_factor(data, factor):
    ...
    for i in range(n_new):
        chunk = data[i*factor:(i+1)*factor]
        values, counts = np.unique(chunk, return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result
```

iii. `CONVERSION_NOTES.md`: "At each timepoint, assigned the most recently presented image. During gray screen intervals, the last shown image identity persists. Omitted stimuli retain the previous image identity"; "Categorical data (image identity): mode of 3 frames for downsampling". The AI validated the result against an expected uniform distribution ("All 8 images appear with approximately equal frequency (~12.3–12.8%)").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on `trial_ophys_ts` (the same frame times used to slice the neural matrix) and downsampled with the same factor, so it is per-bin aligned with the neural data by construction and is time-varying within the trial (it changes at the change flash).

ii.
```python
        trial_ophys_ts = ophys_ts[frame_indices]
        image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
        ...
        if ds_factor > 1:
            neural_trial = downsample_by_factor(neural_trial, ds_factor)
            image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. Follows from the decision to put every stream on the ophys frame grid (2-d): "All signals aligned to ophys timestamps".

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From `dataset.stimulus_presentations`: the `is_change` flag and the `start_time` of the change presentation (the omitted flashes are kept in this table, `trial_stim`). The trials table's `change_time`/`go` columns are not used.

ii.
```python
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
```
```python
        change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
```

iii. `CONVERSION_NOTES.md`: "Only actual changes are marked (not sham changes in catch trials)" — `is_change` is set only for real image transitions, so catch (sham-change) trials are automatically all-zero, matching the definition of the variable in the instructions ("value of 1 right after a change in image identity").

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary vector over the trial's ophys frames: 1 for frames in `[change_time, change_time + 0.75)`, i.e. the changed flash (250 ms) plus the following grey period (500 ms), 0 elsewhere. If the data are downsampled, the signal is block-averaged and then re-binarized with `> 0`, so any bin overlapping the 750 ms window is marked 1 (this slightly widens the event; the delivered change fraction is 8.2% vs 7.7% in the reference).

ii.
```python
def get_image_change_signal(stim_presentations, ophys_timestamps):
    change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
        mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
        change_signal[mask] = 1.0
    return change_signal
```
```python
            change_signal = downsample_by_factor(change_signal, ds_factor)
            change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md`: "Binary signal: 1 during the 750 ms following a change in image identity, 0 otherwise", with the 750 ms motivated by the flash cadence described in the whitepaper ("250 ms on, 500 ms gray = 750 ms per flash"). Validation: "~8.2% of timepoints are marked as image change, which is consistent with changes occurring once per trial".

## 4-c. How is `output` *Image change* thresholded into categories?

i. The variable is already binary — two categories, `output_values = ['no_change', 'change']`. The only threshold is the 750 ms post-change window and, after downsampling, the `> 0` rule that makes a bin a "change" bin if any of its constituent frames fell in that window.

ii.
```python
        'output_values': [
            all_image_names,                # image identity values
            ['no_change', 'change'],        # image change values
            ...
```
```python
            change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. See 4-b; the instruction specifies a binary variable that is 1 right after a change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Computed on `trial_ophys_ts` and downsampled with the same factor as the neural data, so it occupies the same bins. The stimulus table is pre-selected with a ±0.5 s pad around the trial so a change flash near a trial boundary is not missed; only frames inside the trial are ever marked.

ii.
```python
        trial_stim = stim_cd[
            (stim_cd['start_time'] >= t_start - 0.5) &
            (stim_cd['start_time'] <= t_stop + 0.5)
        ]
        change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
```

iii. Same rationale as 2-d/3-c — everything lives on the ophys frame grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, using its `timestamps` and `speed` columns (the SDK's 10 Hz-lowpass-filtered running speed in cm/s, not `running_speed_raw`).

ii.
```python
    rs = ds.running_speed
    running_at_ophys = interpolate_to_ophys(
        rs['timestamps'].values, rs['speed'].values, ophys_ts
    )
```

iii. `CONVERSION_NOTES.md`: "Uses the filtered running speed from the SDK (10 Hz lowpass Butterworth filtered)" — the whitepaper describes both the raw and filtered variants and the filtered one is the SDK default for `running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Linear interpolation from the encoder timebase onto every ophys timestamp (NaN samples dropped first, `bounds_error=False`, out-of-range → NaN); (2) remaining NaNs replaced by 0 cm/s; (3) the whole session's interpolated trace (all frames, including frames outside any kept trial) is appended to a global pool used later for the percentile edges; (4) per trial, the values at the trial's frames are taken and, for single-plane data, block-averaged over 3 frames; (5) discretized with the global edges.

ii.
```python
def interpolate_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    ...
    valid = ~np.isnan(signal_values)
    ...
    f = interpolate.interp1d(signal_timestamps[valid], signal_values[valid],
                             kind='linear', bounds_error=False, fill_value=np.nan)
    return f(ophys_timestamps)
```
```python
    running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
    all_running_speeds.extend(running_at_ophys.tolist())
    ...
        running_trial = running_at_ophys[frame_indices]
        ...
            running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. `CONVERSION_NOTES.md`: "Running speed: linearly interpolated from ~60 Hz analog input to ophys timestamps... Running speed NaN filled with 0"; "Continuous data (running speed, pupil): averaged over 3 frames".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins defined by the 20/40/60/80th percentiles of the **pooled full-session** interpolated speeds across all processed experiments; `np.digitize` then clips to 0–4. Because the edges are computed on all session frames at native resolution while the values actually stored are trial-restricted and 3-frame-averaged, the delivered bins are not exactly equal-sized: the verifier reports fractions 0.238 / 0.233 / 0.168 / 0.184 / 0.176 (reference: exactly 0.200 each). Edges: [0.013, 1.53, 17.5, 33.9] cm/s.

ii.
```python
    running_arr = np.array(all_running_speeds)
    running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
```
```python
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. `CONVERSION_NOTES.md`: "Discretized into 5 equal-percentile bins using global percentile boundaries computed across all sessions", per the instruction "discretized into five equal percentile bins". Global (rather than per-session) edges keep the category meaning consistent across sessions. The trajectory shows the AI debugging this binning (steps 55–63) after noticing only 4 distinct levels appeared, and fixing an off-by-one in `np.digitize`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation target is the full `ophys_timestamps` vector, so the speed trace shares indices with the neural matrix; the trial slice uses the same `frame_indices` and the same downsampling factor.

ii.
```python
    running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
    ...
        neural_trial = neural_full[:, frame_indices]
        running_trial = running_at_ophys[frame_indices]
```

iii. Same as 2-d: all data streams are hardware-synced to a common clock (whitepaper "DATA SYNCHRONIZATION" section, which the AI read), so interpolating onto the ophys timebase is valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`, column `pupil_width` with its `timestamps`. Blinks are not explicitly filtered by name, but the SDK already sets `pupil_width` to NaN on `likely_blink` frames, and the AI's interpolator drops NaN samples before fitting — so blink frames are effectively excluded and interpolated across, the same net effect as the reference's `~likely_blink` mask. The whole block is wrapped in `try/except`; if eye tracking is missing, the pupil trace becomes all-NaN.

ii.
```python
    try:
        et = ds.eye_tracking
        # Use pupil width as diameter (from tutorial: pupil_width)
        pupil_vals = et['pupil_width'].values
        pupil_ts = et['timestamps'].values
        pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
    except Exception:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```
```python
    valid = ~np.isnan(signal_values)
    f = interpolate.interp1d(signal_timestamps[valid], signal_values[valid], ...)
```

iii. `CONVERSION_NOTES.md`: "Uses pupil_width from eye tracking data as the diameter measure"; "NaN values (blink artifacts, tracking failures) filled via nearest valid value interpolation". The choice of `pupil_width` follows the SDK tutorial the AI read.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) NaN (blink/untracked) samples dropped, linear interpolation onto the ophys timestamps; (2) non-NaN interpolated values from the whole session are appended to a global pool for the percentile edges; (3) per trial, values at the trial's frames are taken and 3-frame-averaged for single-plane data; (4) residual NaNs (edge extrapolation) are filled with the nearest valid value in the trial; if a trial has no valid pupil value at all, the whole trial is assigned the middle bin (2); (5) discretized with the global edges.

ii.
```python
    valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
    all_pupil_diameters.extend(valid_pupil.tolist())
```
```python
            nan_mask = np.isnan(pupil_vals)
            if nan_mask.all():
                pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)  # median bin
            else:
                pupil_clean = pupil_vals.copy()
                if nan_mask.any():
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) > 0:
                        for j in range(len(pupil_clean)):
                            if nan_mask[j]:
                                dists = np.abs(valid_indices - j)
                                nearest = valid_indices[np.argmin(dists)]
                                pupil_clean[j] = pupil_clean[nearest]
                pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. `CONVERSION_NOTES.md`: "Pupil diameter (width): linearly interpolated from ~30 Hz eye tracking to ophys timestamps"; "NaN values (blink artifacts, tracking failures) filled via nearest valid value interpolation" — chosen so that no trial has to be dropped for missing eye data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five bins from the 20/40/60/80th percentiles of the pooled non-NaN full-session pupil widths across all experiments (edges [36.8, 41.2, 45.8, 53.0] px), applied with the same `np.digitize` + clip helper. Delivered fractions are 0.202 / 0.189 / 0.216 / 0.201 / 0.191 — close to equal despite the same pool/downsample mismatch noted in 5-c.

ii.
```python
    if len(pupil_arr) > 0:
        pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
    else:
        pupil_percentiles = np.array([0, 1, 2, 3])
        print("WARNING: No valid pupil data!")
```

iii. `CONVERSION_NOTES.md`: "Discretized into 5 equal-percentile bins using global percentile boundaries" — same rationale as running speed, per the instruction.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the full `ophys_timestamps` vector, then sliced with the same `frame_indices` and downsampled with the same factor as the neural data.

ii.
```python
    pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
    ...
        pupil_trial = pupil_at_ophys[frame_indices]
        ...
            pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. Same synchronization rationale as running speed (2-d, 5-d).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The mutually exclusive boolean columns `hit`, `miss`, `false_alarm`, `correct_reject` of the trials table, checked in that order; anything else becomes the string `'unknown'`.

ii.
```python
def get_trial_outcome(trial):
    if trial['hit']:
        return 'hit'
    elif trial['miss']:
        return 'miss'
    elif trial['false_alarm']:
        return 'false_alarm'
    elif trial['correct_reject']:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. `CONVERSION_NOTES.md`: "4 categories: hit (0), miss (1), false_alarm (2), correct_reject (3)", matching the whitepaper's description of the four trial types produced by go/catch trials crossed with the animal's response. Validated against expected behaviour: "~28.4% hits, ~59.0% misses, ~1.9% false alarms, ~10.7% correct rejections. The high miss rate is consistent with including all sessions regardless of engagement level."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to an integer 0–3 by a fixed dictionary and broadcast to a constant row over the trial's bins, so the static per-trial variable is stored as a time-varying output row (row 4 of the `(5, T)` output matrix). An unrecognized outcome falls back to index 0 (`hit`) rather than a sentinel — no such trial can occur given the go/catch filter, but the fallback is silent.

ii.
```python
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
    outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
...
            outcome_idx = outcome_to_idx.get(outcome, 0)
            output_trial = np.stack([
                image_idx.astype(np.int64),
                change_signal.astype(np.int64),
                running_binned.astype(np.int64),
                pupil_binned.astype(np.int64),
                np.full(n_timepoints, outcome_idx, dtype=np.int64),
            ], axis=0)
```

iii. `CONVERSION_NOTES.md`: "Static per trial (same value across all timepoints within a trial). Represented as time-varying for format consistency" — the format spec asks for time-varying outputs where possible and requires a uniform output shape across the five variables.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling, by failure mode:
- **Experiment fails to load / no `events` table / zero neurons** → the experiment is skipped with a printed reason, the run continues.
- **Missing eye tracking** → `try/except` produces an all-NaN pupil trace; per trial, all-NaN pupil is assigned the middle bin (2).
- **Behavioral samples outside the ophys range** → interpolation returns NaN; running NaN → 0 cm/s, pupil NaN → nearest valid value in the trial.
- **NaN samples inside a behavioral stream** (including SDK blink-blanked pupil values) → dropped before interpolation, i.e. interpolated across.
- **Degenerate trials** → trials with fewer frames than the downsample factor are skipped; experiments with < 2 valid/usable trials are skipped.
- **Insufficient pupil data overall** → a warning and placeholder edges `[0,1,2,3]`.
- Trials with all-zero event traces are kept (the verifier flags hundreds of them, mostly in low-neuron-count Multiscope planes).

ii.
```python
    try:
        ds = load_experiment(cache, exp_id)
    except Exception as e:
        return None, None, None, None, None, f"Failed to load: {e}"
```
```python
    running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
```
```python
            if nan_mask.all():
                pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)  # median bin
```
```python
        if len(frame_indices) < ds_factor:
            continue
    ...
    if len(neural_trials) < 2:
        return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. `CONVERSION_NOTES.md`: "NaN handling: Running speed NaN filled with 0; Pupil NaN filled via nearest-neighbor interpolation." The intent is that no otherwise-valid trial is lost because a behavioral sample is missing; zero speed and the nearest/median pupil value are treated as conservative defaults.

## 9-a. What are the most time-consuming steps of the code?

i. Dominated by I/O: 111 calls to `cache.get_behavior_ophys_experiment()` (the first experiment is loaded twice), each of which parses a full-session NWB (dF/F, events, stimulus, eye tracking, running). Secondary costs: building two Python lists holding every interpolated running / pupil sample of every session (~12 M floats each, several hundred MB), the per-timepoint Python loops (`get_image_at_timepoints`, `downsample_categorical_by_factor`, the pupil nearest-fill), re-filtering the stimulus table once per trial (~27.6 k times), and pickling the resulting 1.4 GB file.

ii.
```python
    all_running_speeds.extend(running_at_ophys.tolist())
    ...
    valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
    all_pupil_diameters.extend(valid_pupil.tolist())
```
```python
        trial_stim = stim_cd[
            (stim_cd['start_time'] >= t_start - 0.5) &
            (stim_cd['start_time'] <= t_stop + 0.5)
        ]
```

iii. Not discussed by the AI; the design (one experiment loaded, processed, and released per iteration, everything kept in memory until the end) implies it treated NWB loading as the unavoidable bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Four avoidable Python loops: (1) `get_image_at_timepoints` walks every ophys timestamp of every trial — a single `np.searchsorted(stim_starts, t)` would give the same result vectorized; (2) `downsample_categorical_by_factor` calls `np.unique` per output bin — a reshape plus `bincount`/`mode` over the last axis would do; (3) the pupil nearest-valid fill computes `np.abs(valid_indices - j)` for every NaN index — O(n_nan × n_valid), replaceable by a single `np.interp`/forward-fill; (4) `get_image_change_signal` and the trial loop use `iterrows()` over pandas rows. None of them dominate runtime relative to NWB loading, which is presumably why they were left alone.

ii.
```python
    for t_idx, t in enumerate(ophys_timestamps):
        while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
            stim_idx += 1
```
```python
                        for j in range(len(pupil_clean)):
                            if nan_mask[j]:
                                dists = np.abs(valid_indices - j)
                                nearest = valid_indices[np.argmin(dists)]
                                pupil_clean[j] = pupil_clean[nearest]
```

iii. No justification given; the AI never profiled or commented on runtime.

## 9-c. What processing does the code repeat multiple times?

i. (1) The first experiment is loaded twice — once up front just to discover the image names, then again inside the main loop. (2) The change-detection stimulus table is re-filtered for every trial (two boolean masks over the whole table, ~27.6 k times) instead of being indexed once per experiment. (3) Every interpolated running/pupil sample is stored twice: once in the global percentile pool and again in the per-trial slices. (4) `stim_cd`-derived `stim_images` is computed per experiment and never used (see 9-d).

ii.
```python
    first_exp = cache.get_behavior_ophys_experiment(int(exp_table.iloc[0]['ophys_experiment_id']))
    stim = first_exp.stimulus_presentations
    ...
    for i, (_, exp_info) in enumerate(exp_table.iterrows()):   # loads the same experiment again
```
```python
        trial_stim = stim_cd[(stim_cd['start_time'] >= t_start - 0.5) &
                             (stim_cd['start_time'] <= t_stop + 0.5)]
        trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
```

iii. Not discussed. The two-pass structure (process everything, then compute global bin edges) is deliberate and documented ("global percentile boundaries computed across all sessions"); the duplicated first load appears to be incidental.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Dead or discarded work: `stim_images` (line 245), `actual_dt` (line 228), `stim_ends` (line 129) and the `image_names` parameter of `process_experiment` are computed/passed and never used; `trial_outcomes` is built, returned and stored in `experiment_results` but the outcome actually written comes from `out_raw['outcome']`. The running/pupil percentile pools contain every frame of every session, including the ~half of the recording that lies outside the kept trials — that work is not merely wasted, it shifts the bin edges away from the distribution of the data actually stored (see 5-c). The change signal is block-*averaged* and then immediately re-binarized (a max/any reduction would be equivalent and cheaper). Empty `(0, n_timepoints)` input arrays are allocated per trial even though `input_names` is empty.

ii.
```python
    stim_images = stim_cd[stim_cd['image_name'] != 'omitted']   # never used
    ...
    actual_dt = dt * ds_factor                                   # never used
```
```python
            session_input.append(np.zeros((0, n_timepoints), dtype=np.float32))
```
```python
            change_signal = downsample_by_factor(change_signal, ds_factor)
            change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. Not discussed by the AI. The full-session pooling for percentiles is implicitly justified in `CONVERSION_NOTES.md` as computing "global percentile boundaries... across all sessions", but the notes do not acknowledge that the pool differs from the data being binned.
