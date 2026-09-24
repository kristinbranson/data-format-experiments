# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data`, treating every sub-directory matching `sub-*` as a subject and every `.nwb` file inside it as one session, producing a flat list of 152 (subject, filepath) records over 11 subjects. Files are opened directly with `h5py` (raw HDF5) rather than with `pynwb`; the NWB group paths (`processing/behavior/BehavioralTimeSeries/...`, `processing/ophys/...`, `general/...`) are hard-coded. For each session the AI reads the complete behavioral time series (position, speed, lick, reward_zone, trial_start, teleport, trial number, environment, scanning, Reward data+timestamps, position timestamps) and the complete ophys arrays (`Deconvolved`, `Fluorescence`, `Neuropil` for every plane, plus `iscell` and `planeIdx`). Multi-plane sessions (m17, m18) are handled by concatenating the ROIs of `plane0` and `plane1` along the neuron axis. Trials are then cut out of these whole-session arrays. A `--sample` mode hard-codes two sessions (m11 ses-03, m3 ses-03).

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    """Get all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })
```
```python
    with h5py.File(filepath, 'r') as f:
        behav = f['processing']['behavior']['BehavioralTimeSeries']
        position = behav['position']['data'][:]
        ...
        ophys = f['processing']['ophys']
        planes = sorted(ophys['Deconvolved'].keys())
        if len(planes) == 1:
            deconv_data = ophys['Deconvolved']['plane0']['data'][:]
            ...
        else:
            for plane in planes:
                deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
                ...
            deconv_data = np.concatenate(deconv_parts, axis=1)
```

iii. From CONVERSION_NOTES Step 2: "NWB files: `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`; each file contains one session for one mouse". The AI cross-checked the directory listing against the paper: "11 subjects (all switch-condition mice), 152 total sessions. sub-m11 has 12 sessions (imaging started day 3); all others have 14", and separately reasoned from `sessions_dict.py` that the 3 fixed-condition mice (GCAMP2/6/10) are not present in the NWB release. Multi-plane pooling is justified as "consistent with paper" (Step 5 decision 8). `h5py` was chosen without explicit justification (the NWB layout was mapped out by direct inspection in Step 2).

## 1-b. How are the data split into subjects?

i. Subjects are the `sub-*` directories, but the subject label that actually ends up in the output is read from the NWB file itself (`general/subject/subject_id`, e.g. `b'm11'`), which agrees with the directory name. `subjects` is built lazily in first-encounter order while iterating sessions, and `subject_idx` is the index of each session's subject in that list. The resulting list is the same 11 mice as the reference (m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7) with the same sessions-per-subject counts.

ii.
```python
        subject_id = f['general']['subject']['subject_id'][()].decode() if isinstance(
            f['general']['subject']['subject_id'][()], bytes) else str(f['general']['subject']['subject_id'][()])
```
```python
        subj = session_info['subject']
        if subj not in subjects_list:
            subjects_list.append(subj)
        subject_idx_list.append(subjects_list.index(subj))
...
        'subjects': subjects_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2: subjects are "m3, m4, m7, m11-m15, m17-m19"; Step 3 records the paper's "n = 11 mice" for the counterbalanced switch task and notes the 3 fixed-condition mice are "NOT in NWB data". The AI verified the GCAMP# ↔ m# mapping against `sessions_dict.py` in the reference repo (trajectory steps 47–50), using the fact that m11 has only 12 sessions because imaging started on day 3 for that mouse.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. No session-level curation is applied other than a guard that drops any session yielding fewer than 2 valid trials (never triggered: all 152 sessions are kept). Sessions are ordered by subject directory then filename, so `session_id` is implicitly `ses-01 … ses-14`. No attempt is made to align cells across days.

ii.
```python
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```
```python
        neural_trials, input_trials, output_trials, session_info = process_session(
            file_info['filepath'], show_processing=show_processing, session_idx=i)

        if len(neural_trials) < 2:
            print(f"  WARNING: Skipping session with <2 valid trials")
            continue
```

iii. Step 5 decision 9: "Each NWB file = 1 session in the output data structure." The <2-trial guard implements the instruction's requirement that "there needs to be at least two trials within each session in order to evaluate the decoder performance". The session count (152) was checked against the paper's 11 mice × 14 days (12 for m11).

## 1-d. How are the data split into trials?

i. A trial runs from the sample where the `trial_start` binary channel is 1 up to (but excluding) the sample where the `teleport` binary channel is 1. Both channels are single-sample pulses, so `np.where(signal > 0)` recovers the event indices directly. The AI then defensively truncates both index vectors to the shorter one and keeps only pairs where the teleport index is after the trial-start index. The NWB `trial number` channel is loaded but is *not* used to define trials. This reproduces the reference segmentation exactly: 12,216 trials, T mean 216.78, median 197.49, min 96, max 3359 — identical to the reference statistics.

ii.
```python
    # ---- Find trial boundaries ----
    trial_starts = np.where(trial_start_signal > 0)[0]
    teleports = np.where(teleport_signal > 0)[0]

    # Ensure matching number of starts and teleports
    n_trials = min(len(trial_starts), len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]

    # Ensure each teleport comes after its corresponding trial start
    valid = teleports > trial_starts
    trial_starts = trial_starts[valid]
    teleports = teleports[valid]
    n_trials = len(trial_starts)
```
```python
    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        n_timepoints = end - start
```

iii. CONVERSION_NOTES Step 1: "Trial boundaries: `trial_start_inds` to `teleport_inds`" taken from the reference `sess` object; Step 5 decision 10: "Trial definition: trial_start to teleport indices. Include all valid frames between these." The teleport period (inter-trial interval) is deliberately excluded ("Teleport periods excluded from analysis", Step 3). The resulting trial count per session (~80, with 41–100 in some sessions) was checked against the paper's "80.5 ± 7.4 trials".

## 1-e. How are trials filtered based on quality controls?

i. Only two guards: (a) a trial is skipped if it is shorter than 5 samples; (b) a session is skipped if fewer than 2 trials survive. Neither ever fires on this dataset (shortest trial is 96 frames), so all 12,216 detected trials are kept. Trials with a suspected lick-sensor fault are *not* removed — instead only their lick channel is neutralised (see 9-b). No speed-based or behaviour-based trial exclusion is applied; the paper's 2 cm/s criterion is deliberately not used to drop samples because speed is itself a decoder output.

ii.
```python
        if n_timepoints < 5:
            continue  # Skip very short trials
```
```python
        if len(neural_trials) < 2:
            print(f"  WARNING: Skipping session with <2 valid trials")
            continue
```
```python
    # Lick error correction per trial
    for t in range(n_trials):
        trial_lick = lick[start:end]
        if len(trial_lick) > 0:
            frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
            if frac_bad > LICK_ERROR_FRACTION:
                lick_binary[start:end] = np.nan
                lick_error_trials.append(t)
```

iii. Step 5 decision 2: "Speed threshold applied as masking. Following reference code, samples with speed<2 are kept in time but masked. For the decoder format, we include all timepoints but the speed output will reflect this" — i.e. the AI consciously declined to drop low-speed samples because the decoder must predict the speed bin. Step 10 edge-case check records "no very short trials", so the 5-sample rule is documented as a no-op safety net. The lick-artifact rule (>35% of samples with cumulative lick > 2) is copied from the reference `get_timeseries_data()`; the AI reported 69 such trials against the paper's 81 (the paper's count includes the 3 fixed-condition mice absent from this data).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are slices of the NWB `processing/ophys/Deconvolved/plane*/data` array — suite2p's stored deconvolved trace — restricted to ROIs that pass `iscell` and are not putative interneurons. `Fluorescence` and `Neuropil` *are* loaded and a dF/F is computed from them, but that dF/F is used **only** to detect interneurons; it never enters the saved `neural` field. The paper's own pipeline (dF/F from F and Fneu with a per-trial maximin baseline, then OASIS deconvolution → `sess.timeseries['events']`) was read and documented by the AI but not used for the output.

ii.
```python
        ophys = f['processing']['ophys']
        seg = ophys['ImageSegmentation']['PlaneSegmentation']
        iscell = seg['iscell'][:, 0].astype(bool)
        ...
            deconv_data = ophys['Deconvolved']['plane0']['data'][:]  # (n_samples, n_rois)
            fluor_data = ophys['Fluorescence']['plane0']['data'][:]
            neuropil_data = ophys['Neuropil']['plane0']['data'][:]
```
```python
    # ---- Get deconvolved data for selected neurons ----
    neural_all = deconv_data[:, final_neuron_mask]  # (n_samples, n_neurons)
...
        trial_neural = neural_all[start:end, :].T.copy()  # (n_neurons, n_timepoints)
```

iii. CONVERSION_NOTES Step 1, "Critical Discovery: NWB files already contain deconvolved activity": "The NWB `processing/ophys/Deconvolved/plane0/data` contains pre-computed deconvolved events. This means we do NOT need to compute dF/F from scratch - the NWB data is at the `sess` level with pre-processed neural data." Step 4 consistency table: "Deconvolved data | Computed from dF/F via OASIS | Pre-computed in NWB | 'OASIS algorithm' | Use pre-computed deconvolved directly". Step 5 decision 1: "Use Deconvolved events directly: NWB has pre-computed deconvolved data matching the reference pipeline." Step 10 sanity check 1 verifies the saved neural values are an exact `np.allclose` match to the NWB `Deconvolved` array.

## 2-b. How is the `neural` data processed?

i. Essentially none. The stored deconvolved trace is sliced per trial, transposed to (n_neurons, n_timepoints), cast to `float32`, and any NaN replaced by 0. There is no dF/F normalisation, no baseline correction, no per-cell scaling, no smoothing and no re-deconvolution, so the saved values are in raw fluorescence units (range 0 to ~1.5e4, session-mean ≈ 32).

A separate dF/F *is* computed (for interneuron detection only) and it does follow the paper's recipe fairly closely: subtract `0.7 × Fneu`, then per trial smooth with a Gaussian of σ=15 frames, take a 300-frame (≈20 s) running minimum followed by a 300-frame running maximum as the baseline, form `(F − baseline)/|baseline|`, and smooth with a σ=2-frame Gaussian. Two things differ from the paper's `preprocessing.dff`: the per-trial mean neuropil is not added back before forming the ratio, and no OASIS deconvolution is run on it.

ii.
```python
def compute_dff(F, Fneu, trial_starts, teleports):
    F_corr = F - NEUROPIL_COEF * Fneu
    dff = np.full_like(F_corr, np.nan)
    for i, (start, stop) in enumerate(zip(trial_starts, teleports)):
        ...
        smoothed = gaussian_filter1d(trial_F, sigma=15, axis=0)
        min_filtered = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
        baseline = maximum_filter1d(min_filtered, size=BASELINE_WINDOW, axis=0)
        abs_baseline = np.abs(baseline)
        abs_baseline[abs_baseline < 1e-10] = 1e-10
        trial_dff = (trial_F - baseline) / abs_baseline
        dff[start:stop, :] = gaussian_filter1d(trial_dff, sigma=DFF_SMOOTH_SIGMA, axis=0)
    return dff
```
```python
        trial_neural = neural_all[start:end, :].T.copy()  # (n_neurons, n_timepoints)
        # Replace NaN with 0 in neural data (following reference: X[np.isnan(X)] = 0)
        trial_neural[np.isnan(trial_neural)] = 0
        ...
        neural_trials.append(trial_neural.astype(np.float32))
```

iii. Step 6: "dF/F computation (for interneuron check only): Neuropil subtraction (0.7), maximin baseline (300 frames), Gaussian smooth (sigma=2)". The AI's stated reason for not processing the output neural stream is the "critical discovery" quoted in 2-a — it believed the stored `Deconvolved` array already is the paper's deconvolved event trace. In Step 7 it noted "The neural data values are very large (max ~10000) which is normal for deconvolved calcium events" and accepted them. `metadata['neural_data_type']` is set to `'deconvolved calcium events (OASIS)'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper. First, ROIs are restricted to `iscell` (suite2p manual curation), applied per session across all planes. Second, putative interneurons are removed: the per-cell Pearson correlation between the computed dF/F and the running speed is evaluated on all in-trial samples with valid speed, and cells with r > 0.5 are dropped. The correlation is computed in one vectorised matrix operation rather than per neuron. Across the dataset this removed 385 of 138,678 `iscell` ROIs (0.28%), leaving 138,293 neurons (mean 909.8/session, range 155–2322).

ii.
```python
    n_neurons = iscell_mask.sum()
    dff_cells = dff[:, iscell_mask]  # (n_samples, n_neurons)
    speed_valid = ~np.isnan(speed)
    neuron_valid = ~np.isnan(dff_cells[:, 0]) if n_neurons > 0 else np.zeros(len(speed), dtype=bool)
    valid = speed_valid & neuron_valid
    ...
    cov_XY = (X_centered * Y_centered[:, np.newaxis]).sum(axis=0) / (n - 1)
    std_X = np.sqrt((X_centered ** 2).sum(axis=0) / (n - 1))
    std_Y = np.sqrt((Y_centered ** 2).sum() / (n - 1))
    r = cov_XY / (std_X * std_Y)
    is_interneuron = r > INTERNEURON_CORR_THRESHOLD
```
```python
    iscell_indices = np.where(iscell)[0]
    non_interneuron = ~is_interneuron
    final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
    final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. Step 3 curation rules: "1. `iscell` from Suite2p manual curation (already in NWB); 2. Interneuron exclusion: Pearson correlation of dF/F with speed > 0.5", with the paper quotes "correlation of >0.5" and "excluding 0.42 ± 0.85% of cells". Step 4 notes that dF/F had to be computed specifically to make this check possible. Step 10 compares the achieved 0.28% exclusion rate to the paper's 0.42 ± 0.85% and the neuron range 155–2322 to the paper's 155–2172, calling both consistent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial cut itself: the first column of every neural matrix is the frame at which `trial_start` fires, and the trial runs to the frame before teleport with no pre-event window. Behaviour and imaging are stored on a common frame index in the NWB file, so the same `start:end` slice is applied to neural and behavioural arrays. Where the imaging array is one frame longer than the behaviour arrays (10 multi-plane sessions in m17/m18), everything is truncated to the common length before trials are cut. Metadata records `temporal_alignment_event = 'start of each trial (first imaging frame on track after teleport)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
    n_samples = min(n_behav_samples, n_neural_samples)
    if n_behav_samples != n_neural_samples:
        position = position[:n_samples]
        ...
        deconv_data = deconv_data[:n_samples, :]
```
```python
        start = trial_starts[t]
        end = teleports[t]
        trial_neural = neural_all[start:end, :].T.copy()
        ...
        trial_pos = position[start:end]
        trial_speed = speed[start:end]
        trial_lick = lick_binary[start:end]
```

iii. Step 5 decision 4: "Temporal alignment: Align to trial start (first frame of each trial)". Step 4 conclusion 1: "NWB behavioral data matches reference code's `sess.vr_data` format (same variables at same frame rate)", i.e. the streams are already co-registered frame-by-frame. Step 6: "Shape mismatch fix: For multi-plane animals, behavioral and neural data can differ by 1 frame; truncated to common length."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of any kind. One time bin = one imaging frame at 15.5078125 Hz = 64.4836 ms, identical for every trial and session. For the two-plane animals (m17, m18) the NWB `imaging_rate` attribute is 31.015625 Hz, but each plane's series is already stored at the per-plane rate, so the same 15.5078125 Hz applies; the AI hard-codes this as a module constant and uses it both for `metadata['time_bin_size']` and for the time-from-trial-start input. The per-file `imaging_rate` is read into `session_info` but is not used or asserted against the constant.

ii.
```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
```
```python
            'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
            'temporal_alignment_event': 'start of each trial (first imaging frame on track after teleport)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'imaging_rate_hz': IMAGING_RATE,
```

iii. Step 5 decision 3: "Time bin = 1 imaging frame: ~64.5 ms (1/15.5078125 Hz). This matches the native temporal resolution." Step 2 records "Imaging rate 15.5078125 Hz (31.015625 Hz for 2-plane)" and Step 3 quotes the paper's "~15.5 Hz", with the note "(31 Hz for 2-plane, 15.5 per plane)". The behavioural channels are stored at the imaging frame rate already (Step 2), so a single common bin size needs no interpolation.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not read from any stored timestamp array. It is synthesised from the sample index within the trial multiplied by the fixed frame period (1/15.5078125 s). The NWB `position` timestamps *are* loaded, but only to map reward event times onto frame indices. The resulting range (0 to 216.5 s) is identical to the reference's timestamp-derived values (0 to 216.536 s), confirming the frame clock is uniform.

ii.
```python
        behav_timestamps = behav['position']['timestamps'][:]
```
```python
        # [0] Time from trial start in seconds (time-varying)
        time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
        ...
        trial_input[0, :] = time_from_start
```

iii. Step 5 mapping table: "Time from trial start | input[0] | seconds, time-varying | Custom | Computed from frame timestamps relative to trial start". The implicit justification is Step 5 decision 3 — the frame rate is constant across the whole dataset, so index × frame period is the sampling time. Step 10 input sanity check states "Time from trial start ... match NWB source".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond the multiplication: because the counter restarts at `np.arange(n_timepoints)` for every trial, the first sample of each trial is exactly 0 s and the value increases monotonically in 64.4836 ms steps. It is stored as a time-varying row of the (4, n_timepoints) float32 input array.

ii.
```python
        trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
        trial_input[0, :] = time_from_start
```

iii. Not separately justified beyond the mapping-table entry above; the decoder spec lists this input as "continuous, time-varying" and Step 5 decision 4 fixes the zero point at the trial-start frame.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the time vector has exactly `n_timepoints = end - start` entries, the same length as the neural slice for that trial, and index 0 of both corresponds to the trial-start frame. The global truncation to `min(n_behav_samples, n_neural_samples)` before trial cutting guarantees the neural and behavioural arrays share an index space. No timestamp equality assertion is made.

ii.
```python
        n_timepoints = end - start
        trial_neural = neural_all[start:end, :].T.copy()
        time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. Step 4 conclusion 1 ("NWB behavioral data matches reference code's `sess.vr_data` format (same variables at same frame rate)") is the stated basis for treating the frame index as a shared clock; Step 6 documents the truncation as the fix for the 1-frame multi-plane discrepancy.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioural time series. Its values are −1 before the imaging TTL and 0 (ENV1) or 1 (ENV2) during the session; it is constant within a session in this dataset.

ii.
```python
        environment = behav['environment']['data'][:]
```
```python
        env_vals = environment[start:end]
        valid_env = env_vals[env_vals >= 0]
```

iii. Step 2 variable glossary: "`environment`: -1 pre-TTL, 0=ENV1, 1=ENV2". Step 5 mapping table: "Environment (0/1) | input[1] | binary per trial | `environment` in NWB | ENV1=0, ENV2=1", matching the decoder spec's "binary, ENV1 vs ENV2, per trial".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial the AI takes the median of the non-negative environment samples within the trial window, casts to int, and broadcasts that single value across all timepoints of the trial (row 1 of the input array). Negative (pre-TTL) samples are excluded from the median; if a trial had no valid sample the value defaults to 0. The observed range in the converted data is [0, 1].

ii.
```python
    trial_env = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        env_vals = environment[start:end]
        valid_env = env_vals[env_vals >= 0]
        if len(valid_env) > 0:
            trial_env[t] = int(np.median(valid_env))
        else:
            trial_env[t] = 0
```
```python
        env_val = float(trial_env[t])
        trial_input[1, :] = env_val
```

iii. The decoder spec calls for a per-trial binary value; the median over valid samples is the AI's way of collapsing the time series to one robust per-trial label while discarding the −1 pre-TTL sentinel documented in Step 2. Step 10's input sanity check reports environment values verified against the NWB source.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. In the code, trial number is the loop index over the detected trial-start/teleport pairs within the session (0, 1, 2, …), i.e. it is derived from the `trial_start`/`teleport` channels, not from any stored counter. The NWB `trial number` channel is loaded (and truncated) but never used. Note that the documentation disagrees with the code here: the Step 5 mapping table states the input is derived from "`trial number` in NWB". The two are not interchangeable — e.g. in m11 ses-03 the stored channel runs 0…80 (81 distinct values) against 80 `trial_start` pulses. The saved range is [0, 99], matching the reference.

ii.
```python
        trial_number = behav['trial number']['data'][:]   # loaded, never used afterwards
```
```python
    for t in range(n_trials):
        ...
        # [2] Trial number (per trial)
        trial_num = float(t)
        ...
        trial_input[2, :] = trial_num
```

iii. Step 5 mapping table: "Trial number | input[2] | continuous per trial | `trial number` in NWB | 0-indexed trial within session" — the AI's stated intent was a 0-indexed within-session trial index, which is what the loop counter delivers. Step 10's input sanity check claims "trial number ... match NWB source data".

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None: the integer loop index is cast to float and broadcast across every timepoint of the trial as row 2 of the input array. The counter indexes *detected* trials, so it would leave a gap if a trial were ever skipped by the <5-sample rule (this never happens here). It is not normalised or rescaled.

ii.
```python
        trial_num = float(t)
        trial_input[2, :] = trial_num
```

iii. The decoder spec asks for "Trial number (continuous, per trial)"; the AI treats the within-session sequential index as that quantity (Step 5 mapping table, "0-indexed trial within session"). Keeping it unnormalised preserves the raw trial ordinal, which is what the reference also stores.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behavioural series, which is event-based: it has its own short `timestamps` vector (74 events in m11 ses-03) and a `data` vector of delivered volumes (all 0.004). The AI uses only the timestamps, mapping them onto behavioural frame indices with `np.searchsorted` against the `position` timestamps and clipping into range; `reward_data` (the amounts) is loaded but unused. A trial counts as rewarded if any reward frame index falls in `[trial_start, teleport)`; the previous-trial outcome is that flag shifted by one trial.

ii.
```python
        reward_data = behav['Reward']['data'][:]
        reward_timestamps = behav['Reward']['timestamps'][:]
        behav_timestamps = behav['position']['timestamps'][:]
```
```python
    reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
    reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)

    trial_rewarded = np.zeros(n_trials, dtype=bool)
    for t in range(n_trials):
        trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
        trial_rewarded[t] = trial_rewards
```

iii. Step 2: "`Reward`: event-based (timestamps + data), data values are 0.004 (reward amount?)"; Step 5 mapping table: "Previous trial outcome | input[3] | binary per trial (0=omitted, 1=rewarded) | Custom from Reward events | Check if previous trial had reward". Step 6: "Reward detection: Reward event timestamps mapped to frame indices using NWB timestamps."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_trial_outcome[t] = int(trial_rewarded[t-1])` for t ≥ 1; the first trial of every session is set to 0 because there is no preceding trial. The value is broadcast across all timepoints of the trial as row 3 of the input array. Range in the converted data is [0, 1].

ii.
```python
    # ---- Previous trial outcome ----
    prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
    for t in range(1, n_trials):
        prev_trial_outcome[t] = int(trial_rewarded[t - 1])
    # First trial: no previous, default to 0 (unknown/omitted)
```
```python
        prev_outcome = float(prev_trial_outcome[t])
        trial_input[3, :] = prev_outcome
```

iii. The decoder spec defines this input as "binary, omitted = 0, rewarded = 1, per trial". The code comment gives the rationale for the first trial ("no previous, default to 0 (unknown/omitted)"). The overall omission rate produced by this reward detection (15.7% not rewarded) was checked in Step 10 against the paper's "~15%".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` time series together with a per-trial reward-zone assignment. The zone is inferred from the `reward_zone` channel (0–6, non-zero while the animal is in the zone): the mean `position` over samples with `reward_zone > 0` is computed and matched to whichever of the three canonical zones A (80–130 cm), B (200–250 cm), C (320–370 cm) has the nearest centre. Trials in which the animal never triggers the zone signal (typically 6–17 per session) inherit the last known zone; if the leading trials have no zone signal they are back-filled from the first trial that does. Distance is then computed from the trial's `position` trace to that zone's edges.

ii.
```python
def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]
    in_rz = rz_trial > 0
    if not np.any(in_rz):
        return None  # No reward zone entry (omission or no entry)
    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)
    best_zone = None
    best_dist = np.inf
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone_name
    return best_zone
```
```python
    last_known_zone = None
    for t in range(n_trials):
        zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
        if zone is not None:
            last_known_zone = zone
        trial_rz_label.append(zone if zone is not None else last_known_zone)

    if trial_rz_label[0] is None:
        for t in range(n_trials):
            if trial_rz_label[t] is not None:
                for tt in range(t):
                    trial_rz_label[tt] = trial_rz_label[t]
                break
```

iii. Step 4 discrepancy table: "NWB has no scene info ... Must determine reward zone from position where rz>0". Step 5 decision 6: "Reward zone identification: Infer from position where reward_zone>0 in NWB data. Map to A (80-130), B (200-250), C (320-370) based on position range." The zone coordinates come from the paper/reference code (Step 3 table). The AI separately established from the data (trajectory step 45–46) that "the reward zone switches after 30 trials", which is why carry-forward rather than a single per-session label is used.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the signed distance to the nearest edge of the trial's reward zone: `position − rz_start` when before the zone (negative), exactly 0 while inside `[rz_start, rz_end]`, and `position − rz_end` when past it (positive). This continuous distance is then discretized (see 7-c) and written as row 0 of the (6, n_timepoints) integer output array.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start  # negative
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end  # positive
    return distance
```
```python
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
        dist_bins = discretize_distance(dist_to_rz)
        ...
        trial_output[0, :] = dist_bins
```

iii. Step 5 decision 7: "Distance to reward zone: Signed distance from animal position to nearest edge of reward zone. Negative = before zone, positive = past zone, 0 = within zone." This follows the decoder spec's requirement for "distance to *any* location in the reward zone" (hence the flat 0 inside the 50 cm zone rather than distance to the zone centre).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit boolean masks rather than `np.digitize`: bin 0 = d < −50; 1 = −50 ≤ d < −10; 2 = −10 ≤ d < 0; 3 = d == 0 (in the zone); 4 = 0 < d ≤ 10; 5 = 10 < d ≤ 50; 6 = d > 50. The resulting distribution over the whole dataset is 0.253 / 0.102 / 0.074 / 0.237 / 0.021 / 0.072 / 0.242, identical to the reference distribution to three decimals.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3  # in reward zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```
```python
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm (in zone)', '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
```

iii. The edges are transcribed directly from the Decoder Task specification (reproduced in the Step 5 mapping table: "Bins: <-50, -50 to -10, -10 to 0, 0, >0 to +10, +10 to +50, >+50"). The exact-zero test for bin 3 is safe because `compute_distance_to_reward_zone` assigns literal `0.0` inside the zone rather than a computed value.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[start:end]`, the same slice indices used for the neural matrix, so it is sample-for-sample aligned by construction and has exactly `n_timepoints` entries. No lag, shift or interpolation is applied.

ii.
```python
        start = trial_starts[t]
        end = teleports[t]
        trial_neural = neural_all[start:end, :].T.copy()
        ...
        trial_pos = position[start:end]
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. Same basis as 2-d/3-c: behaviour and imaging share one frame index in the NWB file (Step 4 conclusion 1), with the length mismatch in 10 multi-plane sessions resolved by truncation. The `--show-processing` plots overlay position, continuous distance and the discretized bins on a common time axis to check for misalignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioural time series alone (cm along the 450 cm virtual track; −500 before the TTL, but pre-TTL samples never fall inside a trial).

ii.
```python
        position = behav['position']['data'][:]
```
```python
        trial_pos = position[start:end]
        pos_bins = discretize_position(trial_pos)
```

iii. Step 2: "`position`: cm on track (0-450), -500 pre-TTL"; Step 5 mapping table: "Absolute position | output[1] | Discretized to 5 equal bins (90 cm each), time-varying | `position` in NWB".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No transformation of the values themselves — the raw per-trial position slice is discretized directly (see 8-c) and stored as row 1 of the output array. The only implicit handling is the clipping in the discretizer, which absorbs the handful of samples marginally below 0 cm or above 450 cm into the end bins.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. Step 3 records the paper's "450 cm linear track" and "45 bins of 10 cm each"; the decoder spec asks for 5 equal bins over that track, so the AI uses the raw cm values without rescaling.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `floor(position / 90)` clipped to [0, 4], i.e. bins 0–90, 90–180, 180–270, 270–360, 360–450 cm, with the first and last bins open-ended so out-of-range samples do not create extra classes. Resulting distribution 0.211 / 0.178 / 0.231 / 0.227 / 0.154, identical to the reference.

ii.
```python
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)  # [0, 90, 180, 270, 360, 450]
```
```python
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```
```python
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
```

iii. Directly from the Decoder Task spec ("Discretized into 5 equal-sized bins spanning the 450 cm track"), recorded in the Step 5 mapping table. `TRACK_LENGTH = 450.0` is taken from the paper (Step 3 table).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `start:end` slice as the neural data; length `n_timepoints`; no shift or resampling.

ii.
```python
        trial_pos = position[start:end]
        pos_bins = discretize_position(trial_pos)
        trial_output[1, :] = pos_bins
```

iii. As in 7-d: neural and behavioural channels are on the same NWB frame index after the common-length truncation; the processing plots show position, neural traces and the discretized bins on one time axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioural time series, which holds the cumulative lick count within each imaging frame (integer values 0–7).

ii.
```python
        lick = behav['lick']['data'][:]
```
```python
    lick_binary = np.clip(lick, 0, 1).astype(np.float64)
```

iii. Step 2: "`lick`: cumulative lick count per frame"; Step 4 resolution: "Lick data | Cumulative count then binarized | Cumulative count in NWB `lick` | Binary after correction | Need to binarize: clip to 0/1".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) Binarisation: `np.clip(lick, 0, 1)`, so any frame with ≥1 lick becomes 1. (2) A lick-sensor artifact correction copied from the reference code: within each trial, if more than 35% of samples have a cumulative lick count > 2, the whole trial's lick channel is set to NaN. Because the output array is integer, those NaNs are then converted to 0 at write time — so the 69 affected trials (0.6% of trials) are saved as "no lick" rather than being excluded or flagged. The overall lick fraction is 0.220 versus the reference's 0.230.

ii.
```python
    lick_binary = np.clip(lick, 0, 1).astype(np.float64)

    # Lick error correction per trial
    lick_error_trials = []
    for t in range(n_trials):
        trial_lick = lick[start:end]
        if len(trial_lick) > 0:
            frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
            if frac_bad > LICK_ERROR_FRACTION:
                lick_binary[start:end] = np.nan
                lick_error_trials.append(t)
```
```python
        # [3] Lick (time-varying, binary)
        lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
        trial_output[3, :] = lick_vals
```

iii. Step 1: "Lick error correction: If >35% of samples in a trial have cumulative lick count >2, lick data for that trial set to NaN" and "Lick binarization: After correction, licks > 1 set to 1 (binary)", both attributed to the reference `get_timeseries_data()` in `glmUtils.py`. Step 3 records the paper's check value "n = 81 out of 12,376 trials removed"; Step 10 reports 69 such trials in the converted data and attributes the shortfall to the paper's count including the fixed-condition mice. (Note: Step 6 also claims "Lick binarization: `diff(cumulative_lick) > 0` per frame", which the code does not do — the clip is the correct operation for a per-frame count.)

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The corrected/binarized lick vector is sliced with the same `start:end` indices as the neural matrix, giving one lick value per neural timepoint. No shifting or event-time mapping is involved because the lick channel is already sampled on the imaging frame clock.

ii.
```python
        trial_lick = lick_binary[start:end]
        ...
        lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. Same justification as the other behavioural channels: Step 2 establishes that all `BehavioralTimeSeries` channels are stored "at imaging frame rate ~15.5 Hz", so index alignment is sufficient.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Identical to 7-a: the `reward_zone` channel combined with `position`, resolved to one of the three canonical zones per trial by nearest-centre matching of the mean in-zone position, with carry-forward/back-fill for trials in which the zone is never entered.

ii.
```python
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
```
```python
        zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
        if zone is not None:
            last_known_zone = zone
        trial_rz_label.append(zone if zone is not None else last_known_zone)
```

iii. See 7-a. Step 4 records that the NWB release lacks the scene strings the reference code used (`get_reward_zones()`), so the label must be inferred from behaviour; Step 5 decision 6 states the inference rule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The A/B/C label is mapped to 0/1/2 and broadcast across all timepoints of the trial as row 4 of the output array (per-trial value stored in time-varying form). Trials whose label could not be resolved at all fall back to zone B (200, 250) — a defensive branch that never triggers on this dataset. Resulting distribution 0.328 / 0.337 / 0.335, essentially identical to the reference (0.329 / 0.337 / 0.334).

ii.
```python
    for t in range(n_trials):
        if trial_rz_label[t] is not None:
            trial_rz_start[t], trial_rz_end[t] = REWARD_ZONES[trial_rz_label[t]]
        else:
            # Fallback - shouldn't happen
            trial_rz_start[t], trial_rz_end[t] = 200, 250

    rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
    trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```
```python
        rz_loc = trial_rz_idx[t]
        trial_output[4, :] = rz_loc  # broadcast per-trial
```

iii. The decoder spec requires "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the instructions also ask for outputs to be time-varying "if at all possible", which is why the per-trial label is broadcast. Step 10 checks the A/B/C balance ("34.3/32.8/32.9%") against the expectation of a counterbalanced design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event series' `timestamps`, mapped to behavioural frame indices via `np.searchsorted` on the `position` timestamps (with clipping to the valid index range). The `Reward` `data` values (all 0.004 mL) are read but not used, and the `autoreward` channel is not read at all.

ii.
```python
    reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
    reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

iii. Step 2: "Reward events stored separately with timestamps (not at frame rate)"; Step 6: "Reward detection: Reward event timestamps mapped to frame indices using NWB timestamps." Step 5 mapping table: "Reward outcome | output[5] | Binary per trial | From Reward events | 0=no reward, 1=rewarded".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if at least one reward frame index lies in `[trial_start, teleport)`, otherwise 0. The flag is broadcast across all timepoints of the trial as row 5 of the output array (and reused, shifted by one trial, for input 3). Unlike the reference, there is no assertion that a reward timestamp lands within half a time bin of the matched frame; out-of-range reward times are silently clipped to the last frame. Resulting distribution 0.157 no-reward / 0.843 reward, identical to the reference.

ii.
```python
    trial_rewarded = np.zeros(n_trials, dtype=bool)
    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
        trial_rewarded[t] = trial_rewards
```
```python
        reward_out = int(trial_rewarded[t])
        trial_output[5, :] = reward_out  # broadcast per-trial
```

iii. The decoder spec asks for a per-trial binary reward outcome. Step 7 notes an early sanity check on the omission rate ("the reward omission rate is ~9.5% on the sample ... the paper counts omission differently"), and Step 10 reports the full-dataset omission rate of 15.3%, matching the paper's "randomly omitted on ~15% of trials".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases are handled, all silently or with a printed warning rather than by dropping data:
- **Neural/behaviour length mismatch**: all behavioural and ophys arrays are truncated to `min(n_behav, n_neural)` (affects 10 m17/m18 sessions where the imaging array is one frame longer).
- **Unpaired trial boundaries**: trial-start and teleport index vectors are truncated to the shorter one, and pairs where the teleport does not follow the start are discarded.
- **Degenerate trials/sessions**: trials shorter than 5 samples are skipped; sessions with fewer than 2 surviving trials are dropped. Neither fires on this dataset.
- **NaN in neural data**: replaced by 0 per trial.
- **Trials with no reward-zone entry**: the zone label is carried forward from the previous trial (back-filled for leading trials); an unreachable fallback assigns zone B.
- **Lick-sensor artifacts**: affected trials are NaN-ed, then written out as 0.
- **Reward timestamps outside the behavioural clock**: `searchsorted` result clipped into range, without an error-magnitude check.
- Degenerate dF/F cases are guarded too: trials with <10 samples or out-of-range indices are skipped in `compute_dff`, near-zero baselines floored at 1e-10, and sessions with <100 valid samples return no interneurons.

ii.
```python
    n_samples = min(n_behav_samples, n_neural_samples)
    if n_behav_samples != n_neural_samples:
        position = position[:n_samples]
        ...
        deconv_data = deconv_data[:n_samples, :]
```
```python
        if n_timepoints < 5:
            continue  # Skip very short trials
        trial_neural = neural_all[start:end, :].T.copy()
        trial_neural[np.isnan(trial_neural)] = 0
```
```python
        if start >= n_samples or stop >= n_samples:
            continue
        if len(trial_F) < 10:
            continue
        ...
        abs_baseline[abs_baseline < 1e-10] = 1e-10
```

iii. Step 6: "Shape mismatch fix: For multi-plane animals, behavioral and neural data can differ by 1 frame; truncated to common length." The NaN-to-zero substitution carries the in-code comment "following reference: X[np.isnan(X)] = 0". Step 10's edge-case section reports the results of the checks: "No NaN in neural data; No negative values in neural data (deconvolved events >= 0); Trial lengths: min=96, max=3359" — i.e. the AI verified that the degenerate-trial guards were not silently discarding anything.

## 13-a. What are the most time-consuming steps of the code?

i. The script prints per-session timings for load / dF/F / interneuron detection and a grand total, so the profile is measurable from `conversion_full_out.txt` (152 sessions, 879.0 s total):
1. **dF/F computation** — 544.7 s (62%), 2.4–9.4 s per session. This is the dominant cost even though the result is only used to detect interneurons.
2. **NWB reading** (`h5py` slurp of Deconvolved + Fluorescence + Neuropil + behaviour) — 217.4 s (25%).
3. **Interneuron correlation** — 61.5 s (7%), after a 10× vectorisation speed-up.
4. **Pickle write** of the 9.4 GB output — ≈18 s (the residual between the summed session times, 860.7 s, and the 879.0 s total).
The per-trial Python loops that build the input/output arrays are negligible by comparison.

ii.
```python
    t_load = time.time() - t0
    ...
    t1 = time.time()
    dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
    t_dff = time.time() - t1
    t1 = time.time()
    is_interneuron = identify_interneurons(dff, speed, iscell)
    t_int = time.time() - t1
```
```python
    for info in data['metadata']['session_info']:
        print(f"  {info['subject']} ses-{info['session']}: "
              f"load={info['load_time']:.1f}s, dff={info['dff_time']:.1f}s, "
              f"int={info['interneuron_time']:.1f}s, total={info['total_time']:.1f}s")
```

iii. Step 6: "Performance Optimization — Vectorized interneuron detection (matrix correlation instead of per-neuron loop): 4.3s -> 0.4s per session. Full conversion: ~879s (~14.6 min)". Trajectory step 79: "Interneuron detection went from 4.3s to 0.4s for m3 ... The dFF computation is now the main bottleneck (6.4s). The main dFF bottleneck is the per-trial filtering which is hard [to avoid]". The AI extrapolated sample timings to estimate the full run against the instructions' 15-minute budget before launching it.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Two were: the per-neuron Pearson correlation in `identify_interneurons` became a single matrix operation (4.3 s → 0.4 s per session), and two per-trial loops inside `compute_dff` were merged into one. What remains:
- `compute_dff` still loops over trials to apply `gaussian_filter1d` / `minimum_filter1d` / `maximum_filter1d`; the filters could run once on the whole session with NaN-masked inter-trial gaps, or trials could be batched.
- `process_session` walks the trial list **six separate times** (reward detection, reward-zone identification, zone-coordinate assignment, lick-error correction, environment median, previous-outcome shift) before the main construction loop. All six are trivially fusible into the main loop or into segment-wise vectorised operations (e.g. `np.add.reduceat` / `np.searchsorted` over the trial boundaries).
- `identify_reward_zone` re-slices the full `position` and `reward_zone` arrays per trial and then loops over the three zones in Python; the zone match is a 3-element `argmin`.
- The per-trial discretization calls (`discretize_distance`, `discretize_position`, `discretize_speed`) could be applied once to the whole session array and then sliced.

ii.
```python
    # Vectorized Pearson correlation
    X = dff_cells[valid, :]  # (n_valid, n_neurons)
    Y = speed[valid]
    cov_XY = (X_centered * Y_centered[:, np.newaxis]).sum(axis=0) / (n - 1)
```
```python
    for i, (start, stop) in enumerate(zip(trial_starts, teleports)):
        ...
        smoothed = gaussian_filter1d(trial_F, sigma=15, axis=0)
```
```python
    for t in range(n_trials): ...   # trial_rewarded
    for t in range(n_trials): ...   # identify_reward_zone
    for t in range(n_trials): ...   # trial_rz_start/end
    for t in range(n_trials): ...   # lick error
    for t in range(n_trials): ...   # trial_env
    for t in range(1, n_trials): ... # prev_trial_outcome
    for t in range(n_trials): ...   # main construction loop
```

iii. Step 6 "Code speedups added" documents the two optimisations that were made and the motivation (the instructions' 15-minute target; trajectory step 72: "The estimated time exceeds 15 minutes. I need to optimize the dF/F computation and interneuron detection"). The AI judged the remaining per-trial filtering "hard" to vectorise because of variable trial lengths, and did not document the six redundant trial passes.

## 13-c. What processing does the code repeat multiple times?

i. The conversion is a single pass over the NWB files — each file is opened once, and the reward-zone inference, interneuron detection and trial construction all happen inside that one pass (no separate survey pass, and no intermediate files). The repetition that does exist is within a session:
- the seven successive trial loops listed in 13-b each re-slice the same behavioural arrays by trial;
- `identify_reward_zone` recomputes `position[start:end]` and `reward_zone[start:end]` that the main loop slices again;
- in `--show-processing` mode `plot_processing` recomputes `compute_distance_to_reward_zone` for the displayed trial even though the discretized result is already available;
- the module-level `POS_BIN_EDGES`, `SPEED_BINS` and `DIST_BINS` constants are defined but never used — the discretizers re-express the same edges as inline literals.

ii.
```python
    dist = compute_distance_to_reward_zone(position[start:end], trial_rz_start[t], trial_rz_end[t])
```
```python
SPEED_BINS = [0, 2, 10, 20, 40, np.inf]  # defined but unused
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)  # defined but unused
DIST_BINS = [-np.inf, -50, -10, 0, 0, 10, 50, np.inf]  # defined but unused
```

iii. Not discussed in CONVERSION_NOTES. The single-pass design is implicit in `process_session`, which returns the finished per-trial lists directly to `build_dataset`; the AI's only efficiency commentary (Step 6) concerns the interneuron and dF/F loops.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, the first of which is the single largest cost in the pipeline:
- **dF/F is computed for every ROI, not just the curated cells.** Only 44.4% of ROIs pass `iscell` (138,678 of 312,110), and `identify_interneurons` immediately subsets to those. Roughly 56% of the 544.7 s spent in `compute_dff` (≈300 s, a third of the whole run) is spent on ROIs that are thrown away.
- **The dF/F itself is discarded** after being used solely to exclude 385 cells (0.28%); it does not contribute to the saved neural data (see 2-a/2-b).
- **`Fluorescence` and `Neuropil` are read in full for every plane** (doubling the ~217 s of I/O relative to reading only `Deconvolved` plus the curated columns), then dropped.
- **Loaded-but-never-used variables**: `scanning`, `trial number`, `reward_data` (reward amounts), `plane_idx`, and the per-file `imaging_rate` (read into `session_info` but superseded by the hard-coded constant). `scanning` and `trial number` are even truncated in the length-mismatch branch before being ignored.
- **Per-trial scalars are broadcast to full time series**: environment, trial number, previous outcome, reward-zone location and reward outcome are each stored as `n_timepoints` identical values, which inflates the pickle (9.4 GB) though the format does permit per-trial vectors.
- The `rz_counts` placeholder line in `plot_processing` computes a constant list that is never used.

ii.
```python
    dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)   # all ROIs
    ...
    dff_cells = dff[:, iscell_mask]   # only iscell columns are ever used
```
```python
        trial_number = behav['trial number']['data'][:]   # never used
        scanning = behav['scanning']['data'][:]           # never used
        reward_data = behav['Reward']['data'][:]          # never used
        plane_idx = seg['planeIdx'][:]                    # never used
        imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]  # only stored
```
```python
    rz_counts = [sum(1 for lbl in [None]*0), 0, 0]  # placeholder
```

iii. Not identified in CONVERSION_NOTES. The AI's efficiency review (Step 6, trajectory steps 71–79) focused only on the cost of the two bottleneck routines, not on whether their inputs or outputs were needed; the broadcasting of per-trial values is a deliberate choice, justified by the instruction "If at all possible, make it time-varying".
