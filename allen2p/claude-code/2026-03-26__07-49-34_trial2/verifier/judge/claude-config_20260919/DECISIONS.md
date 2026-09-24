# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK object API. It reads the local NWB files directly with `h5py`, and uses the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) as the index of what is available. The set of experiments processed is the intersection of (a) the 284 NWB files present on disk and (b) rows of the experiment table, further restricted to `passive == False`. This yields **202 active experiments from 38 mice** (168 single-plane `VisualBehavior` + 34 `VisualBehaviorMultiscope` planes). No filter on `project_code` is applied, so both the single-plane and the mesoscope projects are included. Each NWB file is opened once per pass (two passes total) and every needed stream (dF/F + ophys timestamps, cell_specimen_table, stimulus presentations, trials, running speed, eye tracking) is read from it.

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
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]   # (timepoints, neurons)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
    stim = f['intervals'][stim_key]
    trials = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][:]
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
```

iii. From CONVERSION_NOTES Step 4 and the trajectory: the AI established that the 284 local NWB files are a subset of the 1,936 released experiments, that "each NWB file = one imaging plane from one session", and that dF/F is already pre-computed in the NWB so the SDK object layer adds nothing it needs. Passive experiments were excluded because "Passive sessions have no meaningful trial outcomes (no licking)" (Step 5, Key Decision 3; trajectory step 31: "passive sessions lack the trial outcomes (Hit/Miss/FA/CR) that the task requires, so I should focus on active sessions only"). It never explicitly considered restricting to `project_code == 'VisualBehavior'`; it instead read "Visual Behavior" as the whole Visual Behavior 2P dataset and handled the two imaging rigs by re-binning in time (see 2-e).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the 202 active experiments (38 mice), taken from the experiment metadata table, sorted as strings. `subject_idx` for each session is the index of that experiment's `mouse_id` in this list.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
        'mouse_id': str(exp_info['mouse_id']),
...
        subject_idx = subjects_list.index(result['mouse_id'])
        all_subject_idx.append(subject_idx)
```

iii. Not discussed at length; the AI simply treated the metadata `mouse_id` column as the animal identifier (Step 2 of CONVERSION_NOTES records "Unique subjects in metadata | 107" and "38 in our active subset"), and cross-checked 38 mice against the 82 mice of the full release, attributing the difference to the on-disk subset.

## 1-c. How are the data split into sessions?

i. **One NWB experiment (i.e. one imaging plane) is treated as one "session"** in the output. Experiments are ordered by `ophys_experiment_id`; no grouping by `ophys_session_id` is done. Consequently the 174 real behavioural sessions covered by the 202 active experiments become 202 output sessions: the five `VisualBehaviorMultiscope` sessions appear 3–7 times each (once per plane), each copy carrying the same trials and the same behavioural outputs but a different subset of neurons. This is visible in the verification log as repeated trial counts (`... 209, 209, 209, 209, 209, 209, 209, ...`) and as one mouse with 34 sessions.

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
```

iii. Trajectory step 31: "Each experiment = one imaging plane... For multiscope sessions, one session can have multiple experiments (planes)... Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session". The AI noticed the consequence in step 79 ("Subject 457841 has 34 sessions, which seems high... 5 actual MESO sessions could generate 34 'sessions' in our output if each has 7 planes") but did not change the design. A technical motivation it did not state explicitly is that mesoscope planes have *different* ophys timestamps, so they cannot be concatenated without resampling.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go | catch) & ~aborted & ~auto_rewarded`. The trial's time support is **not** taken from `start_time`/`stop_time`; instead it is the set of stimulus presentations whose `trials_id` equals the trial id, i.e. every 750 ms image flash belonging to that trial (verified: `trials_id` partitions all flashes and spans `start_time` to `stop_time`). Each flash becomes one time bin, so trials are variable length (mean 11.6 bins ≈ 8.7 s; typically 1–17 flashes). Result: 51,992 trials over 202 sessions (mean 257/session).

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

iii. CONVERSION_NOTES Step 3/5: "Per task instructions: Include Go and Catch trials, exclude Aborted and Auto-rewarded". Trajectory step 41: the flash is "the natural unit of the experiment", so defining the trial as its constituent flashes makes the trial boundaries and the time bins consistent by construction, and it means the trial includes both the pre-change flashes and the post-change response window.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the go/catch, non-aborted, non-auto-rewarded mask there is almost no trial-level QC:
- trials with zero associated stimulus presentations are skipped;
- experiments with 0 valid ROIs, or with no natural-image stimulus block, are skipped entirely;
- sessions with fewer than 2 surviving trials are skipped (0 sessions were actually dropped: min 39 trials).
No `change_time`-validity check, no engagement/d-prime filter, no per-session behavioural QC is applied. There is also no `try/except` around per-experiment processing, so a corrupt file would abort the whole run.

ii.
```python
        if n_neurons == 0:
            print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
            return None
        stim_key = find_stim_key(f)
        if stim_key is None:
            print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
            return None
...
            if len(trial_stim_indices) == 0:
                continue
...
        if result is None or result['n_trials'] < 2:
            skipped += 1
```

iii. CONVERSION_NOTES Step 3 lists the whitepaper's session QC (z-drift, d-prime ≥ 1, etc.) and notes the paper's extra exclusion of "images where licking bout already ongoing", but the AI applied neither, on the grounds that the released dataset has already passed the Allen QC pipeline and the task instruction names only the four trial types. Step 10 Check 5 records "Minimum trials per session: 39 (above the 2-trial minimum)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. dF/F traces stored in the NWB at `processing/ophys/dff/traces/data` (shape `(timepoints, ROIs)`), with their own `timestamps` used as the ophys time base, plus `processing/ophys/image_segmentation/cell_specimen_table/valid_roi` used as a neuron mask. Detected calcium `events` (used by Piet et al.) were deliberately *not* used.

ii.
```python
        dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
        ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
        valid_roi = cell_table['valid_roi'][:].astype(bool)
        dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
```

iii. CONVERSION_NOTES Step 1: "dF/F is pre-computed in NWB files — no need to compute from raw fluorescence"; Step 5 Key Decision 2: "Use dF/F (not events): dF/F is the standard neural signal for decoding; pre-computed in NWB; events are sparser and may not decode as well for time-varying outputs."

## 2-b. How is the `neural` data processed?

i. Three operations: (1) restrict to `valid_roi` columns; (2) build a running cumulative sum over time; (3) for each 750 ms stimulus-presentation bin, take the **mean dF/F over all ophys frames whose timestamp falls in `[flash_onset, flash_onset + 0.75 s)`**. No z-scoring, baseline subtraction, deconvolution, smoothing or neuron-level normalisation is applied (dF/F from the Allen pipeline is already neuropil-corrected and baseline-normalised). Neurons from different planes of the same session are never merged (see 1-c).

ii.
```python
        dff_cumsum = np.cumsum(dff_valid, axis=0)
        dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
        ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
        ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
...
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                s, e = ophys_bin_starts[si], ophys_bin_ends[si]
                n_frames = e - s
                if n_frames > 0:
                    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. CONVERSION_NOTES Step 5: dF/F "Average within 750ms stimulus bins". The cumsum trick was introduced in Step 6 as an optimisation ("Cumulative sum-based bin averaging (avoids per-bin boolean masking)"). Averaging over the whole 750 ms window (250 ms image + 500 ms grey) was justified in trajectory step 41 on the grounds that the grey-period activity "is still responding to the previous image as a sustained or decaying response, which is actually useful for decoding."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the SDK-equivalent `valid_roi` mask is applied (plus dropping experiments with 0 surviving ROIs). No SNR, event-rate or activity-based neuron filtering. In practice this filter is a **no-op on this release**: across all 202 active files, 29,444/29,444 ROIs have `valid_roi == True` (independently verified), so the converted neuron count equals the raw ROI count.

ii.
```python
        valid_roi = cell_table['valid_roi'][:].astype(bool)
        n_total_rois = dff_data.shape[1]
        dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
        n_neurons = dff_valid.shape[1]
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "valid_roi filter: Use only cells marked valid_roi=True in NWB, matching SDK default behavior", mapped in Step 10 Check 3 to the SDK's `CellSpecimens(exclude_invalid_rois=True)`. Step 3 documents what the Allen classifier excludes (unions, duplicates, edge ROIs, dendrites, ghost cells), which the AI took as sufficient neuron curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **stimulus (image flash) onset**, not to trial start or change time: bin *k* of a trial covers `[start_time(flash_k), start_time(flash_k)+0.75 s)` of the ophys clock, and the ophys frames inside that window are averaged. Because the first flash of a trial begins ~20 ms after `trials.start_time`, this is effectively trial-start-aligned at bin 0 as well. Every output stream is binned on exactly the same flash boundaries, so neural and outputs are aligned by construction. `metadata['temporal_alignment_event'] = 'Stimulus presentation onset (each 750ms image flash)'`, with `off_start = off_end = None` (variable-length trials, no fixed window).

ii.
```python
        bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
        all_stim_starts = stim_start
        all_stim_ends = all_stim_starts + bin_duration
        ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
        ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')
```
```python
            'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
            'off_start': None,
            'off_end': None,
```

iii. Trajectory step 41: binning on flash onsets makes image identity constant within a bin, puts the change event exactly at a bin boundary, and gives a single alignment rule that works for both the 31 Hz and 11 Hz rigs. The instruction "temporally align based on ophys timestamp" was satisfied by using the dF/F traces' own `timestamps` array as the clock into which the flash windows are searched.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — the data are rebinned.** The native ophys rate is ~30.94 Hz (CAM2P single-plane, 168 experiments) or ~10.73 Hz (MESO, 34 experiments). All streams are resampled to a uniform **750 ms bin = one stimulus presentation** (250 ms image + 500 ms grey), giving ~11.6 bins per trial and `metadata['time_bin_size'] = 750.0`. Rebinning is by simple mean over the samples of each stream falling inside the bin.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
            'time_bin_size': TIME_BIN_MS,
```

iii. Trajectory steps 35/39/41 show an extended deliberation: the format demands a single `time_bin_size` for the whole dataset, but the dataset mixes 31 Hz and 11 Hz recordings; upsampling MESO to 31 Hz "introduce[s] interpolation artifacts", downsampling to 11 Hz discards resolution, so the AI chose the flash interval, which is "1. Consistent across all equipment types 2. Aligned to the natural task structure 3. Each time bin corresponds to one image presentation". It explicitly acknowledged the cost ("too coarse for capturing the temporal dynamics the decoder needs — onset transients, sustained responses, offset effects all get averaged together") and went ahead anyway.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The `image_name` column of the natural-images stimulus presentations table (`intervals/<Natural_Images_...>_presentations/image_name`), selected per flash via `trials_id`. The stimulus block key is found by skipping `trials`, `spontaneous*` and `natural_movie*`. The trials table's `initial_image_name` / `change_image_name` are *not* used.

ii.
```python
def find_stim_key(f):
    """Find the stimulus presentations key for change detection task."""
    for key in f['intervals']:
        if key == 'trials' or key.startswith('spontaneous') or key.startswith('natural_movie'):
            continue
        return key
    return None
...
        stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                    for x in stim['image_name'][:]])
...
            images = stim_image_name[trial_stim_indices].copy()
```

iii. CONVERSION_NOTES Step 5 variable mapping: `image_name → output[0]`, source `stimulus_presentations`. Using the presentations table gives the identity of the image actually on screen at every flash (including mid-trial repeats and omissions) rather than inferring it from the trial's initial/change labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. (1) Omitted flashes (`image_name == 'omitted'`, ~4–5 % of presentations) are **forward-filled** with the previous image (back-filled if the omission is the first flash of a trial); (2) names are mapped to integers with a global mapping over all image names found in the dataset — **16 codes**, because sessions use image set A or B (8 each); (3) the code is emitted once per 750 ms bin, i.e. time-varying. The global image list is built by scanning only the **first 10 experiments**, and unknown names fall back silently to code 0 (`img_to_idx.get(img, 0)`). I verified independently that the full active set contains exactly those 16 images, so nothing was mis-coded here — but the shortcut is not robust in general.

ii.
```python
def collect_all_image_names(active_exps, nwb_map):
    """Collect all unique image names across experiments (sample a few to be fast)."""
    all_images = set()
    sample_exps = active_exps.head(min(10, len(active_exps)))
    ...
    return sorted(all_images)
```
```python
            # Forward-fill omitted presentations
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

iii. Step 5 Key Decision 5: "Omitted stimuli: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions." Trajectory step 41 weighed dropping omissions versus forward-filling and chose forward-fill to keep the bin grid uniform. Trajectory step 73 justifies the global 16-class code: "global encoding is correct since the decoder needs consistent output dimensions" (the decoder weights unseen classes to zero per session).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. One image code per stimulus bin, using exactly the same `trial_stim_indices` that define the neural bins — so alignment is identity-by-construction; the image code in bin *k* labels the image whose onset starts bin *k*'s neural averaging window.

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

iii. Documented in Step 10 Check 3 as "Temporal alignment: Average dF/F in 750ms stimulus bins ... Consistent", and spot-checked in Step 10 Check 2 ("Image ID Match: True") for 3 sessions against the raw NWB.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the stimulus presentations table (float 0/1, NaN-safe), per flash. The trials table's `change_time` is loaded but not used for this.

ii.
```python
        stim_is_change = stim['is_change'][:]
...
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Step 5 mapping: "`is_change` → output[1], Binary (0/1), 1 at change flash only". CONVERSION_NOTES Step 1 notes the SDK's `is_change_event()` in `stimulus_processing.py` as the reference implementation of this flag, so the AI took the pre-computed column rather than re-deriving it.

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond `nan_to_num` and casting to int. Because `is_change` is a per-flash flag and a bin *is* a flash, the variable is 1 for exactly the single 750 ms bin in which the image changed and 0 elsewhere. Catch ("sham change") trials have `is_change == False` throughout, so they are all-zero. Overall 7.5 % of bins are 1; the AI verified 45,477 go trials each with exactly one change bin and 6,515 catch trials with none (87.5 % / 12.5 %, matching the 7/8 – 1/8 transition matrix).

ii.
```python
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Step 10 Check 4: "Go trials all have change → 45,477 with change, 0 without; Catch trials no change → 0 with change, 6,515 without — Exact". Trajectory step 87: "Go/Catch numbers ... 45,477 + 6,515 = 51,992 ... that's the expected 7/8 and 1/8 split".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding — the variable is natively binary with value names `['no_change', 'change']`.

ii.
```python
    change_value_names = ['no_change', 'change']
```

iii. Task instruction: "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same flash-bin indexing as the neural data (`trial_stim_indices`), so the change bin is precisely the bin whose neural average covers `[change_flash_onset, +750 ms)` — i.e. the change flag marks the flash at which the change occurred and the neural response to it, which is the "right after a change" bin.

ii.
```python
            trial_stim_indices = np.where(stim_trials_id == tid)[0]
            ...
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```
(spot-checked output for one trial: `image_identity = [5 5 5 5 14 14 ...]`, `image_change = [0 0 0 0 1 0 ...]` — the change flag sits exactly at the identity transition.)

iii. Step 10 Check 2 recorded "Change Match: True" for 3 sessions checked directly against the NWB.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` (~60 Hz) — the Allen-filtered running speed (10 Hz low-pass Butterworth, transient removal), i.e. the same series the SDK exposes as `running_speed`, not `speed_unfiltered`.

ii.
```python
        running_speed = f['processing']['running']['speed']['data'][:]
        running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. Step 1 notes "Running speed has both raw and 10 Hz lowpass Butterworth filtered versions"; Step 5 maps "running speed → output[2] ... 10 Hz Butterworth filtered", and Step 10 Check 3 equates it with `RunningSpeed.from_nwb()`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Mean of all running samples in each 750 ms flash window (again via cumsum + `searchsorted`), then discretisation into 5 global percentile bins. Bin edges are computed in a **first pass over all 202 experiments** on the binned (not raw-sample) values, so the edges are global across the dataset, not per-session.

ii.
```python
        run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
        run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
        run_cumsum = np.concatenate([[0], np.cumsum(running_speed)])
...
            running_binned = np.zeros(n_bins, dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                s, e = run_bin_starts[si], run_bin_ends[si]
                n_pts = e - s
                if n_pts > 0:
                    running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. Step 5 Key Decision 7: "Discretization: Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins." Global edges keep the five categories comparable across sessions and mice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (0, 20, 40, 60, 80, 100th percentile). The outermost edges are replaced by ±inf and the result is clipped to [0, 4] so no value can fall outside the five classes. Realised edges: `[-inf, 0.0041, 0.788, 15.82, 33.13, inf]` cm/s; the realised marginal distribution is exactly 20 % per bin.

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
```

iii. Task instruction: "Running speed, discretized into five equal percentile bins." The AI verified uniformity in the sample run ("Running speed bins | 20% each (uniform)").

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running samples are averaged over exactly the same `[flash_onset, +750 ms)` windows used for the dF/F average, indexed by the same `trial_stim_indices`, so the two streams share the bin grid. No interpolation onto the ophys clock is needed because both are binned from their own native timestamps into a common wall-clock window (the NWB streams are hardware-synchronised).

ii.
```python
        all_stim_starts = stim_start
        all_stim_ends = all_stim_starts + bin_duration
        ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
        run_bin_starts  = np.searchsorted(running_ts, all_stim_starts, side='left')
```

iii. Step 10 Check 2 verified the discretised running series of 3 sessions against values recomputed from the raw NWB ("Running Match: True", `np.allclose`, atol 1e-5). I reproduced this independently for experiment 1007107386, trial 0: recomputed bins `[1 1 1 0 2 2 2 1 2 2]` == stored bins.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (ellipse-fit pupil area) with its `timestamps` (~30 Hz), plus `acquisition/EyeTracking/likely_blink/data` as a blink mask. Diameter is computed as the equivalent-circle diameter of that area. Sessions with no `EyeTracking` group are handled (pupil set to NaN).

ii.
```python
        has_eye_tracking = 'EyeTracking' in f['acquisition']
        if has_eye_tracking:
            pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
            pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
            likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. Trajectory step 41 explicitly compared `width` (whitepaper: major axis of the ellipse fit) against `area`, could not settle the semi-axis/full-axis ambiguity of `width`, and concluded: "For simplicity, I'll just compute diameter as 2*sqrt(area/pi) which gives the diameter of a circle with the same area. This is the most common definition."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink frames → NaN (redundant: the NWB `area` column is already NaN at `likely_blink`, verified — NaN fraction == blink fraction); (2) non-positive areas → NaN; (3) `diameter = 2*sqrt(area/pi)`; (4) **linear interpolation over all NaN gaps for the whole session** (`np.interp`, which also flat-extrapolates at the edges); (5) mean within each 750 ms flash bin; (6) any bin still NaN is linearly interpolated within the trial, and if a whole trial/session has no pupil data at all, every bin is assigned the **middle bin (2)**; (7) discretisation into 5 global percentile bins computed in pass 1.

ii.
```python
            pupil_area = pupil_area_raw.copy().astype(float)
            pupil_area[likely_blink] = np.nan
            pupil_area[pupil_area <= 0] = np.nan
            pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
            pupil_diameter = interpolate_nans(pupil_diameter)
```
```python
            nan_mask = np.isnan(pupil_raw)
            if nan_mask.all():
                pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
            else:
                if nan_mask.any():
                    pupil_raw = interpolate_nans(pupil_raw)
                pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. Step 5 Key Decision 6: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging"; Step 4 records the source reasoning ("DeepLabCut, blink detection z>3 → Use area → compute diameter, interpolate NaNs"). The middle-bin default for completely missing pupil data is not discussed in the notes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same routine as running speed: 5 equal-percentile bins with global edges computed over all binned pupil values from pass 1 (NaNs excluded when computing the edges). Realised edges `[-inf, 73.88, 83.84, 92.87, 105.38, inf]` px. Realised marginal: 19.6 / 19.6 / 21.4 / 19.6 / 19.6 % — the excess in bin 2 is the 4 sessions with no eye tracking that were forced to the middle bin.

ii.
```python
    valid_pupil = all_pupil[~np.isnan(all_pupil)]
    if len(valid_pupil) > 0:
        _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
    else:
        pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
```

iii. Task instruction ("Pupil diameter, discretized into five equal percentile bins") plus Step 5 Key Decision 7 (global 2-pass percentile edges).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed: pupil samples are averaged over the same `[flash_onset, +750 ms)` windows, indexed by the same `trial_stim_indices`. A NaN-aware cumsum (value cumsum + valid-count cumsum) is used so that blink-masked samples do not bias the bin mean.

ii.
```python
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

iii. Same rationale as 5-d — one bin grid for all streams removes any possibility of relative shift between neural and behavioural series.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
        trial_hit = trials['hit'][:].astype(bool)
        trial_miss = trials['miss'][:].astype(bool)
        trial_fa = trials['false_alarm'][:].astype(bool)
        trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. Step 5 mapping: "hit/miss/fa/cr → output[4], Categorical (4 classes), trials table, Static per trial". These are the canonical change-detection outcome labels and are well defined for exactly the go/catch, non-aborted, non-auto-rewarded trials the AI keeps.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Priority-ordered mapping hit→0, miss→1, false_alarm→2, correct_reject→3, with a fallback of 1 (miss) if none of the four flags is set. The scalar is then broadcast across all bins of the trial, so the static variable is delivered as a time-varying row. Resulting distribution: 30.3 % hit, 57.2 % miss, 1.7 % FA, 10.8 % CR. (I checked the raw tables: no kept trial actually lacks all four flags, so the "default to miss" branch never fires.)

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
```

iii. The format spec asks for time-varying outputs "if at all possible", so the per-trial label is replicated across bins. The AI flagged the high miss rate as a concern in trajectory step 73 ("The high miss rate concerns me") but resolved it as genuine (mice disengage late in the 60-min session; Step 3 records 72.2 % mean engagement).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases: blinks and non-positive pupil areas → NaN → linear interpolation; sessions with no `EyeTracking` group or an all-NaN pupil trial → every bin forced to the middle pupil bin (affects 4 of 202 sessions, which therefore carry fabricated pupil labels); `is_change` NaN → 0; omitted image flashes → forward/backward fill; empty bins (`n_frames == 0`) → left at 0 for neural/running and NaN for pupil; experiments with 0 valid ROIs or no natural-image block → skipped; trials with no stimulus presentations → skipped; sessions with <2 trials → skipped. Not handled: an unrecognised image name is silently coded as image 0 (`img_to_idx.get(img, 0)`), and there is no `try/except` around per-experiment processing, so a single unreadable file would terminate the run.

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
```
```python
            if nan_mask.all():
                pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
```
```python
            image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Step 5 Key Decisions 5–6 cover omissions and pupil NaNs; Step 10 Check 5 reports "No NaN/Inf in neural data; All output values are valid integers; Minimum trials per session: 39". The middle-bin pupil default and the image-code fallback are undocumented.

## 9-a. What are the most time-consuming steps of the code?

i. The run is I/O-bound on the NWB files (each 260–340 MB): Pass 1 (statistics) took 261.6 s and Pass 2 (full conversion) 246.6 s, 510 s total for 202 experiments. Within each pass, the dominant costs are `dff_data[:]` (reading the entire `(≈140k × n_neurons)` dF/F array into memory) and the per-bin averaging loops. The script prints per-pass timing and a running ETA, and the AI used a sample run to extrapolate (~24 min estimated, 8.5 min actual).

ii.
```python
    for idx, (_, row) in enumerate(active_exps.iterrows()):
        stats = process_experiment(nwb_path, row, collect_stats_only=True)
        ...
        if (idx + 1) % 20 == 0 or idx == len(active_exps) - 1:
            print(f"  Pass 1: {idx+1}/{len(active_exps)} experiments processed")
...
            rate = (idx + 1) / elapsed
            remaining = (len(active_exps) - idx - 1) / rate
            print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
```

iii. Step 7 of CONVERSION_NOTES tabulates the measured sample timings and the extrapolation to 202 sessions; Step 6 lists the optimisations the AI applied (cumsum binning, `searchsorted`, sampling only 10 files for image names). Trajectory step 61: "The main bottleneck appears to be the per-bin loop that searches for ophys frames within each 750ms window, so I should vectorize that operation."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised bin-boundary finding (`searchsorted`) and bin averaging (cumsum differences), but three per-bin Python loops remain inside the per-trial loop — neural, running and pupil averaging — each of which is a pure gather over precomputed index arrays and could be done with a single fancy-indexed cumsum difference for the whole session at once:
`means = (cumsum[ends] - cumsum[starts]) / (ends - starts)[:, None]`. The omitted-image forward-fill is also an element-by-element Python loop (`np.maximum.accumulate` on a valid-index mask would do it), the per-trial `trials_id` lookup is an O(n_trials × n_flashes) `np.where` scan (a single `np.argsort`/`groupby` would be O(n log n)), and `running_values.extend(...tolist())` accumulates ~600k Python floats instead of appending arrays.

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
...
            trial_stim_indices = np.where(stim_trials_id == tid)[0]
```

iii. The AI treated the cumsum rewrite as sufficient ("Same results" after the optimisation, trajectory step 69) because total runtime (8.5 min) already met the 15-minute budget in Step 7, so it stopped optimising. It never revisited the remaining loops.

## 9-c. What processing does the code repeat multiple times?

i. **The entire per-experiment pipeline runs twice.** `process_experiment(..., collect_stats_only=True)` in Pass 1 executes the same code path as Pass 2 — opening the NWB, reading the full dF/F matrix, building the dF/F cumsum, looping over trials and computing `neural_matrix`, image codes, change flags and outcome — and then throws all of it away except the running/pupil lists, because the `collect_stats_only` early-`continue` sits *after* that work. This is ~262 s of the 510 s runtime. In addition: `collect_all_image_names` re-opens 10 of the same files a third time; `discretize_values` is called once per trial (2 × 51,992 calls) rather than once per session-length array; and `img_to_idx` is rebuilt on every experiment.

ii.
```python
            # --- Neural data: vectorized bin averaging ---
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                ...
            # --- Image identity --- ... --- Image change --- ... --- Trial outcome ---
            if collect_stats_only:
                continue
```
```python
        for ot in result['output_trials']:
            running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)
            ...
            pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. The two-pass design is documented and motivated in Step 6 ("1. Pass 1: Collect running speed and pupil statistics for percentile bin computation; 2. Pass 2: Full conversion with discretization using global percentile bins") — global percentile edges genuinely require seeing all data first — but the AI never noted that Pass 1 could have skipped the neural work (or that Pass 2's trial data could have been cached in memory as the human reference does).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Pass-1 waste** (above): full dF/F read, cumsum, per-bin neural averaging, image forward-fill, change flags and outcome computed and discarded for all 202 experiments.
- **Loaded but never used**: `n_total_rois`, `stim_stop`, `stim_omitted`, `trial_start`, `trial_stop`, `trial_change_time`, `valid_trial_ids` (the omission test is done on the image name string instead of the `omitted` column, and trial boundaries come from `trials_id` instead of start/stop).
- **Redundant QC**: `pupil_area[likely_blink] = np.nan` — the NWB `area` column is already NaN on blink frames; and the `valid_roi` mask is a no-op on this release (0/29,444 ROIs invalid).
- **Near-irrelevant transform**: `2*sqrt(area/pi)` is monotonic, and the final output is percentile bins, so the transform changes nothing except through the within-bin averaging order.
- `input_trials` allocates a `(0, n_bins)` array per trial although `input_names` is empty.
- `plot_processing` is only ever exercised for 2 experiments and its `session_idx`/`nwb_path` arguments are unused.

ii.
```python
        n_total_rois = dff_data.shape[1]          # never used
        stim_stop = stim['stop_time'][:]          # never used
        stim_omitted = stim['omitted'][:]         # never used
        trial_start = trials['start_time'][:]     # never used
        trial_stop = trials['stop_time'][:]       # never used
        trial_change_time = trials['change_time'][:]  # never used
        valid_trial_ids = trial_ids[trial_mask]   # never used
...
            pupil_area[likely_blink] = np.nan     # already NaN in the NWB
            pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. None of this is documented as waste; Step 6 only lists the optimisations that were made. The AI's efficiency review stopped once the measured runtime (510 s) came in under the 15-minute instruction threshold.
