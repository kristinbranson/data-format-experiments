# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every recording by globbing for Suite2p plane directories two levels below `/app/data` (`<subject>/<session>/suite2p/plane0`) and sorting the resulting paths. This single glob simultaneously enumerates subjects (first path component), sessions (second path component), and guarantees that only directories that actually contain Suite2p output are treated as sessions. Each session then loads six arrays: `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy` from `suite2p/plane0`, plus `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. `F`, `Fneu`, `motion` and `tstamps` are memory-mapped (`mmap_mode='r'`) and only materialized as needed; sessions are processed one at a time. Trials are not stored in the raw data — they are created by the AI as consecutive 60-s windows (see 1-d). All 41 sessions from all 6 subjects are loaded in `--full` mode; `--sample` takes the first 2 sessions.

ii.
```python
DATA_ROOT = Path('/app/data')

def discover_sessions() -> list[Path]:
    return sorted(DATA_ROOT.glob('*/*/suite2p/plane0'))
```
```python
    session_dir = sp.parent.parent
    subject = session_dir.parent.name
    session_name = session_dir.name
    sid = f'{subject}/{session_name}'

    F = np.load(sp / 'F.npy', mmap_mode='r')
    Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(sp / 'iscell.npy')
    ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
    motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
```
```python
    sessions = discover_sessions()
    if sample:
        sessions = sessions[:2]
    if not sessions:
        raise FileNotFoundError(f'No sessions found under {DATA_ROOT}')
    subjects = sorted({p.parent.parent.parent.name for p in sessions})
```

iii. From CONVERSION_NOTES Step 2 and Step 5: the data tree is a fixed two-level hierarchy (`subject/session`) in which each session holds exactly one imaging plane at `suite2p/plane0/` and behaviour at `move_deve/`. The AI states that sessions are "ordered lexicographically by subject ID and then ISO-format session directory name, which is chronological within subject", so sorting gives a deterministic, chronological ordering. Memory-mapping plus one-session-at-a-time processing is justified in Step 6 as a speed/memory optimization ("Memory-map source arrays and process one session at a time"; "processing all sessions simultaneously would use excessive memory"). `ops.npy` is loaded so that the acquisition rate and Suite2p preprocessing parameters are read from the data rather than hard-coded; `iscell.npy` is loaded so that the paper's stated cell-probability criterion can be enforced/verified.

## 1-b. How are the data split into subjects?

i. Subjects are the top-level directory names under `/app/data`, recovered from the discovered session paths (`p.parent.parent.parent.name`), deduplicated with a set and sorted alphabetically. This yields the 6 mice `jm031, jm032, jm038, jm039, jm040, jm046`. A `subject -> index` map is built and each session records its subject index in `subject_idx`.

ii.
```python
    subjects = sorted({p.parent.parent.parent.name for p in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    ...
        subject_idx.append(subject_to_idx[info['subject']])
    ...
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 documents "Six subject directories: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`", and Step 3 records the paper's "full dataset of 6 mice", so the directory-per-mouse convention is confirmed against the paper. Step 4 further notes that neuron count is constant within a subject across its sessions (221/370/685/746/541/435), which the AI takes as confirmation that each directory is one mouse whose ROIs have been Track2p-reindexed. Deriving subjects from the discovered session paths (rather than listing `/app/data` directly) means only subjects that actually contain usable Suite2p sessions appear.

## 1-c. How are the data split into sessions?

i. One session = one dated subdirectory of a subject that contains `suite2p/plane0`. Sessions are the sorted glob results, so they are ordered by subject then by ISO date. Each session is processed independently by `process_session()` and becomes one element of the top-level `neural`/`input`/`output`/`brain_region_idx` lists. No merging of days and no cross-day (Track2p identity) concatenation is performed. 41 sessions result (7 per mouse except jm040 with 6). A session is rejected (hard error) if it cannot supply at least two complete 60-s trials, and the sampling rate is checked to be 30 Hz.

ii.
```python
    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{sid}: expected 30 Hz, found {fs}')
    if F.shape != Fneu.shape or F.ndim != 2:
        raise ValueError(f'{sid}: incompatible F/Fneu shapes {F.shape}, {Fneu.shape}')
    ...
    if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
        raise ValueError(f'{sid}: fewer than two complete 60-s trials')
```
```python
    for i, sp in enumerate(sessions):
        show = show_processing and i < 2
        n, x, y, r, info = process_session(sp, show)
        neural.append(n); inputs.append(x); outputs.append(y); regions.append(r); infos.append(info)
```

iii. CONVERSION_NOTES Step 2: "Each subject contains dated session directories. Each session has one imaging plane at `suite2p/plane0/` and behavior at `move_deve/`." Step 1 concluded that although the reference code (Track2p) can map identities across days, "cross-day tracking should only be imposed if the supplied processed data/reference methods explicitly use it" — so each daily recording remains its own decoder session. The ≥2-trial guard is justified by the target-format requirement that "There needs to be at least two trials within each session in order to evaluate the decoder performance." The `fs` assertion exists because the elapsed-time input and the 60-s trial length both depend on the 30 Hz rate.

## 1-d. How are the data split into trials?

i. There is no natural trial structure (the recordings are continuous spontaneous-behaviour sessions), so the AI imposes the task-mandated structure: consecutive, non-overlapping 60-second windows = 1800 native frames = 180 binned samples at 3 Hz. The session is first truncated to the largest whole multiple of 1800 native frames that is available in *all* streams (`min(F, Fneu, motion, tstamps)` length), then 10-frame averaged, then reshaped to `(n_trials, 180)`. Any remainder is discarded. This yields 1,081 trials (20 or 30 per session normally, 19 or 29 in the nine sessions whose behaviour stream is short).

ii.
```python
NATIVE_FRAMES_PER_TRIAL = 1800  # 60 s * 30 Hz
AVERAGE_FRAMES = 10
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES
```
```python
    common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
    keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
```
```python
    n_trials = keep_native // NATIVE_FRAMES_PER_TRIAL
    neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                     for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
    input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                    for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
    output_trials = [np.ascontiguousarray(x[None, :], dtype=np.int64)
                     for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. CONVERSION_NOTES Step 4 records "Trial definition … Required task overrides this: split into consecutive non-overlapping 60-s trials, retaining only complete trials after common-stream truncation", and Step 3 notes "Trials … None in the original experiment; recordings are continuous. Trialization into 60-s chunks is required only by the decoder task." Step 5 spells out the arithmetic: "Retain `floor(common_length / 1800) * 1800` native samples, because 1,800 frames = 60 s at 30 Hz. Endpoint samples outside a complete trial are discarded", giving "1,081 trials: jm031 137, jm032 137, jm038 210, jm039 209, jm040 179, jm046 209."

## 1-e. How are trials filtered based on quality controls?

i. No behavioural/quality trial filtering is applied — there is no task structure and the paper describes no trial exclusion. The only exclusions are structural: (a) the incomplete final 60-s window of each session is dropped; (b) because the session is first truncated to the shortest available stream, the nine sessions with shorter motion-energy arrays lose one extra 60-s window each (19 or 29 trials instead of 20 or 30); (c) a session with fewer than two complete trials would raise (never triggered). Every retained trial is then asserted to have the right shape and be finite.

ii.
```python
    if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
        raise ValueError(f'{sid}: fewer than two complete 60-s trials')
    dropped_neural = F.shape[1] - keep_native
    dropped_motion = motion.size - keep_native
```
```python
    for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
        assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)
        assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
        assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
    assert labels.min() == 0 and labels.max() == 4
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": "Original recordings are continuous rather than trial based. No event/reward trial exclusion is described. For conversion, only complete 60-second intervals with all required synchronized streams should be retained." Step 5 Key Decision 5: "**Incomplete endpoints**: Drop incomplete final 60-s windows. This excludes nine trials whose behavior lacks 1-148 samples and prevents fabricated labels." The AI frames this as refusing to fabricate labels rather than as a quality filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p `plane0` outputs `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence). `iscell.npy` supplies the ROI mask and `ops.npy` supplies the neuropil coefficient (`neucoeff`), sampling rate (`fs`), and the baseline parameters (`baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`). The AI explicitly *rejects* `spks.npy` (Suite2p deconvolved activity), which is present in the data.

ii.
```python
    F = np.load(sp / 'F.npy', mmap_mode='r')
    Fneu = np.load(sp / 'Fneu.npy', mmap_mode='r')
    iscell = np.load(sp / 'iscell.npy')
    ops = np.load(sp / 'ops.npy', allow_pickle=True).item()
```

iii. CONVERSION_NOTES Step 3 quotes the paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses." Step 4 therefore resolves the choice: "Reproduce Suite2p default preprocessing: neuropil-correct `F - 0.7*Fneu`, then default maximin baseline correction … use this fluorescence stream, not `spks`." Step 5 Key Decision 1: "Use Suite2p baseline-corrected neuropil-corrected fluorescence because this exactly follows the paper; do not substitute inferred spikes."

## 2-b. How is the `neural` data processed?

i. Three steps, in order: (1) neuropil subtraction `Fc = F - neucoeff*Fneu` using the coefficient stored in the session's `ops` (0.7 in all sessions); (2) Suite2p `maximin` baseline correction — Gaussian smoothing along time with σ = `sig_baseline` (10 frames), then a rolling minimum filter and a rolling maximum filter of width `win_baseline * fs` (60 s × 30 Hz = 1800 frames), with the resulting baseline subtracted (no division by F0); (3) non-overlapping 10-frame averaging to 3 Hz, stored as float32. Importantly, steps (1)–(2) are applied to the **full** recording and only afterwards sliced to the retained complete-trial prefix, so the 60-s baseline filter never sees an artificial endpoint. Suite2p's `dcnv.preprocess` is not imported; the AI re-implements it with `scipy.ndimage.gaussian_filter`, `minimum_filter1d`, `maximum_filter1d`.

ii.
```python
def suite2p_preprocess(F: np.ndarray, ops: dict) -> np.ndarray:
    """Reproduce suite2p.extraction.dcnv.preprocess using session ops."""
    x = np.asarray(F, dtype=np.float32).copy()
    baseline = ops.get('baseline', 'maximin')
    sig = float(ops.get('sig_baseline', 10.0))
    fs = float(ops['fs'])
    win = max(1, int(float(ops.get('win_baseline', 60.0)) * fs))
    prct = float(ops.get('prctile_baseline', 8.0))
    if baseline == 'maximin':
        flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
        flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
        flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
    ...
    x -= flow
    return x
```
```python
    neucoeff = float(ops.get('neucoeff', 0.7))
    # Match Suite2p/reference ordering: estimate the baseline on the complete
    # neural recording, then restrict to synchronized complete trials.
    raw_corr_full = (np.asarray(F[cell_mask, :], dtype=np.float32)
                     - neucoeff * np.asarray(Fneu[cell_mask, :], dtype=np.float32))
    corrected_full = suite2p_preprocess(raw_corr_full, ops)
    raw_corr = raw_corr_full[:, :keep_native]
    corrected = corrected_full[:, :keep_native]
    neural_binned = average_blocks_2d(corrected)
```
```python
def average_blocks_2d(x, block=AVERAGE_FRAMES):
    if x.ndim != 2 or x.shape[1] % block:
        raise ValueError(...)
    return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 4: "dF/F terminology — Installed Suite2p `dcnv.preprocess` subtracts its estimated baseline and returns baseline-corrected fluorescence (it does not divide by baseline) … Match the explicit Suite2p algorithm/ops rather than adding an undocumented division." Step 6: "Local `suite2p_preprocess` avoids the expensive Suite2p import during conversion while reproducing `suite2p.extraction.dcnv.preprocess`; checked on a real-data slice with `np.allclose(rtol=1e-6, atol=1e-6)`." The full-recording-then-slice ordering was introduced as an explicit fix in Step 10 Iteration 1: "Initial code baseline-corrected only the retained complete-trial prefix. In nine short behavior sessions, this made the filter endpoint differ from reference full-recording Suite2p preprocessing. Fixed by baseline-correcting the complete neural recording first and then slicing the synchronized prefix." The 10-frame averaging is justified by the paper's decoding methods (Step 3: "Reference analyses mildly denoised both dF/F and behavior by averaging non-overlapping bins of 10 consecutive timestamps, yielding 3 Hz samples").

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered by the Suite2p cell classifier probability: `cell_mask = iscell[:, 1] > 0.5`. In practice this is a no-op — the AI verified that all 20,445 ROI entries across all 41 sessions pass, because the distributed Suite2p directories have already been Track2p-curated and reindexed to the longitudinally tracked identities. No additional fluorescence-quality, SNR, or activity-rate exclusion is applied. A mismatch between `iscell` length and `F` neuron count raises, and any filtered ROI count is printed.

ii.
```python
    cell_mask = iscell[:, 1] > 0.5
    if len(cell_mask) != F.shape[0]:
        raise ValueError(f'{sid}: iscell/F neuron mismatch')
    # Distributed data are already curated; still enforce the paper criterion.
    if not np.all(cell_mask):
        print(f'  {sid}: filtering {np.sum(~cell_mask)} ROIs below iscell threshold', flush=True)
```

iii. CONVERSION_NOTES Step 1 found that Track2p's `load_stat_ds_plane` "retain[s] ROIs passing the configured `iscell.npy` probability threshold", with `track_ops.iscell_thr` defaulting to 0.50. Step 3 records the paper's rule: "Suite2p ROIs with cell-classification probability above the default threshold 0.5 were considered cells before Track2p." Step 4 resolves: "Data are already Track2p-curated/reindexed. Verify threshold but do not remove any additional cells." Step 5 Key Decision 2: "**No additional neuron filtering**: All distributed ROIs already pass `iscell > 0.5` and are Track2p identities retained across all days." Step 2 documents the verification: "Across all 20,445 repeated entries, the boolean decision agrees exactly with `iscell[:,1] > 0.5`."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to; the recordings are continuous. The AI aligns each trial to the start of its own consecutive non-overlapping 60-s window, with the window grid anchored at the first imaging frame of the session. It therefore records `temporal_alignment_event = 'Start of each consecutive non-overlapping 60-second window in a continuous session'`, `off_start = 0.0`, `off_end = 60.0`. Trial *t* covers native frames `[t*1800, (t+1)*1800)` for all three streams simultaneously (neural, elapsed time, motion class), so alignment across streams is by shared frame ordinal.

ii.
```python
        'metadata': {
            'task_description': 'Decode within-session motion-energy quintile from baseline-corrected calcium fluorescence.',
            'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,
            'temporal_alignment_event': 'Start of each consecutive non-overlapping 60-second window in a continuous session',
            'off_start': 0.0,
            'off_end': 60.0,
            ...
            'alignment_note': 'Imaging-triggered video aligned by frame ordinal; incomplete endpoint windows discarded without interpolation.',
```
```python
    neural_trials = [... for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
    input_trials  = [... for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
    output_trials = [... for x in labels.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "`time_bin_size=333.3333333333333` ms; alignment is the start of each consecutive 60-s window in a continuous session; `off_start=0.0`, `off_end=60.0`." Step 4: the experiment has "Continuous sessions"; the 60-s trial grid exists only because "Required task overrides this". All three streams share one index grid because the acquisition is hardware synchronized (Step 3: "Video was acquired at 30 Hz with microscope acquisition triggering each camera frame, providing direct synchronization between modalities").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native rate is 30 Hz (1/30 s = 33.33 ms). The AI rebins by averaging non-overlapping blocks of 10 consecutive frames, giving 3 Hz and a bin size of 333.333 ms, and 180 bins per 60-s trial. The same 10-frame averaging is applied to the neural matrix, to the motion-energy trace (before discretization), and to the elapsed-time vector (which uses the bin's centre time). `time_bin_size` is written as `1000.0 * 10 / 30.0 = 333.3333…` ms. Averaging is done with an exact reshape (the code raises if the length is not divisible by 10, which is guaranteed because the session is truncated to a multiple of 1800 frames first).

ii.
```python
AVERAGE_FRAMES = 10
N_BINS_PER_TRIAL = NATIVE_FRAMES_PER_TRIAL // AVERAGE_FRAMES   # 180

def average_blocks_2d(x, block=AVERAGE_FRAMES):
    return x.reshape(x.shape[0], -1, block).mean(axis=2, dtype=np.float32)

def average_blocks_1d(x, block=AVERAGE_FRAMES):
    return x.reshape(-1, block).mean(axis=1, dtype=np.float64)
```
```python
    neural_binned = average_blocks_2d(corrected)
    motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
    motion_binned = average_blocks_1d(motion_native)
    ...
        'time_bin_size': 1000.0 * AVERAGE_FRAMES / 30.0,
```

iii. CONVERSION_NOTES Step 3: "Reference analyses mildly denoised both dF/F and behavior by averaging non-overlapping bins of 10 consecutive timestamps, yielding 3 Hz samples." Step 4: "Apply non-overlapping 10-frame means jointly to neural and motion streams, yielding 3 Hz and 180 bins per 60-s trial." Step 5 Key Decision 3: "**Temporal binning**: Average 10 consecutive synchronized samples for both neural and behavior, matching reference decoding and reducing 30 Hz to 3 Hz." Averaging precedes discretization so that class labels are never averaged.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Not derived from any stored raw time variable. It is computed analytically from the binned-sample index and the sampling rate `ops['fs']` (verified to be 30 Hz). The AI deliberately does **not** use the stored `tstamps.npy` values, because their increments (~3.36e-5 in stored units) are not seconds; `tstamps` is loaded only to contribute its length to the common-length computation. Time is measured from the start of the session (not reset per trial).

ii.
```python
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
    fs = float(ops['fs'])
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{sid}: expected 30 Hz, found {fs}')
    ...
    elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
                + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
```

iii. CONVERSION_NOTES Step 2: "Stored `tstamps` increments are about 3.36e-5 in their native units and are not directly seconds despite 30 Hz imaging. Their units/relationship to video timing must be resolved from the paper/methods before temporal alignment is finalized." Step 4 resolves it: "Align by synchronized sample ordinal at 30 Hz … Sample-index alignment is more reliable than interpreting the supplied timestamp magnitudes." Step 5 mapping row: "Synchronized sample ordinal and `ops['fs']=30` → `input[session][trial][0,:]`."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. For each 10-frame bin *k* the AI stores the **centre** time of the bin: `t_k = (10k + 4.5) / 30` seconds. The vector is generated once for the whole retained session and then reshaped into trials, so time runs continuously across trial boundaries (trial 0 ends at 59.8167 s, trial 1 starts at 60.15 s — a single 1/3 s step). Values are stored as float32 with shape (1, 180) per trial. First value is 0.15 s; last value in a 30-min session is 1799.8167 s.

ii.
```python
    elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
                + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
    ...
    input_trials = [np.ascontiguousarray(x[None, :], dtype=np.float32)
                    for x in elapsed.reshape(n_trials, N_BINS_PER_TRIAL)]
```

iii. CONVERSION_NOTES Step 5 mapping: "Absolute elapsed session time in seconds at each 10-frame averaged sample: mean native sample index / 30 … Time does not reset at each trial because task asks elapsed time from beginning of session. First value is 0.15 s, the mean time of native samples 0-9." Key Decision 8: "**Elapsed time**: Store absolute session time at the mean timestamp of each averaged bin. It intentionally continues across trial boundaries." Key Decision 9 covers the dtype: "Store neural/input as float32 and output as int64 to limit memory and satisfy PyTorch classification expectations."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: `elapsed` is built on exactly the same binned-sample grid as `neural_binned` (same length `keep_native // 10`), and both are reshaped with the identical `(n_trials, 180)` split, so binned sample *k* of the neural matrix and element *k* of the elapsed-time vector describe the same 333 ms window. The stored time is the centre of the bin, matching the fact that the neural value is the mean over that same window. Per-trial shape assertions confirm both are 180 long.

ii.
```python
    neural_binned = average_blocks_2d(corrected)          # (n_neurons, keep_native//10)
    elapsed = ((np.arange(keep_native // AVERAGE_FRAMES, dtype=np.float64) * AVERAGE_FRAMES
                + (AVERAGE_FRAMES - 1) / 2) / fs).astype(np.float32)
    ...
    for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
        assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)
        assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
```

iii. CONVERSION_NOTES Step 10 Check 3: "The same script independently calculates each selected trial's mean native frame indices divided by 30 Hz and compares to converted elapsed time with `np.allclose()`. Passed. It additionally checks the complete elapsed-time vector in every one of 41 sessions." Step 7 also verified the trial-boundary spacing: "trial 0 ends at 59.8167 s and trial 1 starts at 60.15 s, a one-bin (1/3 s) interval."

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` is derived from `move_deve/motion_energy_glob.npy` — the pre-computed global motion-energy trace (sum of squared pixel differences between consecutive video frames), stored as a uint64 vector at 30 Hz. `move_deve/tstamps.npy` is also loaded, but only its *length* is used (to compute the common stream length); its values are not used. `move_deve/interframe_int.npy` is present in the data and was identified by the AI in Step 2 as exactly `np.diff(tstamps)`, but it is **not** loaded or used by `convert_data.py`.

ii.
```python
    motion = np.load(session_dir / 'move_deve' / 'motion_energy_glob.npy', mmap_mode='r')
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
    ...
    common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
```

iii. CONVERSION_NOTES Step 3: "Motion energy was calculated from consecutive frames: pixel-wise frame difference, square each pixel difference, then sum across pixels to one scalar per time point" — i.e. `motion_energy_glob.npy` is precisely the paper's behavioural variable and needs no recomputation from video. Step 4: "Raw uint64 global motion energy with session-specific scale … Average raw motion energy in 10-frame bins, then discretize into five equal-frequency bins separately per session as required by this task."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps: (1) the raw uint64 trace is truncated to `keep_native` (the largest multiple of 1800 frames that all streams cover) and cast to float64; (2) non-overlapping 10-frame averaging to 3 Hz, jointly with the neural data; (3) per-session quintile edges are computed on the binned trace with `np.quantile` at [0.2, 0.4, 0.6, 0.8], with a guard that all four edges are distinct; (4) labels 0–4 assigned by `np.searchsorted(edges, x, side='right')` and stored as int64. No smoothing, normalization, log transform, or z-scoring is applied to the raw magnitude, and no interpolation of missing video frames is performed.

ii.
```python
    motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
    motion_binned = average_blocks_1d(motion_native)

    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    if np.unique(edges).size != 4:
        raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping row: "Truncate to common observed length; 10-frame mean; compute session-specific 20/40/60/80th percentile edges over retained complete-trial samples; `searchsorted(..., side='right')`; split to 1 × 180 int64." Step 4: "Motion energy is nonnegative uint64, finite in inspected sessions, with substantial session-specific offset/scale. The required equal-percentile discretization per session naturally accommodates this scaling" — i.e. per-session percentiles remove the need for any amplitude normalization. Key Decision 4: "Do not interpolate or extrapolate missing behavior."

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five equal-percentile (quintile) bins whose edges are computed **within each session**, on the 10-frame-averaged trace restricted to the retained complete trials. Edges are the 20th/40th/60th/80th percentiles (`np.quantile` with linear interpolation). Assignment is `np.searchsorted(edges, value, side='right')`, so a value exactly equal to an edge falls into the lower class (equivalent to `np.digitize(x, edges)` with `right=False`). The code hard-fails if the four edges are not distinct, and asserts the realized labels span 0–4. Class names are `['lowest motion energy', 'low motion energy', 'middle motion energy', 'high motion energy', 'highest motion energy']` under `output_names = ['motion_energy_quintile']`. The realized global distribution is [0.199995, 0.200005, 0.200, 0.200, 0.200].

ii.
```python
    edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    if np.unique(edges).size != 4:
        raise ValueError(f'{sid}: non-distinct quintile edges {edges}')
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
    ...
    assert labels.min() == 0 and labels.max() == 4
```
```python
        'output_names': ['motion_energy_quintile'],
        'output_values': [[
            'lowest motion energy', 'low motion energy', 'middle motion energy',
            'high motion energy', 'highest motion energy'
        ]],
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "**Percentile scope**: Determine quintile edges separately for each session, after 10-frame averaging and complete-trial restriction, exactly as 'selected per session' requires. Use the full retained session so class semantics are consistent across that session's trials." Key Decision 7: "**Quantile ties**: Assign equal values to the same class using right-sided thresholding. This preserves categorical meaning; all edges are distinct and observed class imbalance is negligible." Discretization after binning is required because "averaging class labels would be meaningless" (the same argument the reference gives). Step 9 confirms the realized distribution is essentially exactly 20% per class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is purely by frame ordinal: the video camera is triggered by the microscope at 30 Hz, so the AI assumes motion-energy sample *i* corresponds to imaging frame *i*. Both streams are truncated to `keep_native = floor(min(len(F), len(Fneu), len(motion), len(tstamps)) / 1800) * 1800` frames, 10-frame averaged on the same grid, and reshaped with the same `(n_trials, 180)` split. In 9 of 41 sessions the motion array is shorter than the neural array (by 1, 1, 1, 2, 2, 2, 3, 116, 148 frames); the AI treats this deficit as **missing samples at the end of the recording** and simply discards the trailing incomplete window. No dropped-frame detection or interpolation is performed, and `interframe_int.npy` is not consulted.

ii.
```python
    common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)
    keep_native = (common // NATIVE_FRAMES_PER_TRIAL) * NATIVE_FRAMES_PER_TRIAL
    ...
    dropped_motion = motion.size - keep_native
    ...
    motion_native = np.asarray(motion[:keep_native], dtype=np.float64)
    motion_binned = average_blocks_1d(motion_native)
```
```python
        'alignment_note': 'Imaging-triggered video aligned by frame ordinal; incomplete endpoint windows discarded without interpolation.',
```

iii. CONVERSION_NOTES Step 4: "Temporal synchronization — … Neural/motion lengths usually match; 8 behavior streams are short by 1-148 endpoint samples … Camera is microscope-triggered; both modalities at 30 Hz … Align by synchronized sample ordinal at 30 Hz. Truncate each session to the shorter observed stream before 10-frame averaging; never extrapolate missing endpoint behavior." Step 5 Key Decision 4: "**Alignment**: Use shared frame ordinal at documented 30 Hz because acquisition was hardware synchronized and raw timestamp units are not reliable seconds. Do not interpolate or extrapolate missing behavior." Step 12 adds an empirical check: a lag cross-correlation between mean population activity and motion energy peaks at +1 bin (0.0843 vs 0.0699 at zero lag), which the AI reads as "a plausible calcium-response lag. There is no large multi-bin displacement indicating stream misalignment."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is defensive but conservative: hard errors for structural problems (`fs != 30`, `F`/`Fneu` shape mismatch, `iscell`/`F` neuron-count mismatch, fewer than two complete trials, non-distinct quintile edges), and silent truncation for length mismatches between neural and behavioural streams. Missing motion-energy samples are assumed to be trailing and are handled by dropping the affected incomplete 60-s window; nothing is interpolated or extrapolated. Per-session diagnostics record `native_neural_frames`, `native_motion_frames`, `retained_native_frames`, `discarded_neural_endpoint_frames` and `discarded_motion_endpoint_frames` into `metadata['session_info']`. Every produced trial is asserted to have the correct shape and to be finite, and labels are asserted to span 0–4. A separate, real bug found during Step 10 (baseline filtering applied to a truncated prefix, distorting the last trial of the nine short sessions) was fixed by filtering the full recording first and then slicing.

ii.
```python
    if not np.isclose(fs, 30.0):
        raise ValueError(f'{sid}: expected 30 Hz, found {fs}')
    if F.shape != Fneu.shape or F.ndim != 2:
        raise ValueError(f'{sid}: incompatible F/Fneu shapes {F.shape}, {Fneu.shape}')
    if len(cell_mask) != F.shape[0]:
        raise ValueError(f'{sid}: iscell/F neuron mismatch')
    if keep_native < 2 * NATIVE_FRAMES_PER_TRIAL:
        raise ValueError(f'{sid}: fewer than two complete 60-s trials')
```
```python
    info = {
        'session_id': sid, 'subject': subject,
        'native_neural_frames': int(F.shape[1]),
        'native_motion_frames': int(motion.size),
        'retained_native_frames': int(keep_native),
        'discarded_neural_endpoint_frames': int(dropped_neural),
        'discarded_motion_endpoint_frames': int(dropped_motion),
        ...
    }
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Drop incomplete final 60-s windows. This excludes nine trials whose behavior lacks 1-148 samples and prevents fabricated labels." Step 9: "All nine sessions with incomplete final behavior windows retain 19 or 29 complete trials and report the discarded endpoint frame counts; no source data needed by a retained trial were lost." Step 10 documents the one iteration: "Iteration 1 — baseline boundary ordering … Fixed by baseline-correcting the complete neural recording first and then slicing the synchronized prefix", followed by re-running the full conversion, verification, and all independent raw-data `np.allclose()` checks.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies the Suite2p maximin baseline correction (Gaussian + 1800-sample rolling min/max filters over every neuron of the full recording) as the dominant cost, together with materializing the full float32 `F`/`Fneu` arrays once per session. Measured throughput is ~0.3–1.3 s per session, 37.4 s for the full 41-session conversion (with pickle serialization of the 393 MiB output a noticeable share of the tail).

ii.
```python
def suite2p_preprocess(F, ops):
    ...
        flow = gaussian_filter(x, sigma=(0.0, sig), mode='reflect')
        flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')   # win = 1800
        flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
```
```python
    start = time.perf_counter()
    ...
    elapsed_s = time.perf_counter() - start
    print(f'  {sid}: {region_idx.size} neurons, {n_trials} trials, '
          f'discarded endpoint frames neural={dropped_neural}, motion={dropped_motion}, '
          f'{elapsed_s:.2f}s', flush=True)
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Full raw fluorescence must be materialized once per session for baseline filtering; processing all sessions simultaneously would use excessive memory. Importing the full Suite2p package is slow and unnecessary after its small baseline algorithm has been verified." Step 7 run-time table: "Full 41-session conversion — approximately 1 s/session plus serialization — approximately 40-60 s; safely below 15 minutes", and Step 9 records the actual 37.39 s. The AI avoided the `suite2p` import (and its torch/GPU path) specifically because the import dominated the cost of a ~1 s/session job.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI states that the conversion is already vectorized and identifies no remaining loop as a bottleneck. The only Python-level loops that remain are: (a) the per-session loop in `convert()` (inherently serial, and deliberately so for memory); (b) the three list comprehensions that split the binned arrays into per-trial views and force contiguity/dtype (`np.ascontiguousarray` per trial — this copies each trial and is the main avoidable per-trial work, though it is required by the list-of-arrays target format); (c) the per-trial assertion loop, which re-runs `cell_mask.sum()`, three shape comparisons and three `np.isfinite(...).all()` scans on every trial — these could be done once per session on the whole array instead. Averaging, neuropil subtraction, filtering, quantiles, and label assignment are all already array operations.

ii.
```python
    neural_trials = [np.ascontiguousarray(x, dtype=np.float32)
                     for x in neural_binned.reshape(neural_binned.shape[0], n_trials, N_BINS_PER_TRIAL).transpose(1, 0, 2)]
```
```python
    for ni, ii, oo in zip(neural_trials, input_trials, output_trials):
        assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)
        assert ii.shape == oo.shape == (1, N_BINS_PER_TRIAL)
        assert np.isfinite(ni).all() and np.isfinite(ii).all() and np.isfinite(oo).all()
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Use vectorized neuropil subtraction, SciPy filters, block reshaping/means, percentile thresholding, and trial reshaping." The AI's position is that no meaningful loop remains; the per-session loop is retained on purpose ("One-session-at-a-time processing — Bounded peak memory"). The AI does not discuss the per-trial assertion loop as an inefficiency.

## 6-c. What processing does the code repeat multiple times?

i. Small, cheap repetitions only. `int(cell_mask.sum())` is recomputed inside the per-trial assertion loop and again for `region_idx` and `info['n_neurons']`. `np.isfinite` is evaluated separately per trial rather than once per session array. `np.bincount(labels, minlength=5)` is computed twice when plotting is enabled (once inside `plot_processing`, once for `info['class_counts']`). Nothing substantive (loading, neuropil correction, baseline filtering, binning, quantiles) is recomputed — each session is read and preprocessed exactly once, and the AI explicitly avoided a two-pass design by computing quintile edges within the same pass.

ii.
```python
        assert ni.shape == (int(cell_mask.sum()), N_BINS_PER_TRIAL)   # recomputed per trial
    ...
    region_idx = np.zeros(int(cell_mask.sum()), dtype=np.int64)
    ...
        'n_neurons': int(cell_mask.sum()),
```
```python
    counts = np.bincount(labels, minlength=5)          # inside plot_processing
    ...
        'class_counts': np.bincount(labels, minlength=5).tolist(),
```

iii. CONVERSION_NOTES Step 6 claims "Memory-map source arrays and process one session at a time" and "Avoid unnecessary file I/O"; no repeated processing is flagged in the notes. The repetitions above are O(n_neurons) or O(n_bins) scalar reductions and are negligible next to the ~1 s/session baseline filtering.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items: (1) `tstamps.npy` is memory-mapped every session but only its `.size` is used — the array values are never read; (2) `raw_corr_full` / `raw_corr` (the neuropil-corrected but *not* baseline-corrected trace) is sliced and kept even when `--show-processing` is off, where it is used only by the plotting function; (3) `suite2p_preprocess` reads and branches on `prctile_baseline` (`prct`) which is never used on the `maximin` path, and implements `constant`/`constant_prctile` branches that this dataset never takes; (4) the baseline filter is deliberately run over the *entire* recording including the frames that are later discarded (up to 148 frames) — extra work that is intentional, since it is what makes the retained trials match the reference filtering; (5) `motion_native` at 30 Hz is retained after binning for the plot only; (6) `metadata['session_info']` stores per-session diagnostics the decoder never reads. None of these is costly.

ii.
```python
    tstamps = np.load(session_dir / 'move_deve' / 'tstamps.npy', mmap_mode='r')
    ...
    common = min(F.shape[1], Fneu.shape[1], motion.size, tstamps.size)   # only .size used
```
```python
    prct = float(ops.get('prctile_baseline', 8.0))   # unused on the maximin path
    ...
    raw_corr = raw_corr_full[:, :keep_native]        # only consumed by plot_processing
    ...
    if show_processing:
        plot_processing(sid, raw_corr, corrected, neural_binned, motion_native,
                        motion_binned, labels, edges)
```

iii. The notes do not flag any of this as waste. The intentional item (4) is justified in Step 10: baseline-correcting the complete recording first is required so that "the filter endpoint" matches "reference full-recording Suite2p preprocessing". The diagnostic metadata is justified in Step 9 as supporting the endpoint audit ("All nine sessions … report the discarded endpoint frame counts"). Step 6 notes that the plotting path is restricted to the first two sessions ("Plot only the requested first two sessions"), but the cheap `raw_corr` slice is still taken unconditionally.
