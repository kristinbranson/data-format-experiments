# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads suite2p outputs (`F.npy`, `Fneu.npy`) from `suite2p/plane0/` within each session directory, and motion energy from `motion_energy_glob.npy` plus `interframe_int.npy` from `move_deve/`. Unlike the reference, the AI does **not** load `iscell.npy` or `ops.npy`; instead, parameters like `fs=30`, `neucoeff=0.7`, and baseline settings are hardcoded as module-level constants.

ii.
```python
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
# ...
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. The AI identified the standard suite2p output structure and loaded the core fluorescence and motion energy files. It hardcoded processing parameters after verifying them from `ops.npy` during exploration (documented in CONVERSION_NOTES Step 2). However, it does not load `iscell.npy` or `ops.npy` at runtime, missing the opportunity to verify assumptions per-session.

## 1-b. How are the data split into subjects?

i. Subjects are identified as directories whose names start with `jm` in the data directory, sorted alphabetically.

ii.
```python
def get_subjects(base_path):
    return sorted(
        d.name for d in os.scandir(base_path)
        if d.is_dir() and d.name.startswith('jm')
    )
```

iii. The `jm*` prefix convention is consistent across the dataset (6 mice). Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Sessions are all subdirectories within each subject's folder, sorted alphabetically.

ii.
```python
def get_sessions(base_path, subject):
    subject_dir = os.path.join(base_path, subject)
    sessions = [d.path for d in os.scandir(subject_dir) if d.is_dir()]
    sessions.sort()
    return sessions
```

iii. Each subdirectory corresponds to one daily recording session. Sorting by name (date-based naming) ensures chronological order.

## 1-d. How are the data split into trials?

i. There is no native trial structure. Trials are defined as 60-second non-overlapping segments of the continuous recording. After 10-frame temporal binning (30 Hz to 3 Hz), each trial contains 180 time bins. Any remainder bins that don't fill a complete trial are discarded.

ii.
```python
trial_frames = TRIAL_DUR * FS // BIN_FRAMES  # 60 * 30 // 10 = 180 bins
n_trials = n_frames // trial_frames
# ...
for ti in range(n_trials):
    s = ti * trial_frames
    e = s + trial_frames
```

iii. The instructions specify "Split sessions into 60-second trials." Since the recording is continuous with no stimulus-driven trial structure, fixed-length non-overlapping segmentation is the appropriate approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All complete 60-second segments are retained. Only incomplete final segments (remainders) are discarded.

ii. N/A (no filtering code)

iii. There is no basis for trial filtering in a continuous spontaneous behavior paradigm. The reference paper's cross-validation uses consecutive 2-minute blocks, but no quality-based trial rejection is described.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw ROI fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`.

ii.
```python
F = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'F.npy'))
Fneu = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'Fneu.npy'))
```

iii. These are the standard suite2p output files for raw and neuropil fluorescence traces, consistent with the paper's description of using Suite2p processing.

## 2-b. How is the `neural` data processed?

i. Neuropil subtraction is applied (`Fc = F - 0.7 * Fneu`), followed by suite2p's `dcnv.preprocess` function which performs baseline estimation using the `maximin` method (Gaussian smoothing, rolling minimum, rolling maximum) and subtracts the baseline. The result is NOT divided by the baseline. Then 10-frame non-overlapping temporal binning is applied.

ii.
```python
Fc = F - NEUCOEFF * Fneu  # NEUCOEFF = 0.7
Fc = dcnv.preprocess(
    F=Fc,
    baseline='maximin',
    win_baseline=60.0,
    sig_baseline=10,
    fs=FS,
    prctile_baseline=8.0,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)
# ...
Fc = bin_frames(Fc)  # 10-frame averaging
```

iii. The paper states "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)." The AI uses suite2p's own `dcnv.preprocess` function directly, which implements the same maximin baseline correction. The reference reimplements this manually using scipy filters. Both approaches should produce equivalent results since the underlying algorithm is the same.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron quality filtering is applied. All neurons present in the `F.npy` files are retained. The code does not load or check `iscell.npy`.

ii. N/A (no filtering code in convert_data.py)

iii. The AI's CONVERSION_NOTES document that all supplied `iscell` flags are 1 (already filtered) and neuron counts are constant per mouse across days (Track2p-matched). The decision to retain all neurons is correct, but the code does not verify this assumption at runtime. The reference loads `iscell.npy` and asserts all flags are 1 as a sanity check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to session start. Since trials are contiguous 60-second segments of the continuous recording, there is no event-based alignment. The temporal alignment event is described as "session_start" in the code's metadata (though the actual data pickle says "start of each non-overlapping 60-second trial").

ii.
```python
'metadata': {
    'temporal_alignment_event': 'session_start',
    'off_start': None,
    'off_end': None,
}
```

iii. There is no stimulus event to align to in this spontaneous behavior paradigm. The recording is continuous, and trials are artificial segments starting from the beginning of the session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, converting from 30 Hz to ~3 Hz (333.33 ms time bins). Binning is applied before motion energy discretization.

ii.
```python
BIN_FRAMES = 10
# ...
Fc = bin_frames(Fc)
me = bin_frames(me)
# ...
'metadata': {
    'time_bin_size': BIN_FRAMES / FS * 1000,  # 333.33 ms
}
```

iii. The paper methods state "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." Both streams are binned together before discretization because averaging class labels would be meaningless.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed synthetically from the time bin index within the session, the bin size (10 frames), and the sampling rate (30 Hz).

ii.
```python
t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS).astype(np.float32)
inp_trials.append(t[np.newaxis, :])  # (1, trial_frames)
```

iii. Since the frame rate is constant at 30 Hz and there are no stored timestamps, computing time from bin indices is equivalent to using actual timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code on disk computes the **left edge** of each temporal bin: `bin_index * 10 / 30` seconds from session start. This gives time values starting at 0.0 for the first bin. However, the AI's CONVERSION_NOTES and the actual converted data use **bin centers** (`(bin_index * 10 + 4.5) / 30`, starting at 0.15s), indicating the code was modified after the final conversion was run.

ii. Code on disk (left edge):
```python
t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS).astype(np.float32)
# First bin: 0 * 10 / 30 = 0.0
```

Actual data in pickle (bin center):
```
First 5 values: [0.15, 0.4833, 0.8167, 1.15, 1.4833]
```

iii. The AI's CONVERSION_NOTES Step 5 documents "Elapsed session time at centers of each 10-frame average: (10*k+4.5)/fs", and Step 7 confirms "Trial-1 bin centers are 0.15 to 59.8167 s." The data matches the bin-center approach, but the code on disk does not.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time is computed from the same bin indices used for neural and output data, so alignment is inherent. Time runs continuously across trials within a session (not reset per trial).

ii.
```python
# s is the starting bin index within the session
t = ((s + np.arange(trial_frames)) * BIN_FRAMES / FS).astype(np.float32)
neural_trials.append(Fc[:, s:e])
```

iii. All three data streams (neural, input, output) use the same bin indexing, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. The code on disk also loads `interframe_int.npy` for dropped-frame interpolation, though the actual converted data uses shared-prefix trimming instead.

ii.
```python
me = np.load(os.path.join(session_path, 'move_deve', 'motion_energy_glob.npy'))
dt = np.load(os.path.join(session_path, 'move_deve', 'interframe_int.npy'))
```

iii. The motion energy file contains the pre-computed global motion energy signal (sum of squared pixel differences between consecutive video frames).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The code on disk implements: (1) dropped-frame interpolation using interframe intervals, (2) 10-frame temporal binning, (3) per-session percentile-based discretization into 5 bins. However, the actual converted data uses shared-prefix trimming (not interpolation) and computes quantiles only on used frames.

ii. Code on disk (interpolation approach):
```python
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    for offset, idx in enumerate(drop_indices):
        insert_pos = idx + 1 + offset
        interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
        me = np.insert(me, insert_pos, interp_val)
# ...
me = bin_frames(me)
# ...
percentiles = np.linspace(0, 100, n_levels + 1)
bin_edges = np.percentile(me, percentiles)
output = np.digitize(me, bin_edges[1:-1])
```

Actual converted data metadata shows shared-prefix approach:
```
'trial_policy': 'non-overlapping complete 60-s trials from shared neural/behavior prefix'
'shared_frames': 35884, 'used_frames': 34200  # for a deficit session
```

iii. The CONVERSION_NOTES Step 5 planned shared-prefix trimming: "Handle missing behavior by shared-prefix trimming: Motion deficits occur only at the end. No interpolation, extrapolation, or padding is justified." But the code on disk implements interpolation. The actual data matches the documented plan, not the code.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into 5 equal-percentile bins (quintiles) per session. The code uses `np.percentile` to compute the 0th, 20th, 40th, 60th, 80th, and 100th percentiles, then `np.digitize` to assign labels 0-4. However, the actual data uses `np.quantile` with [0.2, 0.4, 0.6, 0.8] and `np.searchsorted(side='right')`.

ii. Code on disk:
```python
percentiles = np.linspace(0, 100, n_levels + 1)  # [0, 20, 40, 60, 80, 100]
bin_edges = np.percentile(me, percentiles)
output = np.digitize(me, bin_edges[1:-1])  # levels 0..4
```

iii. Both `np.digitize` and `np.searchsorted(side='right')` produce equivalent binning for the same quantile edges. The core decision (5 equal-percentile bins per session) matches the instructions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The code on disk attempts to align motion energy by interpolating dropped camera frames (detected via interframe intervals >0.04s after unit scaling) so that the motion array matches the full neural array length. The actual converted data instead uses shared-prefix trimming: `shared_frames = min(neural_frames, motion_frames)`, using only the overlapping prefix.

ii. Code on disk (interpolation):
```python
if me.shape[0] < expected_len:
    drop_indices = np.where(dt * 1000 > 0.04)[0]
    # ... insert interpolated values ...
assert me.shape[0] == expected_len
```

Actual data metadata (shared prefix):
```
'shared_frames': 35884  # = min(36000, 35884)
'used_frames': 34200    # = 19 * 1800
```

iii. The reference approach (shared-prefix trimming) is simpler and avoids fabricating data. The AI's CONVERSION_NOTES plan explicitly chose shared-prefix trimming and documented "Never interpolate or fabricate missing terminal behavior." The data follows this plan; the code on disk does not.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The converted data handles missing motion frames via shared-prefix trimming (using only frames where both neural and motion data exist). Incomplete final 60-second trial segments are discarded. The code on disk instead attempts dropped-frame interpolation.

ii. Data metadata for a deficit session:
```
'neural_source_frames': 36000, 'motion_source_frames': 35884,
'shared_frames': 35884, 'used_frames': 34200,
'unused_shared_terminal_frames': 1684,
'motion_frames_missing_vs_neural': 116
```

iii. The CONVERSION_NOTES document that 9 of 41 sessions have motion arrays shorter than neural by 1-148 frames, with deficits occurring at the session end. Shared-prefix trimming is the conservative approach, discarding at most one additional trial per deficit session.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the baseline correction (either via `dcnv.preprocess` or manual scipy filtering), which involves sliding-window operations across the full session length for every neuron. The full 41-session conversion completes in ~36 seconds (~0.86s/session).

ii. N/A

iii. GPU acceleration via CUDA (when available) mitigates this. The reference uses scipy CPU filters. Both are sufficiently fast.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. In the code on disk, the dropped-frame interpolation loop uses `np.insert` in a for loop, reallocating the array each iteration. This could be vectorized by pre-allocating the output array. However, the number of dropped frames is typically very small (1-3 for most deficit sessions), so the impact is negligible.

ii.
```python
for offset, idx in enumerate(drop_indices):
    insert_pos = idx + 1 + offset
    interp_val = (me[insert_pos - 1] + me[insert_pos]) / 2.0
    me = np.insert(me, insert_pos, interp_val)
```

iii. The actual converted data uses shared-prefix trimming (no loop needed), making this a moot point for the actual output.

## 6-c. What processing does the code repeat multiple times?

i. No significant repeated processing is identified. Sessions are processed sequentially with each step applied once.

ii. N/A

iii. N/A

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code applies baseline correction to the full neural session before selecting only the shared-prefix used frames. Baseline-corrected frames beyond the shared prefix are computed but discarded. This is actually intentional and correct: baseline estimation should use the full recording context.

ii.
```python
# Baseline uses full session
neural_bc, Fc, Flow = baseline_correct(F, Fneu, ops)
# Only used_frames are binned
neural_binned = mean_bins_2d(neural_bc, used_frames)
```

iii. Using the full session for baseline estimation is the correct approach, as truncating the timeseries could create edge artifacts in the baseline estimate. The "wasted" computation on unused frames is a necessary side effect.
