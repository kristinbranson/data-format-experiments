# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers the dataset purely from the directory tree rooted at `/app/data`. `discover_sessions()` treats every subdirectory of `/app/data` as a subject (sorted lexically) and every subdirectory of a subject as a session (sorted lexically, which is chronological because sessions are named `YYYY-MM-DD_a`). For each candidate session it *requires* the three payload files (`suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `move_deve/motion_energy_glob.npy`) and raises `FileNotFoundError` otherwise, so no session can be silently skipped. Sessions are stored as a flat, ordered list of `SessionRef(subject, session_id, path)` records and converted one at a time.

Loading happens in three places:
- `validate_source_session()` memory-maps `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy` and fully loads `stat.npy` and `ops.npy` to check row/frame invariants (all streams same shape, `iscell` all pass Track2p curation, `ops['fs']==30`, `ops['nframes']==n_frames`, session length an exact multiple of 60 s).
- `baseline_correct_and_bin()` re-opens `F.npy`/`Fneu.npy` with `mmap_mode="r"` and streams them in 64-neuron chunks.
- `align_motion()` loads `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` for the behavioral stream.

Trials are not loaded — they do not exist natively. The whole continuous session is loaded and processed, and only afterwards sliced into 60-s trials. The result is 6 subjects, 41 sessions, 1090 trials, 20,445 session-neurons.

ii.
```python
def discover_sessions() -> tuple[list[str], list[SessionRef]]:
    """Return subjects and sessions in deterministic chronological order."""
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    sessions: list[SessionRef] = []
    for subject in subjects:
        subject_dir = DATA_ROOT / subject
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            required = (
                session_dir / "suite2p" / "plane0" / "F.npy",
                session_dir / "suite2p" / "plane0" / "Fneu.npy",
                session_dir / "move_deve" / "motion_energy_glob.npy",
            )
            if not all(p.exists() for p in required):
                raise FileNotFoundError(f"Incomplete session directory: {session_dir}")
            sessions.append(SessionRef(subject, session_dir.name, session_dir))
    return subjects, sessions
```

```python
f = np.load(plane / "F.npy", mmap_mode="r")
fneu = np.load(plane / "Fneu.npy", mmap_mode="r")
spks = np.load(plane / "spks.npy", mmap_mode="r")
iscell = np.load(plane / "iscell.npy", mmap_mode="r")
stat = np.load(plane / "stat.npy", allow_pickle=True)
ops = np.load(plane / "ops.npy", allow_pickle=True).item()
...
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
```

And the hard-coded whole-dataset inventory assertion:
```python
if len(neural) != 41 or total_trials != 1090 or total_timepoints != 196200:
    raise AssertionError(...)
if sum(x[0].shape[0] for x in neural) != expected_neurons:   # 20445
    raise AssertionError("Session-neuron total mismatch")
```

iii. From CONVERSION_NOTES Step 2/Step 5: the release is organised as "six subject folders … Each subject contains chronologically named daily recording sessions. Each session contains one imaging plane in `suite2p/plane0/` and behavior in `move_deve/`." The AI decided to "preserve every valid supplied sample" (Key Decision 1) and to enforce "Chronological deterministic ordering: Subjects lexical, sessions chronological, trials chronological" (Key Decision 8). The whole-dataset assertion was added as a planned sanity check ("Assert expected totals: 6 subjects, 41 sessions, 2,998 unique longitudinal tracks, 20,445 session-neurons, 1,090 trials, and 196,200 decoder timepoints").

## 1-b. How are the data split into subjects?

i. One subject per top-level directory in `/app/data`, sorted lexically, giving `['jm031','jm032','jm038','jm039','jm040','jm046']`. The AI does *not* filter on the `jm` prefix; it accepts any directory (in practice the only non-directory entries are `README.md`, `load_data.ipynb` and `.fetch_complete`, so the result is identical). `subject_idx` for each session is `subjects.index(ref.subject)`, and because sessions are emitted subject-by-subject the index array is block-contiguous.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
...
subject_idx.append(subjects.index(ref.subject))
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Subject folder name → `subjects`, `subject_idx`; Subjects sorted lexically; each chronological session points to its subject index … `['jm031','jm032','jm038','jm039','jm040','jm046']`." The AI cross-checked this against the paper's "full dataset of 6 mice" (Step 3) and confirmed 6 subjects with 7/7/7/7/6/7 sessions.

## 1-c. How are the data split into sessions?

i. One session per subdirectory of a subject folder (e.g. `jm031/2023-10-18_a`), sorted lexically = chronologically. A session is a single continuous daily recording and becomes one entry in `neural`/`input`/`output`. All 41 sessions are kept; none is dropped. Session lengths are heterogeneous by subject: jm031 and jm032 have 36,000 frames (20 min), the other four mice have 54,000 frames (30 min), and the AI deliberately keeps the full 30 min even though the paper's Methods say sessions were 20 min.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
    ...
    sessions.append(SessionRef(subject, session_dir.name, session_dir))
```
```python
if int(ops["fs"]) != NATIVE_FS_HZ or int(ops["nframes"]) != n_frames:
    raise ValueError(f"Unexpected sampling metadata in {ref.path}")
```
Metadata records per-session provenance:
```python
session_info = {
    "session_id": f"{ref.subject}/{ref.session_id}",
    "subject": ref.subject,
    "date": ref.session_id.removesuffix("_a"),
    "n_neurons": n_neurons,
    "native_frames": n_frames,
    "native_duration_seconds": n_frames / NATIVE_FS_HZ,
    ...
}
```

iii. CONVERSION_NOTES Step 4 discrepancy table, "Session duration" row: "Preserve all complete source frames because the deliverable is the full converted dataset and the extra 10 min have aligned neural/behavior recordings and no invalid-period marker. Cropping would discard 18,000 valid samples from 27 sessions. Record per-session durations in metadata." The `reference_discrepancies` metadata field states this openly.

## 1-d. How are the data split into trials?

i. There is no native trial structure (Step 2: "The native recordings are continuous and contain no trial definitions"). The AI defines trials as non-overlapping consecutive 60-second segments of the already-binned session: `TRIAL_BINS = 60 * 3 = 180` bins of 333.33 ms. Crucially, **all filtering/baseline/binning is done on the full continuous session and trials are cut afterwards**, so no filter sees an artificial trial edge. The AI additionally *asserts* that each session is an exact multiple of a trial (`n_frames % (10*180) != 0` raises), rather than discarding a remainder; in this dataset every session divides exactly, producing 20 trials for the 36,000-frame sessions and 30 for the 54,000-frame sessions (1090 total).

ii.
```python
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * OUTPUT_FS_HZ)      # 180
...
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
...
n_trials = n_binned // TRIAL_BINS
for trial in range(n_trials):
    sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
    neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
    input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
    output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
    if neural_trial.shape != (n_neurons, TRIAL_BINS):
        raise AssertionError("Neural trial shape mismatch")
```

iii. Step 5 mapping table: "split columns into 180-bin (60-s) trials … Process the entire continuous session before trial splitting so filtering has no artificial trial-edge discontinuities." Step 2: "Splitting into non-overlapping 60-second trials is therefore a downstream requirement, not native curation." Step 9: "No samples were lost: 36,000/54,000 native frames become exactly 3,600/5,400 paired bins and 20/30 complete trials."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied — all 1090 trials are retained. The AI examined the only candidate exclusion signal, Suite2p's `ops['badframes']`, and found exactly one flagged frame in the entire dataset (`jm032/2023-10-24_a`); it decided to keep it rather than excise it. It records the badframe count per session in metadata so a downstream user can filter if desired. Sessions with interpolated camera frames (up to 148 out of 36,000) are also kept rather than excluded.

ii. There is no filtering code. The only trial-level checks are shape assertions:
```python
if input_trial.shape != (1, TRIAL_BINS) or output_trial.shape != (1, TRIAL_BINS):
    raise AssertionError("Input/output trial shape mismatch")
```
and the badframe bookkeeping:
```python
"suite2p_badframes": int(np.count_nonzero(ops["badframes"])),
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules"): "No experimental trials were defined. Paper decoder splits were based on consecutive 2-minute blocks … the paper provides no further exclusion rule." Step 4, "Neural bad frames" row: "Retain it; Suite2p has already processed it and 10-frame averaging reduces single-frame influence. Removing it would desynchronize behavior."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from exactly two arrays: `suite2p/plane0/F.npy` (raw ROI fluorescence) and `suite2p/plane0/Fneu.npy` (neuropil fluorescence), both `(n_neurons, n_frames)` float32. `ops.npy` supplies the acquisition parameters used to parameterise processing (`fs=30`, `neucoeff=0.7`, maximin baseline, `win_baseline=60 s`, `sig_baseline=10`). `spks.npy`, `iscell.npy` and `stat.npy` are loaded but used only for validation, not for the output values. The AI explicitly rejected `spks` (deconvolved rates) as the signal.

ii.
```python
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
...
corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu     # NEUROPIL_COEFF = 0.7
```

iii. CONVERSION_NOTES Key Decision 2: "**Paper-style fluorescence, not raw F or spks**: Use neuropil-corrected maximin baseline-subtracted fluorescence because this is the neural signal named for paper decoding. `spks` remains an available alternative but was not the reported decoder input." Step 3 notes the paper "calls its neural representation 'baseline corrected fluorescence traces as our dF/F', using default Suite2p parameters."

## 2-b. How is the `neural` data processed?

i. Three stages, all applied to the full continuous session before trial cutting:
1. **Neuropil subtraction**: `Fc = F - 0.7 * Fneu`. The coefficient 0.7 is taken from the packaged `ops`, overriding the track2p GUI's default of `neucoeff=0.0`.
2. **Maximin baseline subtraction**: Gaussian smooth along time with σ = 10 frames, then a 1800-frame (60 s) rolling minimum, then a 1800-frame rolling maximum, then subtract. This is a line-for-line reproduction of `DataManagement.F_processing` in `/app/code/track2p/gui/data_management.py` (same `scipy.ndimage` calls, same parameters), and is numerically equivalent to suite2p's `dcnv.preprocess(baseline='maximin', ...)` — I verified this directly on `jm031/2023-10-18_a`: r = 0.99996 between the AI's traces and `dcnv.preprocess` output (mean |Δ| = 0.06 against a trace SD of 40.7; residual differences come only from suite2p's odd 1801-frame window, `replicate` padding and 3σ Gaussian truncation).
3. **Temporal averaging**: non-overlapping means of 10 consecutive frames (see 2-e).

No ΔF/F division, no z-scoring, no per-neuron normalisation. Neurons are processed in chunks of 64 to bound memory; because the Gaussian/min/max filters act only along the time axis, chunking is exactly equivalent to whole-array processing. Output dtype is float32; a global `np.isfinite` check is applied.

ii.
```python
for start in range(0, n_neurons, NEURON_CHUNK):
    stop = min(start + NEURON_CHUNK, n_neurons)
    raw_f = np.asarray(f[start:stop], dtype=np.float32)
    raw_fneu = np.asarray(fneu[start:stop], dtype=np.float32)
    corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
    baseline = gaussian_filter(corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES))
    baseline = minimum_filter1d(baseline, size=baseline_window, axis=1)
    baseline = maximum_filter1d(baseline, size=baseline_window, axis=1)
    activity = corrected_neuropil - baseline
    result[start:stop] = activity.reshape(stop - start, n_binned, DOWNSAMPLE).mean(axis=2)

if not np.all(np.isfinite(result)):
    raise ValueError(f"Non-finite processed neural values in {plane_dir}")
```
with
```python
NEUROPIL_COEFF = 0.7
BASELINE_SIGMA_FRAMES = 10.0
BASELINE_WINDOW_SECONDS = 60.0
baseline_window = int(BASELINE_WINDOW_SECONDS * NATIVE_FS_HZ)   # 1800
```

iii. CONVERSION_NOTES Step 4, "Fluorescence representation" row: "Use `Fc=F-0.7*Fneu`, Gaussian sigma 10 frames, min then max filters over 1,800 frames, and `Fc-Flow`. This follows paper/default Suite2p settings and the reference baseline algorithm; **do not divide by `Flow` because the provided implementation does not**." Step 1 notes the GUI function "Computes `Fc=F-neucoeff*Fneu` (default `neucoeff=0`) and subtracts a Suite2p-style maximin baseline after Gaussian smoothing; despite GUI label, it does not divide by baseline" — so the AI kept the algorithm but substituted the recorded `neucoeff=0.7`. Step 10 check 2 re-derived the values independently from `F.npy`/`Fneu.npy` and passed `np.allclose(rtol=1e-6, atol=1e-5)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Instead the AI *verifies* that the curation has already been applied upstream: it asserts that every row has Suite2p label `iscell[:,0] == 1` and probability `iscell[:,1] > 0.5`, that `F`/`Fneu`/`spks`/`stat`/`iscell` have consistent row counts, and that the neuron count is constant across days within a subject (which is what being tracked on all days implies). Any violation raises `ValueError`. All 20,445 session-neurons (2,998 unique longitudinal tracks) are kept.

ii.
```python
if f.ndim != 2 or f.shape != fneu.shape or f.shape != spks.shape:
    raise ValueError(f"Neural stream mismatch in {ref.path}: ...")
n_neurons, n_frames = f.shape
if iscell.shape != (n_neurons, 2) or len(stat) != n_neurons:
    raise ValueError(f"ROI row mismatch in {ref.path}")
if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Packaged neurons do not all pass Track2p curation: {ref.path}")
```

iii. CONVERSION_NOTES Key Decision 3: "**No further neuron filtering**: The release is already restricted to >0.5-probability neurons tracked across all days, and all rows pass this criterion. Applying a second quality rule would diverge from the reference and lose data." Step 4, "Cell filtering" row: "Fully consistent; no additional cell filter is appropriate because the release is already curated." Step 2 confirmed empirically: "Every binary label is 1 and every probability is strictly above 0.5 … confirming that the packaged traces have already undergone Track2p's 0.5 cell filter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to — the recording is continuous spontaneous activity. Trials are contiguous, non-overlapping 60-s blocks measured from the start of the session, so each trial is aligned to its own segment onset. The AI encodes this in metadata as `temporal_alignment_event = "start of each non-overlapping 60-second segment"` with `off_start = 0.0` and `off_end = 60.0` (i.e. the trial spans 0 to +60 s relative to the alignment event). No re-slicing, padding or realignment is done; trial *t* is columns `[180t, 180(t+1))` of the continuous binned session for all three streams simultaneously, so `neural`, `input` and `output` are identically indexed by construction.

ii.
```python
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial  = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
input_trial   = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
output_trial  = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
```
```python
"temporal_alignment_event": "start of each non-overlapping 60-second segment",
"off_start": 0.0,
"off_end": 60.0,
"trial_duration_seconds": TRIAL_SECONDS,
```

iii. Step 2: "The native recordings are continuous and contain no trial definitions." Step 5 metadata row: "alignment is each 60-s segment start, `off_start=0`, `off_end=60`." Step 10, edge-case audit: "Every raw frame is consumed exactly once; every trial has 180 bins … adjacent trials differ by exactly 1/3 s without overlap or gap."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is the core temporal transform. Native acquisition is 30 Hz (33.33 ms). The AI averages non-overlapping blocks of `DOWNSAMPLE = 10` consecutive frames, giving 3 Hz, i.e. a **333.333 ms** bin, which is written to `metadata['time_bin_size']` in ms as required. The identical 10-frame mean is applied to the neural traces *and* to the aligned motion-energy trace, and crucially it is applied to motion energy **before** discretization (averaging the continuous signal, never the class labels). The time-elapsed input is likewise defined on the same 3 Hz grid. Every session therefore has exactly 180 bins per 60-s trial and the three streams are guaranteed the same length (checked by assertion). No smoothing beyond this averaging is applied, and there is no overlap between bins.

ii.
```python
NATIVE_FS_HZ = 30
DOWNSAMPLE = 10
OUTPUT_FS_HZ = NATIVE_FS_HZ / DOWNSAMPLE           # 3.0
TIME_BIN_MS = 1000.0 / OUTPUT_FS_HZ                # 333.333...
```
Neural:
```python
result[start:stop] = activity.reshape(stop - start, n_binned, DOWNSAMPLE).mean(axis=2)
```
Motion (before discretization):
```python
def bin_and_discretize_motion(aligned_motion):
    motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
    quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
```
Length consistency check:
```python
if neural_binned.shape != (n_neurons, n_binned):
    raise AssertionError("Processed neural shape mismatch")
if len(motion_binned) != n_binned or len(labels) != n_binned:
    raise AssertionError("Processed behavior length mismatch")
```
```python
"time_bin_size": TIME_BIN_MS,
"temporal_downsampling": "non-overlapping means of 10 consecutive native timestamps",
```

iii. CONVERSION_NOTES Step 3: "For decoding, both fluorescence and behavior were averaged over the same non-overlapping 10 consecutive timestamps (effective 3 Hz)." Step 4, "Temporal denoising" row: "Apply identical non-overlapping 10-frame means to neural and motion streams, yielding a common 3 Hz (333.333 ms) grid." Key Decision 4: "Ten-frame averaging is the paper's decoder denoising and gives a uniform 333.333-ms bin for every session. Exactly 180 bins form each requested 60-s trial."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not read from any stored variable — there is no per-frame time vector for the imaging stream in the release. It is computed analytically from the bin index, the fixed acquisition rate `ops['fs'] = 30 Hz` (validated per session) and the downsample factor. The AI deliberately uses the *mean acquisition time of the ten native frames represented by each bin*, i.e. `(10k + 4.5)/30` s, rather than the bin's left edge. Time is measured from the start of the session (the recording is one continuous block, so session start = start of the experiment for that recording).

ii.
```python
elapsed_time = (
    np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
) / np.float32(NATIVE_FS_HZ)
```
validated by
```python
if int(ops["fs"]) != NATIVE_FS_HZ or int(ops["nframes"]) != n_frames:
    raise ValueError(f"Unexpected sampling metadata in {ref.path}")
```
```python
"input_names": ["time elapsed from session start (s)"],
```

iii. Step 5 mapping table: "Native frame indices and `ops['fs']=30` → `input[0]`; Mean elapsed times of each 10-frame block: `(10*k + 4.5)/30` s, preserved as global session elapsed time." Key Decision 7: "Each value is the mean acquisition time of the ten samples represented by that bin." Step 10 check 3 independently recomputed the analytic frame-time means for the first 20-min and last 30-min sessions and passed `np.allclose(atol=1e-6)` "including all trial transitions."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Essentially none beyond the arithmetic above. Specifically the AI decided **not** to reset the clock at each trial boundary: `elapsed_time` is built once for the whole session and then sliced with the same slice as the neural data, so trial 0 runs 0.15 → 59.82 s, trial 1 runs 60.15 → 119.82 s, and the final trial of a 30-min session ends at 1799.8167 s. Values are stored as float32 with shape `(1, 180)` per trial. No normalisation, centring or scaling is applied. The resulting ranges reported by the validator are `[0.2, 1199.8]` for jm031/jm032 sessions and `[0.2, 1799.8]` for the rest (validator rounds to 1 d.p.).

ii.
```python
input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
```

iii. Key Decision 7: "**Global elapsed-time input**: Time values continue across trial boundaries, preserving the requested session context." Step 5 notes: "Input is not reset at each 60-s boundary because the requested variable is time since session beginning." This directly implements the Decoder Task spec, "Time elapsed from the beginning of the session in seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction — `elapsed_time` is created with length `n_binned`, exactly the number of neural bins, on the same 3 Hz grid, and the same `slice` object is used to cut both. So input bin *j* of trial *t* is the same 333.33 ms window as neural column *j* of trial *t*. Because the AI uses bin *centres* for both conceptually (the neural value is the mean over the same 10 frames whose mean time is reported), the two are aligned to the same instant rather than offset by half a bin. A length assertion guards against drift, and the `--show-processing` plot overlays the binned population activity and binned motion on the shared time axis.

ii.
```python
n_binned = n_frames // DOWNSAMPLE
elapsed_time = (np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)) / np.float32(NATIVE_FS_HZ)
if neural_binned.shape != (n_neurons, n_binned):
    raise AssertionError("Processed neural shape mismatch")
...
sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
input_trial  = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
```

iii. Step 10 edge-case audit: "first/last elapsed times are 0.15/1199.8167 s or 0.15/1799.8167 s; adjacent trials differ by exactly 1/3 s without overlap or gap." Step 5 planned check: "Use `np.allclose` to compare converted elapsed-time inputs to the independent analytical frame-time means for multiple trials, including the final trial" — reported as passing in Step 10.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From `move_deve/motion_energy_glob.npy` (the pre-computed global motion energy of the infrared behaviour video: summed squared pixel differences between consecutive frames, one uint64 scalar per camera frame). Two auxiliary files are loaded to repair the stream: `move_deve/interframe_int.npy` (timestamp differences, length `n_camera_frames - 1`), which is what actually drives the gap reconstruction, and `move_deve/tstamps.npy` (camera timestamps), which is only length-checked and passed to the diagnostic plot. The expected length is taken from the neural frame count.

ii.
```python
raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
if len(raw) != len(timestamps) or len(intervals) != len(raw) - 1:
    raise ValueError(f"Malformed motion arrays in {move_dir}")
if len(raw) > expected_frames:
    raise ValueError(f"Motion has more frames than neural data in {move_dir}")
```

iii. Step 3: "Motion energy is already provided: squared pixel differences between each pair of consecutive video frames, summed over all pixels to a scalar at each time point." Step 2: "`interframe_int.npy`: float64 timestamp differences, length one less than the camera series; enlarged gaps identify dropped camera frames."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps, in order:
1. **Gap repair / alignment to the neural frame grid** (see 4-d): where the camera array is shorter than the imaging array, the true 30 Hz index of every observed camera sample is reconstructed from `round(interval / median_interval)`, and the missing positions are filled by `np.interp`. If the reconstructed indices do not exactly account for the deficit, the script raises rather than guessing. Sessions with no length deficit are left untouched.
2. **10-frame averaging** onto the same 3 Hz grid as the neural data.
3. **Per-session quintile discretization** (see 4-c).
4. Slicing into `(1, 180)` int64 trials.

No smoothing, log transform, clipping or outlier removal is applied to the continuous values. I re-derived the full label sequence for `jm031/2023-10-22_a` (the worst session, 116 missing frames) from the raw files using the *human reference's* algorithm and compared it to `converted_data.pkl`: the label arrays are **exactly equal** (3600/3600).

ii.
```python
if missing:
    observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
    if observed_positions[-1] != expected_frames - 1:
        raise ValueError(f"Timestamp gaps do not explain frame deficit in {move_dir}: ...")
    aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)
    interpolated_mask = np.ones(expected_frames, dtype=bool)
    interpolated_mask[observed_positions] = False
else:
    aligned = raw.copy()
    ...
if len(aligned) != expected_frames or not np.all(np.isfinite(aligned)):
    raise ValueError(f"Failed motion alignment in {move_dir}")
```
```python
motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
```

iii. Step 5 mapping table: "If behavior length is deficient, derive observed integer positions from rounded interframe interval / median interval and linearly interpolate missing values; average each 10 frames; compute session-specific 20/40/60/80% thresholds." Key Decision 5: "Linear interpolation is explicitly permitted by the release README and avoids NaNs rejected by the validator. Integer positions derived from timestamp gaps reproduce the exact known deficits." Step 10 check 4 independently recomputed labels for an intact session, the 116-missing-frame session and the final 1-missing-frame session and passed `np.allclose`.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into five classes using the 20th/40th/60th/80th percentiles of the **binned** motion energy, computed **independently within each session** (not globally, not per trial). Assignment uses `np.searchsorted(edges, x, side="right")`, which is identical to `np.digitize(x, edges)`. The AI then hard-asserts that the five class counts are exactly equal, i.e. that the quintiles really are equal-percentile — this passes for all 41 sessions (39,240 bins per class overall, 0.200 fraction each, confirmed in `verification_full_out.txt`). The per-session `quantile_edges` and `class_counts` are saved into metadata for auditability. Class names are `'0-20% (lowest)' … '80-100% (highest)'`.

Note the ordering: discretization happens **after** 10-frame averaging, so percentiles are taken over exactly the values the decoder sees, and class labels are never averaged.

ii.
```python
def bin_and_discretize_motion(aligned_motion):
    """Average motion in 10-frame bins and assign session-specific quintiles."""
    motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
    quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
    counts = np.bincount(labels, minlength=5)
    expected = len(labels) // 5
    if not np.array_equal(counts, np.full(5, expected)):
        raise ValueError(f"Motion quintiles are not equal: counts={counts.tolist()}")
    return motion_binned, quantile_edges, labels
```
```python
"output_names": ["motion energy percentile bin"],
"output_values": [["0-20% (lowest)", "20-40%", "40-60%", "60-80%", "80-100% (highest)"]],
"motion_discretization": ("20th, 40th, 60th, and 80th percentiles of aligned, "
                          "10-frame-averaged motion energy, independently per session"),
```

iii. Key Decision 6: "**Per-session quantiles after alignment/averaging**: This applies 'selected per session' to the exact continuous values represented at decoder timepoints. Across source data it produces exactly balanced classes." Step 5 notes: "Quantile check across all 41 sessions gives exactly 20% per class; binned values are almost all unique, so ties do not distort bins." Step 12 also confirms the distributional consequence: "80.3% [of trials] contain all five classes, every trial contains at least two, and strong within-trial dominance (maximum 99.4%) is expected because percentiles are selected per session, not per 60-s segment."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The two-photon microscope triggers the camera, so imaging and video frames are nominally 1:1 at 30 Hz; the only failure mode is dropped camera frames, which make the motion array shorter. The AI does not use a fixed threshold on the interval values (whose stored units are ambiguous — Step 2 notes nominal differences of ~3.36e-5). Instead it computes the **median positive interframe interval** for that session and converts every interval into an integer number of frame steps, `steps = max(1, round(interval / median))`. The cumulative sum of these steps gives the true 30 Hz index of each observed camera sample; `np.interp` then resamples onto the complete `0 … n_frames-1` grid, which linearly interpolates exactly at the missing indices and leaves observed samples bit-identical. A guard raises if the reconstructed last index does not equal `expected_frames - 1`.

Sessions whose motion array is already full length are left completely untouched, including three `jm046` sessions that have anomalously large timestamp intervals but no missing samples. After alignment the motion trace is binned with the same 10-frame mean as the neural data, so alignment is preserved into the output grid; per-trial slicing uses the same `slice` object for all three streams.

I verified the gap detection independently: for all six sessions with deficits, the AI's `round(interval/median)` method and the human reference's `dt*1000 > 0.04` threshold identify **the same number of gaps at the same indices**, and every gap is a single frame (max step = 2), so the interpolated values are identical too. For `jm046/2024-09-05_a` (full length) the AI correctly inserts nothing, whereas a naive threshold would have flagged a gap.

ii.
```python
positive = intervals[intervals > 0]
if len(positive) == 0:
    raise ValueError(f"No positive camera intervals in {move_dir}")
median_interval = float(np.median(positive))
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
timestamp_gap_count = int(np.sum(interval_steps - 1))
missing = expected_frames - len(raw)

if missing:
    observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
    if observed_positions[-1] != expected_frames - 1:
        raise ValueError("Timestamp gaps do not explain frame deficit ...")
    aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)
else:
    # The release README defines dropped frames by a length mismatch. A few
    # full-length jm046 timestamp streams contain clock anomalies; no values
    # are inserted when no camera samples are absent.
    aligned = raw.copy()
```
```python
aligned_motion, motion_info = align_motion(ref.path / "move_deve", n_frames)
motion_binned, quantile_edges, labels = bin_and_discretize_motion(aligned_motion)
...
"interpolated_motion_frames": int(motion_info["missing_frames"]),
"timestamp_gap_count": int(motion_info["timestamp_gap_count"]),
```

iii. Step 4, "Camera gaps" row: "For deficient sessions, reconstruct 30-Hz index positions from rounded interval/median ratios and linearly interpolate inserted values. Leave full-length sessions untouched as instructed by the README length criterion; their timestamp anomalies do not identify absent array entries." Step 2 quantifies it: "Camera-frame deficits relative to neural frames occur in 10 sessions and total 276 missing frames … Timestamp gap counts agree exactly for these deficits." Step 10 check 5: "all 276 length deficits are explained by interval ratios and filled only at those indices."

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's general policy is **fail loudly on anything unexpected, repair only what the release documents, and never silently drop valid samples**. Concretely:
- *Dropped camera frames* (276 across 10 sessions, up to 148 in one session): repaired by timestamp-reconstructed linear interpolation, with a guard that the reconstruction must exactly explain the deficit. Counts stored per session in metadata.
- *Full-length sessions with timestamp anomalies* (`jm046` days 3/5/6): explicitly left alone, on the grounds that the README defines missing frames by length mismatch and no samples are actually absent.
- *Suite2p `badframes`* (one frame, `jm032/2023-10-24_a`): retained, because removing it would desynchronise behaviour; the count is recorded in metadata.
- *Paper/data disagreements* (tracked-cell counts differ from Figure 5 for four mice; four mice have 30-min not 20-min sessions): all released data kept, and the discrepancy written verbatim into `metadata['reference_discrepancies']` rather than papered over.
- *Structural anomalies*: incomplete session directories, mismatched stream shapes, ROI-row mismatches, failed `iscell` curation, unexpected `fs`/`nframes`, motion longer than neural, malformed motion arrays, non-finite neural or motion values, non-divisible session lengths, unequal quintiles, and wrong trial shapes each raise an explicit exception with the offending path.
- *Trailing partial trials*: cannot occur (all sessions are exact multiples of 60 s) and would raise rather than be silently discarded.

ii.
```python
if not all(p.exists() for p in required):
    raise FileNotFoundError(f"Incomplete session directory: {session_dir}")
...
if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
    raise ValueError(f"Packaged neurons do not all pass Track2p curation: {ref.path}")
if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
    raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
...
if not np.all(np.isfinite(result)):
    raise ValueError(f"Non-finite processed neural values in {plane_dir}")
...
if len(aligned) != expected_frames or not np.all(np.isfinite(aligned)):
    raise ValueError(f"Failed motion alignment in {move_dir}")
...
if not np.array_equal(counts, np.full(5, expected)):
    raise ValueError(f"Motion quintiles are not equal: counts={counts.tolist()}")
...
"reference_discrepancies": (
    "The released tracked-cell counts differ from Figure 5 for four "
    "mice, and four mice contain 30-minute rather than the paper's "
    "stated 20-minute sessions; all valid released rows/times are retained."
),
```

iii. Key Decision 1: "Preserve every valid supplied sample … The paper-duration discrepancy is documented rather than silently cropping valid source data." Key Decision 5 on interpolation. Step 4, "Neural bad frames": "Retain it; Suite2p has already processed it and 10-frame averaging reduces single-frame influence. Removing it would desynchronize behavior." Step 4, "Tracked-cell counts": "There is no mapping that could recreate absent paper rows, nor a principled basis to discard extra valid rows. Preserve every supplied curated cell and explicitly report the discrepancy."

## 6-a. What are the most time-consuming steps of the code?

i. By the script's own instrumentation (`conversion_full_out.txt`), the dominant cost is `baseline_correct_and_bin` — the Gaussian smoothing plus the two 1800-sample rolling min/max filters over every neuron of the full session. It accounts for essentially all of each session's runtime: 0.26 s of 0.30 s for a 221-neuron/36k-frame session and 1.99 s of 2.03 s for a 746-neuron/54k-frame session. Total conversion is 47.69 s of computation plus 0.32 s to serialise 395 MiB — far below the 15-minute threshold the instructions set, so no further optimisation was warranted. Secondary costs are disk I/O (F.npy and Fneu.npy are 161 MB each, `ops.npy` is 94 MB and is loaded in full with `allow_pickle=True` once per session) and, in `--show-processing` mode only, matplotlib rendering (~1.1–1.2 s/session).

ii.
```python
t0 = time.perf_counter()
neural_binned, trace_example = baseline_correct_and_bin(plane, n_neurons, n_frames)
neural_seconds = time.perf_counter() - t0
...
print(f"Converted {session_info['session_id']}: ... neural={neural_seconds:.2f}s total={total_seconds:.2f}s", flush=True)
```
Sample of the emitted timings:
```
Converted jm031/2023-10-18_a: 221 neurons, 36000 native frames -> 20 trials x 180 bins; motion missing=0; neural=0.26s total=0.30s
Converted jm039/2024-05-03_a: 746 neurons, 54000 native frames -> 30 trials x 180 bins; motion missing=0; neural=1.99s total=2.03s
Conversion computations finished in 47.69s
```

iii. Step 6: "Loading all F/Fneu into writable arrays simultaneously would require several hundred MB per session in addition to filtering intermediates … `ops.npy` is large and is loaded only once per session." Step 7 run-time table: "Core neural processing — 0.37 s for 221 neurons; scales approximately with neuron×frame count — about 20–30 s for all 41 sessions"; "Full conversion + pickle save … conservatively under 1 minute, far below 15-minute optimization threshold."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The code is already substantially vectorised; the remaining loops are either deliberate or negligible:
- `for start in range(0, n_neurons, NEURON_CHUNK)` in `baseline_correct_and_bin` — an intentional memory/throughput trade-off, not an oversight. Each iteration is fully vectorised across the 64 neurons in the chunk and along time. Removing the chunking would be *faster* but would need ~400–650 MB of float32 intermediates per session; raising `NEURON_CHUNK` from 64 to the full neuron count would be the only available speedup and it is small.
- `for trial in range(n_trials)` in `convert_session` — 20–30 iterations per session doing a slice plus `np.ascontiguousarray`. This could be replaced by a single `neural_binned.reshape(n_neurons, n_trials, TRIAL_BINS).transpose(1,0,2)` view/split, but the target format requires a Python list of per-trial arrays anyway, so the loop is essentially free (<0.05 s/session, visible as the gap between `neural=` and `total=` in the log).
- `for index, ref in enumerate(session_refs)` in `build_dataset` — inherently sequential per-session work; the AI considered and rejected parallelising it.

Notably, the one loop the human reference flags as vectorisable — repeated `np.insert` for dropped-frame repair, which reallocates the array on every insertion — the AI has **already vectorised** into a single `np.interp` call over reconstructed positions.

ii. The intentional chunk loop:
```python
for start in range(0, n_neurons, NEURON_CHUNK):
    stop = min(start + NEURON_CHUNK, n_neurons)
    ...
    baseline = gaussian_filter(corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES))
    baseline = minimum_filter1d(baseline, size=baseline_window, axis=1)
    baseline = maximum_filter1d(baseline, size=baseline_window, axis=1)
```
The already-vectorised gap repair:
```python
interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
aligned = np.interp(np.arange(expected_frames, dtype=np.float64), observed_positions, raw)
```

iii. Step 6, "Code speedups added": "Memory-mapped source arrays; 64-neuron processing chunks; vectorized SciPy filters across neurons; vectorized reshape/mean downsampling; session processing once before slicing; direct list construction and float32 outputs. No parallel disk reads were added because the 16-GB source is I/O-heavy and concurrent session filtering would multiply memory pressure."

## 6-c. What processing does the code repeat multiple times?

i. A few small redundancies, none material:
- `F.npy` and `Fneu.npy` are opened twice per session — once in `validate_source_session` and again in `baseline_correct_and_bin`. Both use `mmap_mode="r"` so this costs ~1.5 ms and no data is read twice.
- `ops.npy` (94 MB, `allow_pickle=True`) and `stat.npy` (4.7 MB object array) are fully deserialised once per session purely for validation; `ops` is then reused for the `badframes` count, but `stat` and `spks` are read only to check row counts. Measured cost ~0.03 s/session warm (a few hundred ms cold), i.e. under ~2% of runtime.
- `np.ascontiguousarray` in the trial loop copies each slice; the underlying `neural_binned` was already built in one pass, so this is a second traversal of the same ~340 MB of data.
- `build_dataset` re-scans `session_refs` with `next(i for i, r in ...)` once per subject to recompute the unique-track total (6 short scans, trivial).
- The `--show-processing` example traces (`trace_example`) are copied out of chunk 0 on **every** run, even when plotting is disabled (see 6-d).

Nothing is recomputed per trial: baseline correction, binning, quantile estimation and time construction all happen exactly once per session, before slicing.

ii.
```python
# in validate_source_session
f = np.load(plane / "F.npy", mmap_mode="r")
fneu = np.load(plane / "Fneu.npy", mmap_mode="r")
...
# again in baseline_correct_and_bin
f = np.load(plane_dir / "F.npy", mmap_mode="r")
fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
```
```python
unique_tracks = sum(
    neural[next(i for i, r in enumerate(session_refs) if r.subject == subject)][0].shape[0]
    for subject in subjects
)
```

iii. Step 6: "Repeating filters separately per trial would be both slow and scientifically incorrect at boundaries" and "session processing once before slicing" — the AI explicitly designed against the expensive form of repetition. The duplicated `np.load` calls are a side-effect of keeping validation in a separate function; the notes do not call them out, but they also record "Avoid unnecessary file I/O" via memory-mapping.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed and then unused by the decoder, all of them small and all of them defensible as validation/provenance:
- **`spks.npy`, `iscell.npy`, `stat.npy`** are loaded solely to assert shape/curation invariants. None of their values reach `converted_data.pkl` (deconvolved spikes are explicitly rejected in favour of baseline-corrected fluorescence).
- **`tstamps.npy`** is loaded and cast to float64, but the alignment is driven entirely by `interframe_int.npy`; the timestamps are only length-checked and stored in the plotting `info` dict.
- **`trace_example`** — six full-length copies of native-resolution traces for neuron 0 (`raw_f`, `raw_fneu`, `neuropil_corrected`, `baseline`, `activity`, `activity_binned`) — is built on every run even when `--show-processing` is off, and `align_motion` likewise always returns `raw`, `timestamps`, `observed_positions` and `interpolated_mask` for plotting. Guarding these behind the flag would be strictly cheaper.
- **`timestamp_gap_count`** is computed for every session, including full-length ones where it is not used for alignment (it only lands in metadata).
- **`ops['badframes']`** is counted but never acted on.
- The full-dataset assertion block recomputes totals that the validator also reports.

None of this changes the output values; total overhead is well under a second across the whole dataset.

ii.
```python
spks = np.load(plane / "spks.npy", mmap_mode="r")      # shape check only
stat = np.load(plane / "stat.npy", allow_pickle=True)  # row count only
...
timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
if len(raw) != len(timestamps) or len(intervals) != len(raw) - 1:
    raise ValueError(f"Malformed motion arrays in {move_dir}")
```
```python
if start == 0:                       # built unconditionally, used only when plotting
    example = {
        "raw_f": raw_f[0].copy(),
        "raw_fneu": raw_fneu[0].copy(),
        "neuropil_corrected": corrected_neuropil[0].copy(),
        "baseline": baseline[0].copy(),
        "activity": activity[0].copy(),
        "activity_binned": result[0].copy(),
    }
```
```python
"suite2p_badframes": int(np.count_nonzero(ops["badframes"])),
"timestamp_gap_count": int(motion_info["timestamp_gap_count"]),
```

iii. The AI frames all of this as required verification rather than waste. Step 5 planned sanity checks include "Assert F/Fneu/spks/stat/iscell row consistency, constant neuron count within subject, and all `iscell` probabilities >0.5" and "Assert expected totals: 6 subjects, 41 sessions, 2,998 unique longitudinal tracks, 20,445 session-neurons, 1,090 trials, and 196,200 decoder timepoints." Step 6 states the processing plots "contain eight audit panels", which is why the example traces are collected. Step 4 records that `badframes` was inspected and deliberately not acted upon.
