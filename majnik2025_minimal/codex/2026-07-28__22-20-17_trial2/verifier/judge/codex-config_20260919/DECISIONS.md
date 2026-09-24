# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the six mouse IDs, finds date-prefixed session directories, and loads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy` for every selected session. Full mode includes all such sessions; sample mode limits each mouse to its first session.

ii.
```python
FULL_SUBJECTS = ["jm031", "jm032", "jm038", "jm039", "jm040", "jm046"]
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
```

iii. The trajectory says the agent confirmed that the directory contents correspond to the paper's six mice and 6–7 consecutive recording days per mouse. It chose explicit subjects and date-prefixed folders to capture the delivered daily recordings deterministically.

## 1-b. How are the data split into subjects?

i. Subject identity is the parent mouse directory. The output subject list is reconstructed as the sorted unique subjects present in the loaded records, and each session gets an integer lookup index.

ii.
```python
subjects = sorted({record.subject for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_idx[record.subject])
```

iii. The agent justified this from the data layout and its check that the six directories are the six paper mice.

## 1-c. How are the data split into sessions?

i. Each date-prefixed directory below a mouse is one session. Sessions are sorted lexically and each `SessionRecord` becomes one top-level session in `neural`, `input`, and `output`.

ii.
```python
sessions = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
session_records = [load_session(session_dir=session_dir, ...) for session_dir in session_dirs]
neural.append(record.neural_trials)
```

iii. The trajectory describes the source as continuous per-day recordings and confirms that the folders represent consecutive days.

## 1-d. How are the data split into trials?

i. The agent splits each continuous session into non-overlapping 120-second blocks after 10-frame averaging, yielding 360 samples per trial at 3 Hz. It drops the incomplete final block. This conflicts with the explicit instruction to use 60-second trials.

ii.
```python
trial_duration_s: float = 120.0
bins_per_trial = int(round(trial_duration_s / time_bin_s))
n_complete_trials = neural_binned.shape[1] // bins_per_trial
for trial_idx in range(n_complete_trials):
    start = trial_idx * bins_per_trial
    end = start + bins_per_trial
```

iii. The agent deliberately prioritized the paper's two-minute cross-validation blocks: its trajectory says it chose a split “faithful to the 2-minute block analysis in the paper,” despite the decoder task's direct 60-second requirement.

## 1-e. How are trials filtered based on quality controls?

i. There is no content-based trial quality filter. Only incomplete trailing blocks are discarded, and all complete blocks are retained.

ii.
```python
n_complete_trials = neural_binned.shape[1] // bins_per_trial
neural_binned = neural_binned[:, : n_complete_trials * bins_per_trial]
motion_binned = motion_binned[: n_complete_trials * bins_per_trial]
```

iii. The agent found no natural trials or stated trial-level exclusion criterion, so it retained every complete fixed-duration segment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from Suite2p `F.npy` and `Fneu.npy`; preprocessing parameters come from `ops.npy`. `iscell.npy` is checked as a quality invariant but does not numerically contribute to the trace.

ii.
```python
F = np.load(suite2p_dir / "F.npy")
Fneu = np.load(suite2p_dir / "Fneu.npy")
ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
iscell = np.load(suite2p_dir / "iscell.npy")
neural = baseline_correct_fluorescence(F=F, Fneu=Fneu, ops=ops, device=device)
```

iii. The agent inspected Track2p and Suite2p code and concluded these were the Track2p-matched fluorescence rows and the relevant source for the paper's baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. It subtracts neuropil (`F - neucoeff * Fneu`), applies Suite2p baseline preprocessing using each session's `ops.npy` parameters (normally maximin), casts to float32, and averages non-overlapping groups of 10 frames. A local scipy implementation is used if Suite2p is unavailable.

ii.
```python
Fc = F.astype(np.float32, copy=False) - neucoeff * Fneu.astype(np.float32, copy=False)
return suite2p_preprocess(Fc.copy(), baseline=baseline,
    win_baseline=win_baseline, sig_baseline=sig_baseline, fs=fs,
    prctile_baseline=prctile_baseline, batch_size=128, device=device)
neural_binned = average_nonoverlapping(neural, frame_bin)
```

iii. The trajectory records a specific investigation of the Suite2p routine and concludes that the paper used neuropil subtraction plus maximin baseline subtraction, without an additional division by baseline. Ten-frame averaging was taken directly from the paper's decoding method.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No rows are filtered during conversion. The script instead requires all stored `iscell` probabilities to be at least 0.5 and aborts otherwise, assuming Track2p has already curated the exported cells.

ii.
```python
iscell_prob = iscell[:, 1]
if np.any(iscell_prob < 0.5):
    raise ValueError(f"{session_dir}: expected Track2p output already filtered at iscell >= 0.5")
```

iii. The agent inspected the files, found every stored ROI above 0.5, and reasoned that the source already contains cells tracked across days, so an additional filter would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Neural trials are consecutive blocks from the session start. Metadata incorrectly describes each block start as the alignment event while setting offsets to 0 and 120 seconds.

ii.
```python
neural_trials.append(neural_binned[:, start:end])
"temporal_alignment_event": "start of each consecutive 2-minute block cut from a continuous session",
"off_start": 0.0,
"off_end": float(trial_duration_s),
```

iii. The agent treated the continuous recording as having no stimulus event and used block boundaries; its stated rationale was to mirror the paper's two-minute decoding blocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten raw 30-Hz frames are averaged per bin, producing 3-Hz samples and a nominal 333.33-ms bin size. Metadata hard-codes 30 Hz even though processing uses each session's `ops["fs"]` for trial timing.

ii.
```python
frame_bin: int = 10
neural_binned = average_nonoverlapping(neural, frame_bin)
"time_bin_size": float((frame_bin / 30.0) * 1000.0)
```

iii. The agent explicitly cites the paper's instruction to denoise neural and behavioral traces by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index, the frame-bin size, and `ops["fs"]`; it does not use the loaded camera timestamps.

ii.
```python
time_bin_s = frame_bin / fs
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
```

iii. The agent regarded a constant imaging rate as sufficient for absolute session time and documented the input as time from session start.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The code constructs bin-center times in seconds, starting at half a bin (about 0.1667 s), then slices the continuous time vector into trials without resetting it.

ii.
```python
time_axis = (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * time_bin_s
input_trials.append(time_axis[start:end][np.newaxis, :].astype(np.float32, copy=False))
```

iii. The agent intentionally chose bin centers and described this in metadata; no deeper rationale appears beyond accurately representing absolute session time.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector has one value per binned neural column and is sliced with exactly the same `start:end` bounds as neural data.

ii.
```python
neural_trials.append(neural_binned[:, start:end])
input_trials.append(time_axis[start:end][np.newaxis, :])
```

iii. The shared bin index and shared slices were intended to preserve exact sample-wise alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output values derive from `motion_energy_glob.npy`. `interframe_int.npy` identifies camera gaps, while `tstamps.npy` is loaded and passed into alignment but never actually used.

ii.
```python
motion_energy = np.load(move_dir / "motion_energy_glob.npy")
tstamps = np.load(move_dir / "tstamps.npy")
interframe_int = np.load(move_dir / "interframe_int.npy")
motion_aligned, motion_info = align_motion_to_imaging(motion_energy, tstamps, interframe_int, neural.shape[1])
```

iii. The trajectory says motion energy is the paper's behavioral trace and timing metadata is needed to handle documented camera dropouts.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The agent repairs length mismatches, averages 10 frames, splits into trials, min–max normalizes the retained values independently within each session, then categorizes them using quintile edges calculated globally across all normalized sessions in the export.

ii.
```python
motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], frame_bin)[0]
normalized = minmax_normalize(concatenated)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. The agent said it wanted the same 10-frame denoising as the paper and chose per-session normalization plus export-wide global quintiles. It validated that this produced balanced classes and above-chance decoding, but did not justify why global thresholds satisfy the explicit per-session requirement.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four global quantiles (20%, 40%, 60%, 80%) are computed after concatenating all per-session-normalized retained motion values. `np.digitize` maps each sample to classes 0–4. Thus thresholds are not selected independently per session as instructed.

ii.
```python
all_motion_concat = np.concatenate(all_motion, axis=0).astype(np.float32)
global_edges = np.quantile(all_motion_concat, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
bins = np.digitize(trial, global_edges, right=False).astype(np.int64)
```

iii. The trajectory and notes explicitly describe “global quintile bins within each export” and cite exact overall balance as a sanity check. This was intentional, though contrary to “selected per session.”

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. If motion is shorter, large interframe intervals (`>1.5 × median`) identify insertion locations; NaNs are inserted and linearly interpolated, with any remaining deficit padded by NaNs and interpolated. Excess motion samples are trimmed. The aligned trace is then binned and sliced with neural bounds.

ii.
```python
gap_idx = np.flatnonzero(interframe_int > 1.5 * median_dt)
repaired = np.insert(repaired, insert_at, np.nan)
repaired = interpolate_nans_1d(repaired)
if diff < 0:
    return motion_energy[:target_frames], info
output_trials.append(motion_binned[start:end][np.newaxis, :])
```

iii. The agent investigated gap patterns and chose to repair only traces whose lengths differ, to avoid shifting already length-matched data. It considered interpolation defensible because the data README permits treating missing frames as missing or interpolating them.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion traces are repaired at detected gaps and interpolated; unresolved deficits become end padding before interpolation. Long traces are silently trimmed. All-NaN data become zeros. Incomplete temporal bins and incomplete final trials are truncated. Unexpected low `iscell` probabilities cause a hard error.

ii.
```python
pad = np.full(target_frames - repaired.shape[0], np.nan, dtype=np.float32)
repaired = interpolate_nans_1d(repaired)
return motion_energy[:target_frames], info
trimmed = arr[..., : n_complete * bin_size]
```

iii. The agent cited documented missing camera frames, chose interpolation to preserve alignment, and retained full delivered session durations despite a methods-text duration discrepancy. Its trajectory emphasizes not silently shifting behavior relative to imaging.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading full arrays and Suite2p/scipy baseline correction over every neuron and frame. Full-dataset conversion and decoder validation were run repeatedly, adding substantial total runtime.

ii.
```python
return suite2p_preprocess(Fc.copy(), ..., batch_size=128, device=device)
session_records = [load_session(...) for session_dir in session_dirs]
```

iii. The trajectory spends considerable effort validating the baseline routine and runs sample/full conversion and training multiple times. It uses CUDA when available to accelerate preprocessing.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Repeated `np.insert` for each camera gap reallocates the array and could be replaced with one preallocated/vectorized reconstruction. Trial slicing and per-trial summary/discretization loops could also be consolidated, although their cost is smaller.

ii.
```python
for pos in gap_positions:
    repaired = np.insert(repaired, insert_at, np.nan)
for trial in record.output_continuous_trials:
    bins = np.digitize(trial, global_edges, right=False)
```

iii. No explicit efficiency justification is recorded. The agent focused on clarity and correctness of gap placement; only a small number of dropped frames was expected.

## 6-c. What processing does the code repeat multiple times?

i. It concatenates each session's trials for normalization, rebuilds them, then iterates them again to concatenate all outputs, discretize them, compute summaries, and later count categories. Conversion itself was also executed repeatedly during validation.

ii.
```python
concatenated = np.concatenate([trial.reshape(-1) for trial in record.output_continuous_trials])
for record in session_records:
    for trial in record.output_continuous_trials:
        all_motion.append(trial.reshape(-1))
```

iii. The trajectory explains repeated executions as sample/full sanity checks and decoder validation. It gives no rationale for repeated in-memory traversals.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tstamps` is loaded, converted, and passed but unused. `math` is imported but unused. Per-trial motion summaries and extensive session metadata do not feed the decoder. The normalization pass is unnecessary relative to the required per-session percentile categorization because monotonic min–max scaling would not change within-session quantile membership.

ii.
```python
import math
tstamps = np.asarray(tstamps)
info["motion_trial_summary_before_discretization"] = normalized_trials[:2]
```

iii. The extra metadata and summaries were intended for auditability and sanity checking. The trajectory shows `tstamps` was initially investigated for synchronization, but the final alignment algorithm only uses interframe intervals and lengths.
