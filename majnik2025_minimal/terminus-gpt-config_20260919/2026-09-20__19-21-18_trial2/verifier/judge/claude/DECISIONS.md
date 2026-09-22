# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks the directory tree under `/app/data`. Subjects are the top-level directories whose name starts with `jm` (sorted); sessions are the sorted sub-directories of each subject (one per recording day). For each session it loads three groups of files:
- `suite2p/plane0/F.npy` and `suite2p/plane0/Fneu.npy` — raw and neuropil fluorescence of the Track2p-tracked ROIs;
- `suite2p/plane0/ops.npy` — read only to recover the Suite2p preprocessing parameters (`neucoeff`, `sig_baseline`, `win_baseline`, `fs`);
- `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy` — the global motion-energy trace and the camera timestamps.

`spks.npy`, `stat.npy` and `iscell.npy` are not loaded. Everything is loaded eagerly, session by session, in a single pass; all 41 sessions from all 6 subjects are retained (14 × 36 000 frames, 27 × 54 000 frames; 221–746 neurons).

ii.
```python
DATA_ROOT = Path('/app/data')
...
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
for si, subject in enumerate(subjects):
    sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
    for session in sessions:
        activity = suite2p_baseline_corrected(session)
        motion = aligned_motion(session, activity.shape[1])
```
```python
def suite2p_baseline_corrected(session: Path) -> np.ndarray:
    p = session / 'suite2p' / 'plane0'
    F = np.load(p / 'F.npy').astype(np.float32, copy=False)
    Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
    ops = np.load(p / 'ops.npy', allow_pickle=True).item()
```
```python
def aligned_motion(session: Path, nframes: int) -> np.ndarray:
    p = session / 'move_deve'
    motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
    stamps = np.load(p / 'tstamps.npy').astype(np.float64)
```

iii. From the trajectory: the AI first read `/app/data/README.md` and `load_data.ipynb`, which state that subject folders are named `jm0xx`, that each contains one folder per recording day, and that the exported `suite2p` folders already contain only the neurons tracked across all days. The notebook's own loader uses exactly the same `os.scandir` + sort idiom. The notebook comment "for more proper analysis compute dF/F the way as described in the paper (or alternatively use `spks.npy`)" led the AI to prefer `F`/`Fneu` with the paper's baseline correction over `spks.npy`. It also enumerated per-session array shapes and `ops` fields up front to confirm 41 sessions, a uniform 30 Hz rate, and identical Suite2p baseline parameters everywhere.

## 1-b. How are the data split into subjects (mice)?

i. One subject per top-level `jm*` directory, sorted alphabetically, giving `['jm031','jm032','jm038','jm039','jm040','jm046']`. The loop index over this list is stored as `subject_idx` for every session produced by that subject. No subject is excluded.

ii.
```python
subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith('jm'))
for si, subject in enumerate(subjects):
    ...
            subject_idx.append(si)
...
'subjects': subjects,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. The data README states that each subject has one folder named with its subject id and that "the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A … jm046 - mouse F)", so alphabetical sorting reproduces the paper's mouse ordering. The `jm` prefix filter also excludes the non-subject entries in the data root (`README.md`, `load_data.ipynb`).

## 1-c. How are the data split into sessions?

i. One session per daily recording folder (`YYYY-MM-DD_a`) inside a subject folder, sorted so days are in chronological order. Each daily recording becomes one entry of `neural` / `input` / `output` / `brain_region_idx` / `subject_idx`. All 41 recordings are kept; none are merged or dropped. A `session_info` record (subject, session name, n_neurons, n_trials, duration, source rate) is stored per session in `metadata`.

ii.
```python
sessions = sorted(p for p in (DATA_ROOT / subject).iterdir() if p.is_dir())
for session in sessions:
    print(f'Processing {subject}/{session.name}', flush=True)
    ...
    session_info.append({
        'subject': subject, 'session': session.name,
        'n_neurons': int(activity.shape[0]), 'n_trials': ntrials,
        'duration_seconds': float(usable / BIN_FS),
        'source_imaging_rate_hz': FS,
    })
```

iii. The README says each session folder "corresponds to one recording day" and the date format is `YYYY-MM-DD` (the trailing `_a` can be ignored), so lexicographic sorting is chronological. Days must stay separate because the Suite2p/Track2p export, the motion-energy normalisation and the percentile binning are all per-recording quantities.

## 1-d. How are the data split into trials?

i. There is no stimulus-driven trial structure (spontaneous activity in the dark), so trials are defined artificially as the instructions require: consecutive, non-overlapping 60 s blocks of each session. After 10-frame averaging the sampling rate is 3 Hz, so a trial is 180 bins. `ntrials = n_bins // 180`, and the tail that does not fill a whole trial is dropped — in practice the tail is always empty (36 000/10/180 = 20 trials, 54 000/10/180 = 30 trials exactly), which the AI notes in the module docstring.

ii.
```python
BIN_FS = FS / AVERAGE_FRAMES            # 3.0 Hz
TRIAL_SECONDS = 60
TRIAL_SAMPLES = int(TRIAL_SECONDS * BIN_FS)   # 180
...
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
```

iii. Module docstring: "Sessions divide exactly into complete, consecutive 60 s trials." The Decoder Task section of the instructions explicitly says "Split sessions into 60-second trials", and the format requires ≥2 trials per session with equal bin sizes, which contiguous fixed-length segmentation satisfies (20 or 30 trials per session).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. Every 60 s block of every session of every subject is kept (820 trials in total). The only data discarded would be a partial trailing block, and no session has one.

ii. There is no filtering code; the only exclusion is the (always empty) tail truncation shown in 1-d:
```python
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
```

iii. From the docstring: "Every supplied subject/session is retained." The distributed dataset is already curated (only cells tracked across all days, one continuous 20/30 min recording per day), there are no behavioural task events that could fail, and the paper applies no trial-level rejection because its analyses run on the continuous recording. The AI also checked the motion-energy quantiles of every session up front ("zero%", quintile edges) and found them all well separated, so no session looked degenerate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw fluorescence `F.npy` and the neuropil fluorescence `Fneu.npy` from `suite2p/plane0` of each session, plus `ops.npy` for the four preprocessing constants (`neucoeff` = 0.7, `sig_baseline` = 10, `win_baseline` = 60 s, `fs` = 30). `spks.npy` (deconvolved) is deliberately not used.

ii.
```python
p = session / 'suite2p' / 'plane0'
F = np.load(p / 'F.npy').astype(np.float32, copy=False)
Fneu = np.load(p / 'Fneu.npy').astype(np.float32, copy=False)
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
```

iii. The methods state "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses", and the decoding analysis in the paper operates on dF/F, not on deconvolved spikes. The AI reasoned (step 10): "Suite2p's `preprocess` generally baseline-subtracts neuropil-corrected fluorescence rather than dividing by baseline, matching the paper's phrase 'baseline corrected fluorescence traces as our dF/F'; we should reproduce that exact implementation." Reading the constants out of `ops.npy` rather than hardcoding them guarantees they match whatever Suite2p actually used for these recordings (the AI verified they are identical across all 41 sessions).

## 2-b. How is the `neural` data processed?

i. Three steps, reproducing Suite2p's `dcnv.preprocess(baseline='maximin')`:
1. neuropil subtraction, `F - 0.7 * Fneu`;
2. maximin baseline removal — Gaussian smoothing along time (σ = 10 frames), then a rolling minimum and a rolling maximum with a 60 s × 30 Hz = 1800-frame window, subtracted from the neuropil-corrected trace;
3. averaging of 10 consecutive frames (30 Hz → 3 Hz).

No z-scoring, no ΔF/F₀ division and no per-neuron normalisation are applied; the stored values are baseline-subtracted fluorescence in float32.

ii.
```python
corrected = F - neucoeff * Fneu
# Suite2p maximin: smooth, rolling minimum, rolling maximum, subtract.
smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                           axis=1, mode='reflect')
window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
corrected -= base
```
```python
def average_ten(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1] // AVERAGE_FRAMES
    x = x[..., :n * AVERAGE_FRAMES]
    return x.reshape(*x.shape[:-1], n, AVERAGE_FRAMES).mean(axis=-1)
```

iii. Docstring: "The paper used Suite2p-default baseline-corrected fluorescence. We form the pipeline's neuropil-corrected trace F - 0.7 Fneu and reproduce its maximin baseline (Gaussian sigma 10 frames, rolling min then max over 60 s)" and "As in the paper's decoding analysis, neural and behavioral traces are averaged over non-overlapping groups of 10 timestamps (30 Hz -> 3 Hz)". The AI read the actual Suite2p source (`suite2p/extraction/dcnv.py`) to copy the algorithm rather than guessing, and chose to reimplement it with SciPy to avoid the slow/possibly GPU-dependent Suite2p import that had stalled earlier in the session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is excluded. Every row of `F.npy` is kept, so each session of a subject contributes the same fixed neuron count (221, 370, 685, 746, 541, 435 for jm031…jm046; 20 445 neuron-sessions in total). `iscell.npy` is never loaded.

ii. There is no filtering code; the neuron axis is passed through untouched:
```python
region_idx.append(np.zeros(activity.shape[0], dtype=np.int64))
```

iii. Docstring: "Track2p's Suite2p exports already contain only cells tracked across every day of a subject. Their iscell scores exceed the paper's default 0.5 threshold, so no second ROI filter is applied." The AI verified this empirically before writing the converter (trajectory step 7: "the provided Track2p outputs already contain only successfully tracked cells and their iscell probabilities are above 0.5, so no additional cell filtering should be needed"). This is consistent with the methods: "We considered all ROIs above the default threshold of 0.5 as true cells" — a filter that has already been applied upstream. Keeping all rows also preserves the cross-day neuron matching that Track2p provides.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external alignment event. Trials are aligned to the start of the session: trial *k* covers bins `[k·180, (k+1)·180)`, i.e. seconds `[60k, 60(k+1))` of the recording, with no gaps or overlap. Metadata records `temporal_alignment_event = 'session start; consecutive 60-second trial segmentation'`, `off_start = 0.0` and `off_end = 60.0` (i.e. offsets relative to each trial's own onset).

ii.
```python
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
...
'temporal_alignment_event': 'session start; consecutive 60-second trial segmentation',
'off_start': 0.0,
'off_end': float(TRIAL_SECONDS),
'trial_duration_seconds': TRIAL_SECONDS,
```

iii. The recordings are continuous spontaneous-activity sessions performed "in the dark, under sensory-minimised conditions" with no stimuli or task events, so the only meaningful temporal reference is the start of the recording. Since the instructions ask for 60 s trials, each trial boundary is itself the alignment point, which is what `off_start`/`off_end` = 0/60 s encodes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes. The 30 Hz source data are rebinned by averaging 10 consecutive frames into non-overlapping bins, giving 3 Hz, i.e. a 333.33 ms bin. The same `average_ten` function is applied to the neural traces and to the (already frame-aligned) motion-energy trace, so both streams stay the same length and stay index-aligned. Binning is done *before* the motion energy is discretised into quintiles. `metadata['time_bin_size']` is 1000/3 = 333.33 ms and `temporal_averaging_frames` = 10 is also recorded.

ii.
```python
FS = 30.0
AVERAGE_FRAMES = 10
BIN_FS = FS / AVERAGE_FRAMES
...
activity = average_ten(activity).astype(np.float32)
motion = average_ten(motion)
labels = quintiles(motion)
...
'time_bin_size': 1000.0 / BIN_FS,
'temporal_averaging_frames': AVERAGE_FRAMES,
```

iii. The methods state, for the decoding analysis: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" (the same 10-frame averaging is also used for the event-rate analysis). The AI restates this in the docstring and reasoned in step 8 that this "yield[s] 3 Hz samples (333.33 ms bins), after which 60-second trials contain 180 samples."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Nothing in the raw files. Elapsed time is synthesised from the bin index and the known 3 Hz post-binning rate: `t[i] = i / 3` seconds, the left edge of bin *i*, measured from the first imaging frame of that session.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
```

iii. The imaging rate is a fixed 30 Hz for every session (the AI checked `ops['fs']` for all 41 sessions), and the camera timestamp clock is in arbitrary units, so counting bins is the exact and simplest way to express elapsed session time. "Elapsed-time input remains relative to session start (rather than resetting each trial)."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Essentially none beyond the arithmetic above. The vector is computed once per session over the whole binned recording, truncated with the same `usable` cut as the neural data, split into the same 60 s blocks, cast to float32 and stored with a leading singleton axis so each trial input is `(1, 180)`. Time does **not** reset at trial boundaries: trial 0 runs 0–59.67 s, trial 1 runs 60–119.67 s, and the last trial of a 30 min session ends at 1799.67 s. It is named `elapsed time from session start (s)`.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
...
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
inputs.append([np.ascontiguousarray(x[None, :], dtype=np.float32)
               for x in np.split(elapsed, ntrials)])
...
'input_names': ['elapsed time from session start (s)'],
```

iii. The Decoder Input specification asks for "Time elapsed from the beginning of the session in seconds. Time-varying." Keeping the counter running across trials (rather than restarting it at each artificial 60 s boundary) is what makes the variable informative — it tells the decoder where in the session a trial sits, which is the only contextual information available in an unstructured spontaneous recording.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: `elapsed` is built with exactly `activity.shape[1]` entries after binning, is truncated by the same `usable` index, and is split into the same number of trials, so element *i* of the input is the same 333.33 ms bin as column *i* of the neural matrix. Every trial is therefore `(1, 180)` against a `(n_neurons, 180)` neural matrix.

ii.
```python
elapsed = (np.arange(activity.shape[1], dtype=np.float32) / BIN_FS)
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
neural.append([... for x in np.split(activity, ntrials, axis=1)])
inputs.append([... for x in np.split(elapsed, ntrials)])
```

iii. Not separately justified in the trajectory — deriving the time axis from the neural array's own length makes misalignment impossible, which the AI treated as self-evident.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `move_deve/motion_energy_glob.npy` — the pre-computed global motion-energy trace from the behaviour video — together with `move_deve/tstamps.npy`, the camera frame timestamps, which are used to place the motion samples on the imaging frame grid. (`interframe_int.npy` is not read; it is exactly `np.diff(tstamps)`, so it carries the same information.)

ii.
```python
p = session / 'move_deve'
motion = np.load(p / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(p / 'tstamps.npy').astype(np.float64)
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
```

iii. The methods define motion energy as the summed squared pixel-wise difference of consecutive video frames, used "as a proxy of [the mouse's] arousal state"; the README says `motion_energy_glob.npy` is exactly that processed quantity, and that "the indices of missing frames can be obtained by looking at `tstamps.npy` or `interframe_int.npy`". The AI preferred `tstamps.npy` because, unlike the length deficit, timestamps locate *where* frames are missing (trajectory step 8).

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three stages:
1. **Resampling to the imaging frame grid** — a regular grid of `nframes` points is built starting at `stamps[0]` with a step equal to the median positive inter-frame interval, and the motion trace is linearly interpolated onto it (`np.interp`, clamped to the first/last sample outside the timestamp range).
2. **10-frame averaging** — the same `average_ten` used for the neural data, 30 Hz → 3 Hz.
3. **Discretisation** — per-session quintiles of the *averaged* trace.

No smoothing, log transform, or per-session amplitude normalisation is applied beyond the averaging, and the continuous value is not stored (only the class index, dtype int64).

ii.
```python
positive_dt = np.diff(stamps)
step = np.median(positive_dt[positive_dt > 0])
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```
```python
motion = average_ten(motion)
labels = quintiles(motion)
```

iii. Docstring: "As in the paper's decoding analysis, neural and behavioral traces are averaged over non-overlapping groups of 10 timestamps" and "The requested categorical target is formed after averaging". Discretising after averaging is necessary because averaging class labels would be meaningless; it also matches the paper, which denoises the behaviour trace before decoding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into five equal-percentile bins (quintiles) whose four boundaries (20/40/60/80th percentile) are recomputed **within each session** from that session's binned motion-energy trace. `np.searchsorted(edges, x, side='right')` maps values to classes 0–4, with class names `['lowest','low','middle','high','highest']`. The verifier confirms each class holds exactly 20.0 % of the bins in every session.

ii.
```python
def quintiles(x: np.ndarray) -> np.ndarray:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    # right=False gives classes 0..4 and is equivalent to cutting at percentile
    # boundaries. The data have no problematic percentile ties.
    return np.searchsorted(edges, x, side='right').astype(np.int64)
...
'output_names': ['motion energy quintile'],
'output_values': [['lowest', 'low', 'middle', 'high', 'highest']],
```

iii. The Decoder Output specification asks for "Motion energy, discretized into five equal-percentile bins, selected per session", so the per-session percentile choice is dictated by the instructions. The AI additionally checked in advance (trajectory step 8/9) that the motion-energy quantiles are well separated and that there are few zero-valued samples, concluding that "five percentile bins should be stable" and that ties would not collapse a class.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera is hardware-triggered by the microscope, so in principle camera sample *i* corresponds to imaging frame *i*; but some sessions have fewer motion samples than imaging frames (2, 3, 116, 148, 1 frames missing in six sessions) and three jm046 sessions have a timestamp gap even though the counts match. Rather than inserting frames at the gaps, the AI resamples the motion trace in *time*: it defines a uniform grid of `nframes` points spaced by the median camera inter-frame interval and linearly interpolates the motion samples onto it. This is applied to **every** session, including the 29 with no dropped frames. After resampling, both streams are binned by 10 and split identically, so trial *k* bin *i* of `output` sits at the same index as the neural column.

ii.
```python
def aligned_motion(session: Path, nframes: int) -> np.ndarray:
    """Interpolate camera motion samples onto the regular imaging-frame grid."""
    ...
    # The timestamp clock has arbitrary units; its median camera-frame interval
    # defines one regular 30 Hz step and preserves the observed dropped-frame gaps.
    positive_dt = np.diff(stamps)
    step = np.median(positive_dt[positive_dt > 0])
    target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
    return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
motion = aligned_motion(session, activity.shape[1])
motion = average_ten(motion)
labels = quintiles(motion)
```

iii. Docstring: "Camera frames occasionally dropped. Camera timestamps identify their exact locations, so motion energy is linearly interpolated to the regular 30 Hz two-photon frame grid rather than shifted or merely padded at the end." Trajectory step 8 adds the decisive observation: "a few jm046 recordings have gaps despite equal array lengths, so timestamp-based interpolation onto the neural frame grid is preferable to simple length padding" — i.e. a deficit-based scheme would silently miss those sessions.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four things:
- **Dropped camera frames** are handled by the timestamp interpolation of 4-d, which works uniformly whether the deficit is 1 or 148 frames and also covers the jm046 sessions where a timestamp gap exists without a length deficit.
- **Consistency check**: if `motion_energy_glob.npy` and `tstamps.npy` disagree in length the script raises `ValueError` instead of proceeding.
- **Out-of-range extrapolation** is clamped to the first/last motion sample rather than extrapolated (`left=`, `right=`).
- **Incomplete trailing data**: any bins beyond the last complete 60 s trial are dropped (always zero here). Sessions with fewer neurons or a different length are not special-cased — nothing is dropped for being short.

There is no NaN handling (the traces contain none) and no interpolation of neural data.

ii.
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
...
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
...
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
```

iii. The README explicitly sanctions interpolation: "In some recordings there might be some missing frames from the camera … they can be treated as missing values for motion energy or they can be interpolated over". The AI chose interpolation over masking because the decoder format has no missing-value representation, and chose a timestamp-driven method over a deficit-driven one for the jm046 reason above. The hard `ValueError` reflects a preference for failing loudly on an unexpected file rather than silently misaligning the two streams.

## 6-a. What are the most time-consuming steps of the code?

i. Per session the dominant cost is the maximin baseline estimation on the full `(n_neurons, 36 000–54 000)` matrix — measured on the largest session (746 × 54 000): `gaussian_filter1d` ≈ 0.78 s, `minimum_filter1d` ≈ 0.24 s, `maximum_filter1d` ≈ 0.12 s, versus ≈ 0.15 s to load `F`/`Fneu`/`ops` and ≈ 0.06 s for the neuropil subtraction. So the Gaussian smoothing alone is roughly half the per-session compute and the three-filter baseline block is ~85 % of it. Whole-dataset conversion is on the order of a minute, plus pickling the 396 MB output. A secondary cost is I/O: `ops.npy` is ~90 MB per session and is loaded in full for four scalars.

ii.
```python
smooth = gaussian_filter1d(corrected, float(ops.get('sig_baseline', 10.0)),
                           axis=1, mode='reflect')
window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
base = minimum_filter1d(smooth, size=window, axis=1, mode='reflect')
base = maximum_filter1d(base, size=window, axis=1, mode='reflect')
```

iii. Not discussed as a cost in the trajectory, but the AI did make a performance-relevant choice: after the `import suite2p` call hung (steps 8–9) it decided to read the Suite2p source and reimplement `preprocess` with SciPy — "Suite2p is installed, and its source can be read directly without triggering the slow package import" — which removes the torch/GPU dependency and the package-import overhead entirely.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little is left to vectorise. The only Python loops are over subjects and sessions, which are inherently serial (separate files, per-session percentile edges), and the two list comprehensions that wrap `np.split` — and those iterate only 20–30 times per session over views. Notably, the dropped-frame repair is already fully vectorised (`np.interp` on a precomputed grid) instead of the natural `np.insert`-in-a-loop formulation, which reallocates the array on every insertion and would be ~150 iterations in the worst session. The only measurable avoidable work is not a loop but redundant copying: `np.split` returns views and `np.ascontiguousarray` then materialises a second full copy of every session's neural data.

ii.
```python
for si, subject in enumerate(subjects):
    for session in sessions:
        ...
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
```
```python
# already vectorized instead of a per-dropped-frame insertion loop:
target = stamps[0] + np.arange(nframes, dtype=np.float64) * step
return np.interp(target, stamps, motion, left=motion[0], right=motion[-1])
```

iii. Not explicitly justified. The vectorised interpolation follows from the AI's decision to treat the problem as resampling onto a time grid rather than as patching individual missing indices, so the loop never needed to exist.

## 6-c. What processing does the code repeat multiple times?

i. Little genuine duplication. The repeated work is:
- `ops.npy` (~90 MB) is loaded once per session (41 times) purely to read four scalars — `neucoeff`, `sig_baseline`, `win_baseline`, `fs` — which the AI had already verified to be identical Suite2p defaults across all sessions.
- `average_ten` is called twice per session (neural, motion), which is necessary, not redundant.
- The neural data are written twice: once by `average_ten`/`astype` into the session array and again by `np.ascontiguousarray` into the 20–30 per-trial arrays.
- `float(ops.get(...))` conversions and `np.median` of the inter-frame intervals are recomputed per session, which is required because they are per-session quantities.

ii.
```python
ops = np.load(p / 'ops.npy', allow_pickle=True).item()
neucoeff = float(ops.get('neucoeff', 0.7))
...
window = int(float(ops.get('win_baseline', 60.0)) * float(ops.get('fs', FS)))
```

iii. Not discussed. Reading the parameters per session instead of hardcoding them is a deliberate robustness choice (they come from the file that actually produced the traces), at the cost of deserialising a large `ops` dictionary each time.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small no-ops, none of which affect the result:
- The trial-truncation logic (`usable`, the three slice assignments) never removes anything — the AI's own docstring states "Sessions divide exactly into complete, consecutive 60 s trials" (3600 or 5400 bins, both multiples of 180). The same is true of the tail drop inside `average_ten` (36 000 and 54 000 are multiples of 10).
- `left=motion[0], right=motion[-1]` only ever applies to the final grid point or two.
- `.astype(np.float32, copy=False)` after `average_ten` is a no-op for the neural array, and `np.ascontiguousarray(..., dtype=np.float32)` re-copies data that is already contiguous-compatible float32.
- Loading the full 90 MB `ops.npy` to extract four scalars — the mean images and the other ~129 keys are discarded immediately.
- The continuous (pre-discretisation) motion-energy values are computed at 30 Hz resolution and then thrown away; only the quintile labels are stored, so the decoder never sees the analogue trace.
- `session_info`'s `duration_seconds` / `source_imaging_rate_hz` fields are stored but unused by the decoder (harmless provenance metadata).

ii.
```python
ntrials = activity.shape[1] // TRIAL_SAMPLES
usable = ntrials * TRIAL_SAMPLES
activity, labels, elapsed = activity[:, :usable], labels[:usable], elapsed[:usable]
...
activity = average_ten(activity).astype(np.float32)
...
neural.append([np.ascontiguousarray(x, dtype=np.float32)
               for x in np.split(activity, ntrials, axis=1)])
```

iii. Not discussed in the trajectory. The truncation and clamping are defensive coding: they cost nothing and make the script correct for a recording whose length is not an exact multiple of 600 frames.
