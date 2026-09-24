# Decisions

> Documented from `/app/convert_data.py`, `/app/CONVERSION_NOTES.md`, `/app/README.md` and
> `/logs/agent/trajectory.json`.
>
> **Note on the prompt the AI actually received.** The trajectory (step 3) shows the AI was given
> an earlier revision of the task statement whose *Decoder Task* section read only:
> *"Decode information regarding animal motion from the neural activities recorded from mouse barrel
> cortex."*, inputs *"Time elapsed from the beginning of the experiment"*, outputs *"Motion energy,
> normalized and discretized into five equal-percentile bins."* It did **not** contain the sentences
> *"Split sessions into 60-second trials"* or *"selected per session"* that appear in
> `/tests/instruction_reference.md`. This is relevant to decisions 1-d and 4-c and is flagged again
> there.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. `discover_sessions()` walks `data/`, taking every directory whose name starts with `jm` as a
subject and every sub-directory whose first four characters are digits (i.e. a `YYYY-MM-DD_a` date
folder) as a session. This yields a flat, sorted list of 41 `SessionInfo(subject, session_id, path)`
records covering all 6 mice. For every session the script reads five raw files:
`suite2p/plane0/ops.npy` (metadata: `nframes`, `fs`, and the Suite2p preprocessing parameters),
`suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `move_deve/motion_energy_glob.npy` and
`move_deve/tstamps.npy`. `spks.npy`, `stat.npy`, `iscell.npy` and `interframe_int.npy` are
deliberately not used. Loading happens in two passes: pass 1 reads only the behavioural stream of
every session (to compute global quantile edges), pass 2 re-reads each session and loads the
neural arrays one session at a time. Trials are not present in the raw data; they are constructed
later by fixed-duration segmentation.

ii.
```python
def discover_sessions(sample: bool) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
        for session_dir in session_dirs:
            sessions.append(
                SessionInfo(
                    subject=subject_dir.name,
                    session_id=f"{subject_dir.name}_{session_dir.name}",
                    path=session_dir,
                )
            )
    ...
    return sessions


def load_ops(session: SessionInfo) -> dict:
    return np.load(
        session.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()
```

```python
    for session in sessions:                                    # pass 1 (behaviour only)
        ops = load_ops(session)
        nframes = int(ops["nframes"])
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

```python
    for idx, session in enumerate(sessions):                    # pass 2 (neural)
        ops = load_ops(session)
        F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
        Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
```

iii. From CONVERSION_NOTES Step 5 / Step 10: the released dataset is "the matched-across-all-days
Suite2p export" produced by Track2p, stored under each session's `suite2p/plane0`, so the
conversion "reads these directly". The AI explicitly rejected the Track2p GUI's `F_processing`
helper because "it defaults to `neucoeff=0.0`" and is therefore not the representation the paper
describes. Key decision 7: "Keep all six mice and all provided sessions... there is no principled
basis for dropping mice or sessions." The two-pass design is justified as a memory measure:
"Two-pass design avoids loading all neural sessions simultaneously" (Step 6).

*Verified:* the discovery logic finds exactly 41 sessions (7/7/7/7/6/7), i.e. every session folder
in the release; the stray `ground_truth.csv` files are excluded by the `is_dir()` test.

---

## 1-b. How are the data split into subjects?

i. One subject per top-level `jm*` directory. Subject names are taken from the folder names,
de-duplicated, sorted alphabetically, and each session is assigned the index of its parent folder.
Six subjects result: `jm031, jm032, jm038, jm039, jm040, jm046`. No data are ever merged across
mice.

ii.
```python
    for subject_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("jm")):
        ...
        sessions.append(SessionInfo(subject=subject_dir.name, ...))
```
```python
    subjects = sorted({session.subject for session in sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[session.subject] for session in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Sorted unique subject IDs; each session points to its
subject index... Sessions remain separate; no merging across mice", justified by the data
directory organisation. The `data/README.md` states each `jm*` folder is one mouse (mouse A–F).

---

## 1-c. How are the data split into sessions?

i. One session per date sub-folder (`YYYY-MM-DD_a`) inside a subject folder, kept in sorted (i.e.
chronological) order and never concatenated across days. 41 sessions in total: 7 for each of
jm031, jm032, jm038, jm039, jm046 and 6 for jm040. Sessions are 36,000 frames (20 min) for
jm031/jm032 and 54,000 frames (30 min) for the other four mice. Per-session provenance
(`session_id`, path, `nframes_raw`, `duration_sec`, `nneurons`, missing-frame counts) is recorded in
`metadata['session_info']`.

ii.
```python
        session_dirs = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
        )
```
```python
        session_meta.append(
            {
                "session_id": session.session_id,
                "subject": session.subject,
                "session_path": str(session.path),
                "nframes_raw": nframes,
                "duration_sec": nframes / float(ops["fs"]),
                "nneurons": int(np.load(session.path / "suite2p" / "plane0" / "F.npy", mmap_mode="r").shape[0]),
                "behavior_missing_frames": align_stats["missing_frames"],
                ...
            }
        )
```

iii. CONVERSION_NOTES Step 2: "Each subject folder contains a number of session folders, each
corresponding to one recording day"; Step 5: session folder name/date is stored in
`metadata['session_info']` as an audit trail. Step 9/10 record the deliberate decision to keep both
the 20-minute and 30-minute recordings even though the paper's methods text mentions only
20-minute sessions: "Conversion preserves actual durations from the release and remains internally
consistent."

---

## 1-d. How are the data split into trials?

i. The recordings are continuous spontaneous-behaviour sessions with no native trial structure, so
trials are created artificially as **consecutive, non-overlapping 120-second (2-minute) blocks**.
After 10-frame averaging (3 Hz) each block is `BLOCK_BINS = 120 × 30 / 10 = 360` time bins. A
20-minute session yields 10 trials and a 30-minute session yields 15 trials, giving 545 trials
overall. The neural, input and output streams are split with the same block boundaries. The code
*requires* the number of binned samples to be an exact multiple of 360 and raises rather than
dropping a remainder, and it raises if a session would produce fewer than 2 trials.

ii.
```python
BLOCK_DURATION_SEC = 120.0
BLOCK_BINS = int(BLOCK_DURATION_SEC * IMAGING_FS / MOTION_BIN_SIZE_FRAMES)   # 360
```
```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```
```python
        if neural_binned.shape[1] % BLOCK_BINS != 0:
            raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")

        neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
        input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
        output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
        ...
        if len(neural_trials) < 2:
            raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "Construct pseudo-trials as consecutive 2-minute
blocks. The native data have no trials, and the paper's decoder splits recordings into consecutive
2-minute blocks. After 10-frame averaging, each block contains `120 s * 3 Hz = 360` time bins. This
satisfies the target format while staying aligned to the published analysis." Trajectory step 83:
"I'm locking it to the paper's temporal handling: ... then 2-minute pseudo-trials."

The paper sentence being cited is: *"We used 5 fold splits for both the inner and outer loops,
splits were done on consecutive 2 minute blocks of the recording."* — i.e. 2 minutes is the paper's
**cross-validation fold unit**, not a trial definition. The AI's notes never mention a 60-second
trial length; the trial-length sentence was absent from the prompt revision it received.

---

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is performed — every 2-minute block of every session is kept
(545/545). Instead of filtering, the script applies hard structural assertions that abort the
conversion if anything is inconsistent: neural and motion binned lengths must be equal, binned
length must be an exact multiple of the block size, the three streams must produce equal trial
counts, and a session must yield at least two trials. In the full run none of these fired and
nothing was discarded.

ii.
```python
        if neural_binned.shape[1] != len(motion_disc):
            raise ValueError(
                f"{session.session_id}: neural/motion binned length mismatch "
                f"{neural_binned.shape[1]} vs {len(motion_disc)}"
            )
        if neural_binned.shape[1] % BLOCK_BINS != 0:
            raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
        ...
        if not (len(neural_trials) == len(input_trials) == len(output_trials)):
            raise ValueError(f"{session.session_id}: trial count mismatch after block splitting")
        if len(neural_trials) < 2:
            raise ValueError(f"{session.session_id}: needs at least two trials, got {len(neural_trials)}")
```

iii. The AI found no quality-control criterion for trials in the paper or code: the trials are its
own construct, the recordings are continuous spontaneous activity, and Key Decision 7 states there
is "no principled basis for dropping" data. CONVERSION_NOTES Step 10 edge-case check: "Sessions
with missing camera frames were converted without NaNs or length mismatches... No off-by-one errors
found at session ends; all trials have exactly `T=360`."

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `suite2p/plane0/F.npy` (raw ROI fluorescence, `n_neurons × n_frames`) and
`suite2p/plane0/Fneu.npy` (neuropil fluorescence), with `suite2p/plane0/ops.npy` supplying the
preprocessing parameters (`neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `fs`,
`prctile_baseline`, `batch_size`, `nframes`). `spks.npy` (deconvolved traces) and `iscell.npy` are
deliberately **not** used.

ii.
```python
        F = np.load(session.path / "suite2p" / "plane0" / "F.npy", allow_pickle=True)
        Fneu = np.load(session.path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True)
        ...
        neural = compute_suite2p_baseline_corrected(F, Fneu, ops)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "The paper explicitly states that subsequent analyses
used baseline-corrected fluorescence traces ('our dF/F') with default Suite2p parameters. The
Track2p GUI's `F_processing` is not faithful because it defaults to `neucoeff=0.0`. I will
therefore reconstruct the fluorescence using Suite2p's own preprocessing logic and the per-session
`ops.npy` parameters." Using `ops.npy` rather than hard-coded constants is justified as reproducing
exactly what the authors' own pipeline was configured with.

---

## 2-b. How is the `neural` data processed?

i. Three steps, in order: (1) neuropil subtraction `Fc = F − neucoeff·Fneu` with
`neucoeff = ops['neucoeff'] = 0.7`; (2) Suite2p's own `dcnv.preprocess` with the session's
`ops` parameters — `baseline='maximin'`, `win_baseline=60 s`, `sig_baseline=10`, `fs=30 Hz`,
`prctile_baseline=8`, forced onto `device='cpu'` — which Gaussian-filters, takes a running
minimum-then-maximum over a 60 s window and subtracts that baseline; (3) averaging over
non-overlapping 10-frame bins (30 Hz → 3 Hz) with the mean accumulated in float64 and stored as
float32. No z-scoring, no ΔF/F division, no smoothing beyond the above. Result per trial:
`(n_neurons, 360)` float32.

ii.
```python
def compute_suite2p_baseline_corrected(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    fc = F.astype(np.float32, copy=False) - float(ops.get("neucoeff", 0.7)) * Fneu.astype(
        np.float32, copy=False
    )
    return dcnv.preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", IMAGING_FS)),
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 100)),
        device=torch.device("cpu"),
    ).astype(np.float32, copy=False)
```
```python
def bin_average_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[1] // bin_size) * bin_size
    x = x[:, :usable]
    nbins = usable // bin_size
    return x.reshape(x.shape[0], nbins, bin_size).mean(axis=2, dtype=np.float64).astype(np.float32)
```

iii. Key Decision 1 (above) plus Key Decision 2: "Use 10-frame temporal averaging before
segmentation: This exactly matches the paper's decoding preprocessing and converts 30 Hz traces
into 3 Hz traces while denoising both neural and behavior data." Step 10 check 2 records an
independent raw-data sanity check: recomputing the Suite2p baseline correction for
`jm038/2023-04-30_a` and binning it gave `np.allclose == True` (max abs diff `3.05e-05`) against the
converted pickle.

*Verified independently:* recomputing `F − 0.7·Fneu → dcnv.preprocess(maximin, 60 s, sig 10, 8th
pct) → 10-frame mean` for `jm031/2023-10-18_a` reproduces the stored trial 0 to within
`1.5e-05` — numerically identical to the human reference pipeline.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied in the conversion. Every row of every session's `F.npy` is
kept (221/370/685/746/541/435 neurons for the six mice; 20,445 neuron-sessions in total), and all
neurons are labelled with a single brain-region index 0.

ii.
```python
        converted["brain_region_idx"].append(np.zeros(neural_binned.shape[0], dtype=np.int64))
```
(There is no `iscell` mask, no variance/SNR threshold and no NaN screen anywhere in the script.)

iii. CONVERSION_NOTES Step 10, check 5(b) "Neuron filtering": "Reference code uses `iscell > 0.5`
and removes rows with `None` in the Track2p match matrix. The release data already reflect this
filtering, so the conversion does not apply a second filtering pass." The `data/README.md` confirms
the export "only includes traces for the cells present across all days".

*Verified:* `iscell[:,0]` is all `1.0` in the released files, so a second `iscell` pass would be a
no-op — the AI's reasoning holds.

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to. Trials are contiguous blocks tiling the
session from frame 0 onward, so each trial is implicitly aligned to its own block onset, and block
onsets are at 0 s, 120 s, 240 s … from session start. Neural, input and output are cut at exactly
the same indices, so the three streams are aligned by construction. The metadata records this as
`temporal_alignment_event = "start of each consecutive 2-minute block; input stores absolute
elapsed time from session start"` with `off_start = 0.0` and `off_end = 120.0`.

ii.
```python
        "metadata": {
            ...
            "temporal_alignment_event": "start of each consecutive 2-minute block; input stores absolute elapsed time from session start",
            "off_start": 0.0,
            "off_end": BLOCK_DURATION_SEC,
```
```python
        neural_trials = split_into_blocks_2d(neural_binned, BLOCK_BINS)
        input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
        output_trials = [x[None, :].astype(np.int64) for x in split_into_blocks_1d(motion_disc, BLOCK_BINS)]
```

iii. Key Decision 5: "Use absolute elapsed time from session start as the sole decoder input: This
is mandated by the task and remains paper-consistent because the recordings are continuous
spontaneous sessions rather than event-locked trials." Key Decision 4: "Treat imaging frames as the
reference clock", so the imaging frame grid defines the alignment for all streams.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning is applied. The raw imaging and behaviour streams are at 30 Hz (33.33 ms/frame).
Both are averaged over non-overlapping bins of `MOTION_BIN_SIZE_FRAMES = 10` consecutive frames,
giving 3 Hz, i.e. a **333.33 ms** time bin, which is recorded in `metadata['time_bin_size']`. Every
trial therefore has 360 bins and every session/trial uses the same bin size. Binning is applied to
the continuous motion-energy trace *before* discretisation, and to the baseline-corrected
fluorescence *before* trial splitting. Any tail shorter than a full 10-frame bin is dropped by the
binning helpers (in practice all sessions are exact multiples of 10 after frame-drop
interpolation).

ii.
```python
MOTION_BIN_SIZE_FRAMES = 10
IMAGING_FS = 30.0
...
            "time_bin_size": 1000.0 * MOTION_BIN_SIZE_FRAMES / IMAGING_FS,   # 333.33 ms
```
```python
def bin_average_1d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (len(x) // bin_size) * bin_size
    x = x[:usable]
    return x.reshape(-1, bin_size).mean(axis=1, dtype=np.float64).astype(np.float32)
```
```python
        motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)   # pass 1
        ...
        neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)           # pass 2
```

iii. Key Decision 2: "Use 10-frame temporal averaging before segmentation: This exactly matches the
paper's decoding preprocessing and converts 30 Hz traces into 3 Hz traces while denoising both
neural and behavior data." The paper methods state: *"For all decoding analysis we slightly denoised
the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."* Key
Decision 6 adds that discretisation happens only after this averaging "to minimize distortion".

---

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Not derived from any stored time variable. It is computed analytically from the bin index and
the imaging sampling rate `ops['fs']` (30 Hz in every session), using the number of binned
timepoints produced from the neural data. The behavioural `tstamps.npy` is *not* used for the time
input — only for motion alignment.

ii.
```python
def make_time_input(nbins: int, fs: float, bin_size_frames: int) -> np.ndarray:
    step = bin_size_frames / fs
    return (np.arange(nbins, dtype=np.float32) * step)[None, :]
```
```python
        time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
```

iii. CONVERSION_NOTES Step 5 mapping table: source is the "Imaging frame clock derived from
`ops['fs']`", justified by "Paper says recordings at 30 Hz; target task specifies time-elapsed
input". Step 10 check 3 reports the raw-data sanity check: the elapsed-time vector reconstructed
directly from frame indices at 30 Hz matched the converted `input` for `jm038/2023-04-30_a` trial 2
with `np.allclose == True`, max abs difference `0.0`.

---

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. `t[k] = k × (10 / 30) = k × 0.3333 s`, the left edge of each 333.33 ms bin, measured from the
first imaging frame of the **session** (not of the trial). The full-session vector is then cut with
the same block boundaries as neural/output, so time runs continuously across trials: trial 0 spans
[0, 119.67] s, trial 1 starts at 120.0 s, and a 30-minute session ends at 1799.67 s. Stored as
`(1, 360)` float32 per trial under the name `elapsed_time_sec`. No normalisation, offsetting or
per-trial resetting is applied.

ii.
```python
        input_trials = [x[None, :].astype(np.float32) for x in split_into_blocks_1d(time_binned, BLOCK_BINS)]
```
```python
        "input_names": ["elapsed_time_sec"],
```

iii. CONVERSION_NOTES Step 5 mapping table: "Create time elapsed from session start in seconds for
every 10-frame bin; split into 2-minute blocks so each trial gets a `(1, 360)` time series of
absolute within-session elapsed time... Input is absolute elapsed time from session start, not
relative block time." Key Decision 5 gives the rationale (the task mandates elapsed time; the
sessions are continuous, so absolute within-session time is the meaningful quantity).

Verified in `verification_full_out.txt`: per-session input ranges are `[0.0, 1199.7]` for the 14
twenty-minute sessions and `[0.0, 1799.7]` for the 27 thirty-minute sessions.

---

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction. The time vector is generated with exactly `neural_binned.shape[1]` entries from
the same 10-frame bin grid used for the neural data, and is then sliced with the identical block
boundaries, so bin *k* of the input is the same 333.33 ms window as bin *k* of the neural matrix.
An explicit check confirms the neural and motion binned lengths agree before splitting, and a
second check confirms the three streams produce the same number of trials.

ii.
```python
        neural_binned = bin_average_2d(neural, MOTION_BIN_SIZE_FRAMES)
        time_binned = make_time_input(neural_binned.shape[1], float(ops["fs"]), MOTION_BIN_SIZE_FRAMES)[0]
        ...
        if not (len(neural_trials) == len(input_trials) == len(output_trials)):
            raise ValueError(f"{session.session_id}: trial count mismatch after block splitting")
```

iii. Key Decision 4: "Treat imaging frames as the reference clock." Because the input is derived
from the imaging frame index itself, no cross-stream interpolation is needed, and the AI's Step 5
sanity-check list includes verifying that `input[0]` equals the elapsed-time vector derived from
frame indices at `fs=30` (confirmed in Step 10 check 3 with max abs difference `0.0`).

---

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `move_deve/motion_energy_glob.npy` (the pre-computed global motion-energy trace from the
behaviour video, stored as `uint64`) together with `move_deve/tstamps.npy` (per-video-frame
timestamps, same length as the motion trace). `ops['nframes']` / `F.shape[1]` supplies the target
length. Notably, the AI uses `tstamps.npy` and **not** `interframe_int.npy` for the missing-frame
problem.

ii.
```python
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
        motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
```

iii. CONVERSION_NOTES Step 5 mapping table cites "`data/README.md` guidance for missing frames;
paper says 30 Hz synchronized camera and 10-timestamp averaging". The `data/README.md` says the
indices of missing frames "can be obtained by looking at `tstamps.npy` or `interframe_int.npy`";
the AI chose the timestamp route because it directly yields the position of every sample on the
imaging clock rather than requiring a threshold on inter-frame intervals. Key Decision 4: "motion
energy will be aligned to the imaging grid using `tstamps.npy`, then interpolated over missing
positions, matching the data README guidance."

---

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps. (1) **Alignment / gap filling** (`align_motion_to_imaging`): if the motion trace is
already `nframes` long it is used as-is; otherwise each behaviour sample's timestamp is mapped
linearly onto the imaging frame index grid `round((t − t₀)/(t_end − t₀) × (nframes − 1))`, samples
falling in the same frame are averaged (`np.add.at`), and frames with no sample are filled by
`np.interp` between the neighbouring known frames. (2) **Binning**: 10-frame averaging to 3 Hz.
(3) **Normalisation**: a single global min–max rescale to [0, 1] using the min and max over the
concatenation of every included session's binned trace. (4) **Discretisation**: `np.digitize`
against the four global quintile edges, producing integer labels 0–4 stored as `(1, 360)` int64 per
trial. Because min–max rescaling is a monotone affine map, step (3) has no effect on step (4)'s
result; it exists only because the AI's prompt asked for a "normalized" output.

ii.
```python
def align_motion_to_imaging(motion, tstamps, nframes):
    if len(motion) == nframes:
        aligned = motion.astype(np.float32, copy=False)
        ...
        return aligned, stats
    if len(motion) != len(tstamps):
        raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")
    denom = tstamps[-1] - tstamps[0]
    if denom <= 0:
        raise ValueError("Non-increasing behavior timestamps")
    frame_idx = np.round((tstamps - tstamps[0]) / denom * (nframes - 1)).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, nframes - 1)
    summed = np.zeros(nframes, dtype=np.float64)
    counts = np.zeros(nframes, dtype=np.int64)
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
    aligned = np.full(nframes, np.nan, dtype=np.float64)
    good = counts > 0
    aligned[good] = summed[good] / counts[good]
    known_idx = np.flatnonzero(good)
    ...
        missing = np.flatnonzero(~good)
        if missing.size:
            aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```
```python
        motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
    ...
    all_motion = np.concatenate(all_motion_binned)
    motion_min = float(all_motion.min())
    motion_max = float(all_motion.max())
    motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
    quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

iii. Key Decision 6: "Discretize motion energy only at the final step: All alignment, interpolation,
and 10-frame averaging will operate on continuous motion energy first. Discretization into five
equal-percentile bins happens after paper-consistent preprocessing to minimize distortion."
Key Decision 4 covers the alignment step. Step 10 check 4 records the raw-data sanity check:
reconstructing the whole chain for `jm031/2023-10-22_a` (the session with 116 dropped camera
frames) directly from `motion_energy_glob.npy` and `tstamps.npy` reproduced the converted trial with
`np.allclose == True`.

---

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into **five equal-percentile (quintile) bins whose edges are computed globally**, over the
concatenation of the 10-frame-averaged motion traces of *all* included sessions, rather than
per session. Pass 1 computes `np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8])` once; pass 2
applies these same four edges to every session with `np.digitize(..., right=False)`, yielding
labels 0–4 named `0-20pct … 80-100pct`. The edges are stored in
`metadata['preprocessing']['motion_quantile_edges']`. The *pooled* distribution is therefore exactly
20 % per class, but the *per-session* distributions are strongly skewed: from
`verification_full_out.txt`, session 0 never reaches level 0, session 39/40 (jm046, late days)
contain only levels 3 and 4, and 7 of 41 sessions are missing 1–3 of the 5 classes entirely.
In `--sample` mode the "global" edges are computed over only the 2 sampled sessions, so sample and
full runs use different thresholds.

ii.
```python
    all_motion = np.concatenate(all_motion_binned)
    motion_min = float(all_motion.min())
    motion_max = float(all_motion.max())
    motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
    quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```
```python
def motion_to_bins(x: np.ndarray, quantile_edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, quantile_edges, right=False).astype(np.int64)
```
```python
        motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(np.float32)
        motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)
```
```python
        "output_values": [["0-20pct", "20-40pct", "40-60pct", "60-80pct", "80-100pct"]],
```

iii. CONVERSION_NOTES Step 5 mapping table: "Normalize globally to `[0,1]` across included sessions
after 10-frame averaging; discretize into 5 equal-percentile bins using global quintiles... **Global
thresholds preserve across-session comparability.**" Trajectory step 83: "motion binned into global
quintiles only at the final step." Step 10 check 6 treats the pooled 20/20/20/20/20 split as the
success criterion — "Output distribution: exactly balanced globally by construction (`0.2` each
bin)" — and the AI's planned distribution sanity check was likewise stated only at the global level
("motion bins should be close to 20% each globally after quintile discretization"). The AI never
examined or commented on the per-session class coverage.

---

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The imaging frame grid is the master clock. When the camera dropped frames (9 of 41 sessions:
2, 3, 116, 2, 2, 148, 1, 1 and 1 missing frames), the behaviour samples are re-placed on the imaging
grid via their timestamps and the empty frames are linearly interpolated, so the aligned trace has
exactly `nframes` entries. The aligned trace is then binned on the same 10-frame grid as the neural
data and split at the same block boundaries; an explicit length check guards the alignment. Counts
of missing frames and of duplicate-timestamp frames are recorded per session in
`metadata['session_info']`.

ii.
```python
        motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
        motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
```
```python
        if neural_binned.shape[1] != len(motion_disc):
            raise ValueError(
                f"{session.session_id}: neural/motion binned length mismatch "
                f"{neural_binned.shape[1]} vs {len(motion_disc)}"
            )
```
```python
                "behavior_missing_frames": align_stats["missing_frames"],
                "behavior_duplicate_timestamp_bins": align_stats["duplicate_timestamp_bins"],
```

iii. Key Decision 4: "Treat imaging frames as the reference clock: Because the behavior stream
occasionally has missing frames, motion energy will be aligned to the imaging grid using
`tstamps.npy`, then interpolated over missing positions, matching the data README guidance."
Step 10 check 5(c): "Track2p code has no behavior loader; paper/README state the camera is
synchronized to imaging and that missing frames should be handled using timestamps. The conversion
aligns behavior to imaging using `tstamps.npy` and interpolates only missing camera frames." Step 7
adds a visual check: "Motion alignment plots show no obvious discontinuities or offset between raw
and aligned traces"; "Discretized motion bins track low- and high-motion epochs rather than
flickering randomly, which argues against a temporal shift bug."

*Verified independently:* for all 9 short sessions the timestamp mapping recovers exactly the
number of missing frames needed, with zero duplicate collisions, and places them at the same
indices the reference's `interframe_int`-threshold method would (e.g. `jm031/2023-10-22_a`
gaps at 654, 1500, 2260, 3753, 3880, 4534 … identical under both methods).

---

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Three classes of issue are handled.
 * **Dropped camera frames** (9 sessions, 1–148 frames): detected as a length mismatch between
   `motion_energy_glob.npy` and `ops['nframes']`, then repaired by timestamp re-gridding plus
   `np.interp` over the empty frames (see 4-b/4-d). A guard raises if motion and timestamps
   disagree in length, or if the timestamps are not increasing.
 * **Multiple behaviour samples landing on one imaging frame**: averaged rather than dropped, and
   counted in `behavior_duplicate_timestamp_bins`.
 * **Tail data that does not fill a whole bin**: silently dropped by `bin_average_1d`/`2d`.
 * **Tail data that does not fill a whole trial**: *not* dropped — the script raises
   `ValueError("binned timepoints not divisible by BLOCK_BINS")`. Every released session happens to
   be an exact multiple of 360 bins, so this never fired, but it makes the script abort rather than
   degrade on a session of non-multiple length.
There is no NaN handling downstream of the interpolation (none survive), and no session, mouse or
neuron is dropped for any reason. Structural invariants are re-checked by calling the decoder's own
`verify_data_format` before the pickle is written.

ii.
```python
    if len(motion) != len(tstamps):
        raise ValueError(f"motion/tstamps length mismatch: {len(motion)} vs {len(tstamps)}")
    denom = tstamps[-1] - tstamps[0]
    if denom <= 0:
        raise ValueError("Non-increasing behavior timestamps")
    ...
    known_idx = np.flatnonzero(good)
    if known_idx.size == 0:
        raise ValueError("No valid behavior samples after timestamp mapping")
    if known_idx.size == 1:
        aligned[:] = aligned[known_idx[0]]
    else:
        missing = np.flatnonzero(~good)
        if missing.size:
            aligned[missing] = np.interp(missing, known_idx, aligned[known_idx])
```
```python
        if neural_binned.shape[1] % BLOCK_BINS != 0:
            raise ValueError(f"{session.session_id}: binned timepoints not divisible by {BLOCK_BINS}")
```
```python
    valid, errors, warnings = verify_data_format(converted)
    if not valid:
        raise RuntimeError("Converted data failed validation:\n" + "\n".join(errors))
```

iii. CONVERSION_NOTES Step 2 enumerates the 9 mismatching sessions and their exact missing-frame
counts; Key Decision 4 and Step 10 check 5(c) justify the timestamp-based repair from the
`data/README.md` ("they can be interpolated over"). Step 10 edge-case check: "Sessions with missing
camera frames were converted without NaNs or length mismatches. Both 20-minute and 30-minute
sessions convert to exact multiples of 2-minute trials. No off-by-one errors found at session ends."
Two known paper-vs-release discrepancies (paper says 20-minute sessions and 526 ± 190 neurons; the
release has 20- and 30-minute sessions and 499.7 ± 180.5 neurons) were investigated and
deliberately left unchanged because "forcing agreement would require unjustified filtering".

---

## 6-a. What are the most time-consuming steps of the code?

i. Suite2p's `dcnv.preprocess` baseline correction, run on CPU, session by session — together with
loading the full `F.npy`/`Fneu.npy` arrays it accounts for essentially all of pass 2. The timing
instrumentation in `conversion_full_out.txt` shows pass 1 (behaviour scan over all 41 sessions)
taking 2.39 s, pass 2 taking 57.79 s, and the whole conversion 60.97 s; per-session cost scales
with neurons × frames (0.40 s for 221 neurons × 36,000 frames, 2.4 s for 746 neurons × 54,000
frames). Pickling the 396 MB result is the only other non-trivial cost.

ii.
```python
def summarize_timing(start_time: float, label: str) -> None:
    print(f"{label}: {time.perf_counter() - start_time:.2f}s")
...
    summarize_timing(pass1_start, "Pass 1 (behavior scan)")
...
        print(
            f"[{idx + 1}/{len(sessions)}] {session.session_id}: "
            f"neurons={neural_binned.shape[0]}, trials={len(neural_trials)}, "
            f"binned_T={neural_binned.shape[1]} in {time.perf_counter() - session_start:.2f}s"
        )
    summarize_timing(pass2_start, "Pass 2 (neural conversion)")
    ...
    summarize_timing(all_start, "Total conversion time")
```

iii. CONVERSION_NOTES Step 6, "Code inefficiencies identified": "Suite2p baseline correction
requires loading full `F.npy` and `Fneu.npy` arrays for each session. Full conversion may still be
moderately expensive because baseline correction is performed per session on CPU." Step 7 projected
~120 s for the full run (actual: 61 s), "far below 15 minutes", so no further optimisation was
judged necessary.

---

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy numerical work is already vectorised — notably the dropped-frame repair, which is done
with `np.add.at` + a single `np.interp` call instead of an element-by-element insertion loop. What
remains:
 * `split_into_blocks_1d` / `split_into_blocks_2d` build a Python list of slices in a loop; a single
   `reshape` + `np.moveaxis` (or just keeping the reshaped array) would avoid the per-block Python
   overhead. This is negligible (≤ 15 iterations per session).
 * `np.add.at(summed, frame_idx, ...)` is NumPy's *un*buffered, notably slow scatter-add;
   `np.bincount(frame_idx, weights=motion, minlength=nframes)` is the vectorised equivalent and is
   typically an order of magnitude faster. It only runs for the 9 short sessions.
 * The outer per-session loops in pass 1 and pass 2 are serial. The instructions suggested parallel
   processing; since `dcnv.preprocess` dominates and sessions are independent, a
   `ProcessPoolExecutor` over sessions would have given a near-linear speed-up. The AI chose not to.
 * `list(map(astype))` over the 10–15 block arrays in `converted["neural"].append([...])` performs a
   redundant `astype(np.float32, copy=False)` on data that is already float32.

ii.
```python
def split_into_blocks_2d(x: np.ndarray, block_bins: int) -> list[np.ndarray]:
    nblocks = x.shape[1] // block_bins
    usable = nblocks * block_bins
    x = x[:, :usable]
    return [x[:, i * block_bins : (i + 1) * block_bins] for i in range(nblocks)]
```
```python
    np.add.at(summed, frame_idx, motion.astype(np.float64))
    np.add.at(counts, frame_idx, 1)
```
```python
        converted["neural"].append([x.astype(np.float32, copy=False) for x in neural_trials])
```

iii. The AI did not identify any of these. Its Step 6 "Code speedups added" list is about memory and
I/O rather than vectorisation: "Two-pass design avoids loading all neural sessions simultaneously.
Motion quantile computation stores only 10-frame-averaged behavior values, which are small. Neural
data are processed one session at a time... Used `mmap_mode='r'` when only shape inspection was
needed." Its stated reason for stopping there is the Step 7 runtime projection: "No efficiency
intervention required before full conversion because projected runtime is far below 15 minutes."

---

## 6-c. What processing does the code repeat multiple times?

i. Several things are done twice because of the two-pass structure:
 * `ops.npy` is loaded 2× per session (once in pass 1, once in pass 2), and 3× for the first two
   sessions in `--sample` mode (`discover_sessions` loads it as well).
 * `motion_energy_glob.npy` and `tstamps.npy` are loaded 2× per session.
 * `align_motion_to_imaging` is **run twice** per session on identical inputs. The pass-2 result
   (`motion_aligned`) is only ever consumed by `plot_processing`, so on a `--full` run it is
   computed 41 times and thrown away 41 times.
 * `bin_average_1d` results are correctly memoised in `motion_binned_by_session`, so the binning
   itself is not repeated — only the load + align that produce it.
 * The min–max normalisation of the binned motion is computed once globally and then recomputed
   per session on the same values.
 * `verify_data_format` is run inside `convert_data.py` and then again by the separate
   `train_decoder.py --verify-only` step.
The cost is small — pass 1 is 2.39 s of a 61 s run — and the two-pass structure is genuinely
necessary given the AI's decision to use *global* quantile edges, which cannot be known until every
session's behaviour has been seen.

ii.
```python
    for session in sessions:                                  # pass 1
        ops = load_ops(session)
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
        motion_aligned, align_stats = align_motion_to_imaging(motion, tstamps, nframes)
        motion_binned = bin_average_1d(motion_aligned, MOTION_BIN_SIZE_FRAMES)
        motion_binned_by_session[session.session_id] = motion_binned
```
```python
    for idx, session in enumerate(sessions):                  # pass 2 — same loads again
        ops = load_ops(session)
        ...
        motion = np.load(session.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        tstamps = np.load(session.path / "move_deve" / "tstamps.npy", allow_pickle=True)
        motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])   # only used for plots
        motion_binned = motion_binned_by_session[session.session_id]               # cached
```

iii. The AI documents the two-pass design as a deliberate memory trade-off (Step 6: "Two-pass design
avoids loading all neural sessions simultaneously"; "Motion quantile computation stores only
10-frame-averaged behavior values, which are small") and caches the expensive part
(`motion_binned_by_session`). It does not acknowledge the duplicated raw loads or the wasted
second `align_motion_to_imaging` call.

---

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items, all cheap:
 * **Global min–max normalisation of motion energy.** `motion_norm_all = (x − min)/(max − min)` is a
   monotone affine map, so the quintile edges and the resulting `np.digitize` labels are bit-for-bit
   the same as if it had been skipped. The normalised continuous trace is never saved — only the
   integer labels are — so the rescale affects nothing but the numeric value of the stored
   `motion_quantile_edges`. It exists solely because the AI's prompt asked for a "normalized"
   output.
 * **The second `align_motion_to_imaging` call in pass 2.** Its output feeds only
   `plot_processing`, which runs for at most 2 sessions and only under `--show-processing`; in a
   `--full` run all 41 results are discarded.
 * **A separate memory-mapped load of `F.npy` in pass 1** purely to record `nneurons` in
   `session_info`, when pass 2 loads the same array in full a moment later.
 * **The in-script `verify_data_format` call**, duplicating the mandated
   `train_decoder.py --verify-only` step.
Nothing computationally heavy is wasted: the expensive `dcnv.preprocess` output is used in full,
`spks.npy`/`iscell.npy`/`stat.npy` are never read, and the raw 30 Hz traces are binned before being
stored rather than after.

ii.
```python
    motion_norm_all = (all_motion - motion_min) / max(motion_max - motion_min, 1e-12)
    quantile_edges = np.quantile(motion_norm_all, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    ...
        motion_binned_norm = ((motion_binned - motion_min) / max(motion_max - motion_min, 1e-12)).astype(np.float32)
        motion_disc = motion_to_bins(motion_binned_norm, quantile_edges)   # identical without the rescale
```
```python
        motion_aligned, _ = align_motion_to_imaging(motion, tstamps, F.shape[1])   # plotting only
```
```python
                "nneurons": int(np.load(session.path / "suite2p" / "plane0" / "F.npy", mmap_mode="r").shape[0]),
```
```python
    valid, errors, warnings = verify_data_format(converted)
```

iii. The AI does not flag any of these as unnecessary. The normalisation is presented as a
requirement, not an inefficiency: Step 5 mapping table, "Normalize globally to `[0,1]` across
included sessions after 10-frame averaging; discretize into 5 equal-percentile bins using global
quintiles", because "the target task requires normalized 5-bin categorical output". The `mmap_mode`
load is presented as a *speedup* in Step 6 ("Used `mmap_mode='r'` when only shape inspection was
needed"), and the built-in verification as a safety feature ("Added built-in validation via
`verify_data_format` before writing the pickle"). Overall runtime (61 s) was far inside budget, so
none of this was revisited.
