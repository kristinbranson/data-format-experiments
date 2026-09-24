# Decisions

> Source material: `/app/convert_data.py`, `/app/CONVERSION_NOTES.md`, `/app/README.md`,
> `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`, and the agent trajectory
> `/logs/agent/trajectory.json` (codex / gpt-5.4, 182 steps).

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI walks the `data/` tree itself rather than using the provided `load_data.ipynb`. `discover_sessions()` collects every directory under `data/` whose name starts with `jm` as a subject, and within each subject every directory whose name is at least 10 characters and begins with 4 digits (i.e. the `YYYY-MM-DD_a` session folders) as a session. Each `(subject, session_id, path)` triple is stored in a frozen `SessionRef` dataclass, and sessions are processed one at a time in a single serial loop.

For every session it loads five arrays/objects:
* `suite2p/plane0/F.npy` — raw ROI fluorescence
* `suite2p/plane0/Fneu.npy` — neuropil fluorescence
* `suite2p/plane0/ops.npy` — suite2p options dict, used to source `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `fs`, `prctile_baseline` instead of hard-coding them
* `move_deve/motion_energy_glob.npy` — global motion energy from the behaviour video
* `move_deve/tstamps.npy` — camera timestamps

It does **not** load `spks.npy`, `iscell.npy`, `stat.npy`, `interframe_int.npy` or `ground_truth.csv` in the conversion path (it inspected `iscell` and `spks` during exploration only). All 41 sessions from 6 subjects are converted (confirmed in `conversion_full_out.txt`: `Found 41 sessions to convert`, and `verification_full_out.txt`: 41 sessions, 20,445 neuron-rows).

ii. <Code snippets>

```python
def discover_sessions(data_root: Path) -> list[SessionRef]:
    sessions: list[SessionRef] = []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
    return sessions
```

```python
def process_session(session: SessionRef, show_processing: bool = False) -> tuple[dict, dict]:
    suite2p_dir = session.path / "suite2p" / "plane0"
    move_dir = session.path / "move_deve"

    F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
    Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)

    fs = float(ops["fs"])
```

iii. <Justification>

From `CONVERSION_NOTES.md` Step 2/Step 4: the AI established from `data/README.md` and from the code in `/app/code/track2p` that the shipped `suite2p/plane0` exports are *already* the Track2p output ("save the outputs in suite2p format"), containing only cells matched across all days of a mouse. It therefore decided to "load the already exported tracked `suite2p` files directly, which is consistent with the provided post-Track2p dataset" and not to re-run any Track2p matching. It read `ops.npy` so that the preprocessing parameters are taken from the recording itself rather than assumed (Step 5 Variable Mapping: "Neuropil subtraction using `ops['neucoeff']`, then Suite2p-style baseline correction ... using `suite2p.extraction.dcnv.preprocess`"). It chose `tstamps.npy` over `interframe_int.npy` because `data/README.md` says missing camera frames "can be obtained by looking at `tstamps.npy` or `interframe_int.npy`".

---

## 1-b. How are the data split into subjects?

i. <Decisions>

One subject per `jm*` directory. The subject list is the sorted set of subject names over the discovered sessions, and `subject_idx` is the index of each session's subject in that list, in session order. Result: 6 subjects (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) with 7/7/7/7/6/7 sessions, matching the raw directory tree and the paper's "6 mice".

ii. <Code snippets>

```python
for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("jm")):
```

```python
def build_dataset(converted_sessions: list[dict], session_refs: list[SessionRef]) -> dict:
    subjects = sorted({session.subject for session in session_refs})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[session.subject] for session in session_refs], dtype=np.int64),
```

iii. <Justification>

`CONVERSION_NOTES.md` Step 2 quotes `data/README.md`: "For each subject there is a folder corresponding to the subject id ... the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A ... jm046 - mouse F)". The AI cross-checked the per-subject tracked-neuron counts (221, 370, 685, 746, 541, 435; mean 499.7) against the paper's "on average 526 (± 190 std) neurons per mouse" and judged them consistent (Step 4 discrepancy table).

---

## 1-c. How are the data split into sessions?

i. <Decisions>

One session per daily recording directory (`YYYY-MM-DD_a`) inside each subject folder; sessions are kept in sorted (chronological) order, and subjects are iterated in sorted order, so the exported session order is `jm031` day1..day7, `jm032` day1..day7, etc. No session is dropped: all 41 are exported, including the 6-session subject `jm040`. The date-prefix filter (`len(name) >= 10 and name[:4].isdigit()`) is what excludes any non-session files/folders.

ii. <Code snippets>

```python
        for session_dir in sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and len(p.name) >= 10 and p.name[:4].isdigit()
        ):
            sessions.append(
                SessionRef(subject=subject_dir.name, session_id=session_dir.name, path=session_dir)
            )
```

```python
            "session_ids": [f"{session.subject}/{session.session_id}" for session in session_refs],
            "session_info": [session["summary"] for session in converted_sessions],
```

iii. <Justification>

Step 4 of `CONVERSION_NOTES.md`: "Session count | Example notebook shows 7-day sequences ... One subject has 6 sessions, others 7 | Paper says minimum 6 consecutive days | 6-day subject is consistent with paper; include all subjects." Step 5 Key Decision 7: "Include all 41 sessions because they match the paper's 'minimum 6 consecutive days' criterion and already represent curated tracked-cell exports." The AI also recorded in Step 4 that sessions are not all 20 min as the methods text says — `jm031`/`jm032` have 36,000 frames (20 min) and the rest 54,000 frames (30 min) — and resolved to "use actual per-session frame counts from data".

---

## 1-d. How are the data split into trials?

i. <Decisions>

There is no native trial structure (continuous spontaneous-behaviour recordings), so the AI synthesises trials as **consecutive, non-overlapping 120-second (2-minute) blocks** of each session: `TRIAL_SECONDS = 120.0`, giving `bins_per_trial = round(120 * 30 / 10) = 360` bins of 333.3 ms. Splitting is done *after* 10-frame binning, on the binned neural/time/output streams jointly. Any tail shorter than a full 360-bin trial is truncated and discarded. This yields 10 trials for the 20-minute sessions and 15 for the 30-minute sessions — 545 trials in total, every trial exactly `T = 360`.

Note this differs from the human reference, which uses 60-second trials (180 bins, 1090 trials total).

ii. <Code snippets>

```python
TRIAL_SECONDS = 120.0
```

```python
def split_trials(neural_binned, time_binned_s, output_one_hot, fs, bin_frames):
    bins_per_trial = int(round(TRIAL_SECONDS * fs / bin_frames))
    usable_bins = (neural_binned.shape[1] // bins_per_trial) * bins_per_trial
    if usable_bins < bins_per_trial:
        raise ValueError("Session is too short to form even one 2-minute trial.")

    neural_binned = neural_binned[:, :usable_bins]
    time_binned_s = time_binned_s[:usable_bins]
    output_one_hot = output_one_hot[:, :usable_bins]

    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

iii. <Justification>

Step 5 Key Decision 2: "**Trial definition**: Define each trial as one consecutive 2-minute block from a continuous session, matching the paper's decoding split unit exactly." Step 3 records the supporting reading: "Decoding in the paper uses ridge regression from neural activity to behavioral motion, nested 5x5 cross-validation, consecutive 2-minute blocks, and 10-frame temporal averaging". Trajectory step 73: "including the specific choice to use 2-minute blocks as trials because that matches the paper's decoding splits exactly." Key Decision 9 adds the uniformity argument: "Use 360 time bins per trial for every session because 2 minutes / 333.3 ms = 360; 20-minute sessions contribute 10 trials, 30-minute sessions contribute 15 trials."

Important context: the Decoder Task text actually delivered to the agent (trajectory step 3) reads only "Decode information regarding animal motion from the neural activities recorded from mouse barrel cortex." — it does **not** contain the "Split sessions into 60-second trials." sentence that appears in `/tests/instruction_reference.md`. The agent had no instruction-level trial length to follow and derived one from the paper.

---

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

No quality-based trial rejection is performed; every complete 120-s block is kept. The only trial-level exclusion is structural truncation of the sub-trial remainder at the end of a session (0 s for every session here, since 36,000 and 54,000 frames are both exact multiples of 3,600 frames). The AI instead adds *assertive* per-session validation that raises rather than filters: at least 2 trials per session, matching trial counts across `neural`/`input`/`output`, constant neuron count, matching time dimensions, no NaNs, and exactly-one-hot outputs.

ii. <Code snippets>

```python
def validate_converted_session(converted: dict) -> None:
    n_trials = len(neural)
    if n_trials < 2:
        raise ValueError("Each converted session must contain at least 2 trials.")
    if not (len(input_) == len(output) == n_trials):
        raise ValueError("Neural/input/output trial counts do not match.")
    n_neurons = neural[0].shape[0]
    for trial_idx in range(n_trials):
        if neural[trial_idx].shape[0] != n_neurons:
            raise ValueError("Neuron count changed across trials within a session.")
        if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
            raise ValueError("Input time dimension does not match neural time dimension.")
        ...
        if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
            raise ValueError("Converted arrays must not contain NaN values.")
```

iii. <Justification>

Step 3 "Trial curation rules": "No native trials are defined in the source dataset or paper for this analysis. Decoder evaluation in the paper operates on consecutive 2-minute temporal blocks from continuous recordings rather than stimulus-locked trials." Because the blocks are arbitrary partitions of continuous spontaneous activity, there is no quality criterion in the paper to inherit. Step 5 planned Check 5 ("Verify every session has at least 10 trials and all trials have shape (n_neurons, 360)") is implemented as the hard validation above; Step 10 records that it passed for all 41 sessions.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

`neural` is derived from suite2p `plane0` `F.npy` (ROI fluorescence) and `Fneu.npy` (neuropil fluorescence), with `ops.npy` supplying the preprocessing parameters. Deconvolved `spks.npy` is deliberately **not** used.

ii. <Code snippets>

```python
    F = np.load(suite2p_dir / "F.npy", allow_pickle=True)
    Fneu = np.load(suite2p_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(suite2p_dir / "ops.npy", allow_pickle=True).item()
    ...
    neural_processed = compute_fluorescence_signal(F, Fneu, ops)
```

iii. <Justification>

Step 4 discrepancy table: "Neural signal type | Track2p package mainly exports `F`, `Fneu`, `spks`; GUI has optional fluorescence preprocessing | Data includes `F`, `Fneu`, `spks` but no explicit saved `dF/F.npy` | Paper states downstream analyses used baseline-corrected fluorescence traces as dF/F | Conversion should reconstruct a paper-consistent fluorescence representation from `F`/`Fneu` rather than use raw `F` directly; `spks` remains a fallback for checks." Step 1 identifies `F_processing` in `code/track2p/gui/data_management.py` as the reference implementation of `F - neucoeff*Fneu` followed by baseline removal.

---

## 2-b. How is the `neural` data processed?

i. <Decisions>

Three steps, in order:
1. Neuropil subtraction with the recording's own coefficient: `Fc = F - ops['neucoeff'] * Fneu` (`neucoeff = 0.7` in every session).
2. suite2p's `dcnv.preprocess` with all parameters taken from `ops`: `baseline='maximin'`, `win_baseline=60.0`, `sig_baseline=10.0`, `fs=30.0`, `prctile_baseline=8.0`. This is a Gaussian filter + running min/max baseline estimate that is then subtracted — i.e. baseline-corrected fluorescence, *not* a division by F0 and *not* deconvolution.
3. Non-overlapping averaging of 10 consecutive frames (see 2-e).

No z-scoring, no per-neuron normalisation, stored as `float32`. I verified this is **bit-identical** to the human reference's neural pipeline: running the reference's call (`batch_size=128`) and the AI's call (`batch_size=min(512, max(32, n_neurons))`) on `jm031/2023-10-18_a` gives `np.allclose == True`, max abs diff `0.0` (batch size only controls chunking).

ii. <Code snippets>

```python
def compute_fluorescence_signal(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    """Approximate the paper's Suite2p-based dF/F signal."""
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    processed = suite2p_preprocess(
        Fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=float(ops["fs"]),
        prctile_baseline=float(ops["prctile_baseline"]),
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
    )
    return processed.astype(np.float32, copy=False)
```

```python
    neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
```

iii. <Justification>

Step 3: "Neural signal used for most downstream analyses in the paper: baseline-corrected fluorescence traces treated as dF/F using default Suite2p parameters." Step 5 Key Decision 1: "Use neuropil-subtracted, baseline-corrected fluorescence derived from `F` and `Fneu` with Suite2p defaults from `ops.npy`, because the paper states downstream analyses used baseline-corrected fluorescence as dF/F." Step 10 Check 2 records an `np.allclose` spot-check recomputing the whole chain from raw `F`/`Fneu`/`ops` for `jm046/2024-09-08_a`, trial 11, with max abs diff `0.0`.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

No neuron is filtered out at conversion time. Every row of `F.npy` becomes a row of `neural`, so the exported dataset carries 20,445 neuron-rows (221/370/685/746/541/435 per subject; mean 498.66 per session). The AI's stated position is that the curation has already been applied upstream: the shipped exports are suite2p `iscell > 0.5` ROIs that Track2p matched across every day of the mouse. All neurons are assigned to one brain region, `"barrel cortex"`.

I confirmed the premise independently: `iscell[:,0]` is all `1.0` and `iscell[:,1] > 0.5` for 100% of rows in the sessions checked, and neuron counts are constant across days within a subject.

ii. <Code snippets>

```python
    converted = {
        "neural": neural_trials,
        ...
        "brain_region_idx": np.zeros(F.shape[0], dtype=np.int64),
```

```python
        "brain_regions": [BRAIN_REGION],           # BRAIN_REGION = "barrel cortex"
        "brain_region_idx": [session["brain_region_idx"] for session in converted_sessions],
```

iii. <Justification>

Step 4: "ROI curation | Track2p defaults to `iscell_thr = 0.5` and filters ROIs before matching/export | Provided `suite2p` exports already have all rows passing `iscell` and the same row count across all days within a mouse | ROIs above 0.5 classifier probability are considered cells | Treat provided rows as already curated tracked cells". Step 10 Check 3: "Converted data does not re-filter rows because the provided exports already satisfy that curation and have constant matched row counts across days within each mouse." The `barrel cortex` label comes from Step 5 Key Decision 8 ("the recordings all come from barrel cortex layer 2/3").

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

There is no stimulus or behavioural event to align to; the recordings are continuous. Each trial is therefore aligned to the start of its own 2-minute block, which is at a fixed offset from the session start. The AI encodes this in metadata as `temporal_alignment_event = "start of each consecutive 2-minute recording block"`, `off_start = 0.0`, `off_end = 120.0`. Alignment across the three streams is index-based: neural, input, and output are all sliced with the *same* `[start:stop]` indices inside `split_trials`, so a temporal offset in one stream is impossible by construction.

The human reference makes the same structural choice but labels it `temporal_alignment_event = 'session_start'` with `off_start = off_end = None`.

ii. <Code snippets>

```python
    for start in range(0, usable_bins, bins_per_trial):
        stop = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
        output_trials.append(output_one_hot[:, start:stop].astype(np.int64, copy=False))
```

```python
            "temporal_alignment_event": "start of each consecutive 2-minute recording block",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
```

iii. <Justification>

Step 2: "There is no native trial structure in the source data; recordings are continuous spontaneous-behavior sessions. Trialization will need to be created during conversion." Step 5 Key Decision 5 states the time input is "time from the beginning of the recording session, carried through each 2-minute trial as an absolute-within-session time vector", so a trial is identified by its offset from session start rather than by an event. The `off_start`/`off_end` pair is filled in to describe the block window rather than left `None`.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

Yes — rebinning is applied. The raw acquisition is 30 Hz (33.3 ms); both the processed fluorescence and the frame-aligned motion-energy trace are averaged over **non-overlapping bins of 10 consecutive frames**, giving a final resolution of **333.33 ms (3 Hz)**, written to metadata as `time_bin_size = 1000 * 10 / 30 = 333.333`. The same `average_nonoverlapping` helper is applied to both streams, with the same tail-truncation rule, so they cannot drift apart in length. Binning is applied *before* discretisation of motion energy and *before* trial splitting; 120 s therefore equals exactly 360 bins for every trial in the dataset (`T: mean 360.00, min 360, max 360` in `verification_full_out.txt`).

ii. <Code snippets>

```python
BIN_FRAMES = 10

def average_nonoverlapping(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n_frames = x.shape[-1]
    usable = (n_frames // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError(f"Not enough frames to bin: got {n_frames}, need at least {bin_frames}")
    x = x[..., :usable]
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

```python
    neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
    ...
    motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
    ...
            "time_bin_size": float(1000.0 * BIN_FRAMES / 30.0),
```

iii. <Justification>

Step 3 Expected Statistics table: "Neural data time bin | Raw acquisition 30 Hz; decoding analysis uses 10-frame averages (333.3 ms) | 'Imaging rate was 30 Hz' / 'averaging in bins of 10 consecutive timestamps'", and the identical row for behaviour. Step 5 Key Decision 3: "Apply non-overlapping 10-frame averaging before trialization for both neural and behavioral streams, matching the paper's decoding preprocessing and giving a final bin size of 333.3 ms." Step 10 Check 3 (binning): "paper decoding uses 10 consecutive timestamps and 2-minute blocks. `convert_data.py` uses non-overlapping 10-frame binning followed by consecutive 2-minute trials, matching the paper."

---

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. <Decisions>

Not derived from any stored raw time variable. It is synthesised from the bin index of the binned neural trace together with the sampling rate read from `ops['fs']` (30.0 Hz) and the bin width `BIN_FRAMES = 10`. The camera `tstamps.npy` are used only for behaviour alignment, never as the decoder's time input. The single input is named `time_from_session_start_s`.

ii. <Code snippets>

```python
def make_time_input(n_bins: int, fs: float, bin_frames: int) -> np.ndarray:
    return (np.arange(n_bins, dtype=np.float32) * (bin_frames / fs)).astype(np.float32)
```

```python
    fs = float(ops["fs"])
    ...
    time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
```

```python
        "input_names": ["time_from_session_start_s"],
```

iii. <Justification>

Step 5 Key Decision 5: "**Time input meaning**: Interpret 'time elapsed from the beginning of the experiment' as time from the beginning of the recording session, carried through each 2-minute trial as an absolute-within-session time vector." Step 10 Check 3 (input construction): "paper does not define a decoder input variable named time; `convert_data.py` adds elapsed session time because this is explicitly required by the task." Because imaging is at a fixed 30 Hz, the bin index is an exact proxy for elapsed time; `fs` is read from `ops` rather than hard-coded so the mapping is tied to the recording.

---

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. <Decisions>

`t[k] = k * 10 / 30 s` for `k = 0 … n_bins-1`, i.e. the **left edge** of each 333.3 ms bin, in **seconds**, measured from the start of the session and **continuing monotonically across trial boundaries** (it is not reset per trial). Trial *j* therefore spans `[120·j, 120·j + 119.667]` s. Cast to `float32`, reshaped to `(1, 360)` per trial. Observed range `[0.0, 1199.7]` for the 20-min sessions and `[0.0, 1799.7]` for the 30-min sessions (`verification_full_out.txt`), identical to the reference's `[0.0, 1799.667]` global maximum. No normalisation, no binarisation, no one-hot encoding of time.

ii. <Code snippets>

```python
    time_binned_s = make_time_input(neural_binned.shape[1], fs=fs, bin_frames=BIN_FRAMES)
    # -> array([0.0, 0.3333, 0.6667, ..., 1799.6667], dtype=float32)
```

```python
        input_trials.append(time_binned_s[np.newaxis, start:stop].astype(np.float32, copy=False))
```

iii. <Justification>

`CONVERSION_NOTES.md` Step 9 consistency table: "`time_from_session_start_s` range | 0 to session duration after binning | ... | 20-min sessions end at 1199.7 s; 30-min sessions end at 1799.7 s | [0.0, 1799.7] | Yes". Step 10 Check 2 records an independent `np.allclose` re-derivation of the elapsed-time vector for `jm046/2024-09-08_a` trial 11 with max abs diff `0.0`. Keeping absolute (not trial-relative) time is what makes it "time elapsed from the beginning" as the Decoder Input asks.

---

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. <Decisions>

By construction. `make_time_input` is called with `neural_binned.shape[1]`, so the time vector is exactly as long as the binned neural trace and shares its bin grid one-to-one; `split_trials` then truncates and slices `time_binned_s` with the *same* `usable_bins` cut and the same `[start:stop]` indices used for `neural_binned`. The validator additionally asserts `neural[t].shape[1] == input[t].shape[1]` for every trial. No interpolation, shifting, or resampling is involved.

ii. <Code snippets>

```python
    neural_binned = neural_binned[:, :usable_bins]
    time_binned_s = time_binned_s[:usable_bins]
    output_one_hot = output_one_hot[:, :usable_bins]
    ...
        neural_trials.append(neural_binned[:, start:stop]...)
        input_trials.append(time_binned_s[np.newaxis, start:stop]...)
```

```python
        if neural[trial_idx].shape[1] != input_[trial_idx].shape[1]:
            raise ValueError("Input time dimension does not match neural time dimension.")
```

iii. <Justification>

Step 6 implementation notes describe the pipeline as binning both streams then "splits continuous recordings into consecutive 2-minute trials", i.e. one shared index space. Planned sanity Check 3 ("verify converted input time values equal the expected session elapsed time vector sampled every 10 imaging frames") was executed in Step 10 and passed exactly.

---

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. <Decisions>

`move_deve/motion_energy_glob.npy` (the paper's pre-computed global motion-energy trace from the behaviour video, stored as `uint64`) together with `move_deve/tstamps.npy` (per-camera-frame timestamps), which is used purely to place the samples on the imaging-frame grid. `interframe_int.npy` — the file the human reference uses for the same purpose — is not loaded.

ii. <Code snippets>

```python
    motion_raw = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(move_dir / "tstamps.npy", allow_pickle=True)
    ...
    motion_aligned, motion_info = reconstruct_motion_trace(motion_raw, tstamps, F.shape[1])
```

iii. <Justification>

Step 3: "Behavioral variable in the paper is global motion energy computed from consecutive video frames by taking pixel-wise differences, squaring, and summing across pixels" — the `_glob` file is that quantity already computed, so the AI consumes it rather than recomputing (the raw `.avi` videos are not shipped). Step 4: "data README states missing camera frames should be treated as missing or interpolated", and the README names `tstamps.npy` as one of the two ways to find them; Step 5 Key Decision 4 chooses `tstamps.npy`: "Use `tstamps.npy` to map behavior samples onto the imaging-frame grid of length `n_imaging_frames`".

---

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. <Decisions>

Four steps:
1. **Regrid to the imaging clock.** `reconstruct_motion_trace` computes a nominal frame period `frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)`, converts each camera timestamp to an imaging-frame index by rounding, accumulates samples into that grid with `np.add.at` (averaging if two samples land on one index), leaves un-hit indices as NaN, and fills them by `np.interp`.
2. **Bin.** The same 10-frame non-overlapping average used for the neural data.
3. **Normalise.** Min–max scaling of the binned trace to `[0, 1]` within the session.
4. **Discretise + one-hot.** Session-local quintiles, expanded into a 5-row binary matrix (see 4-c).

ii. <Code snippets>

```python
    frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)

    full = np.full(n_imaging_frames, np.nan, dtype=np.float32)
    counts = np.zeros(n_imaging_frames, dtype=np.int64)
    sums = np.zeros(n_imaging_frames, dtype=np.float64)
    np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
    np.add.at(counts, frame_idx, 1)
    valid = counts > 0
    full[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    ...
        missing = np.flatnonzero(~valid)
        if len(missing):
            full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
def normalize_motion(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x)); xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)
```

```python
    motion_binned = average_nonoverlapping(motion_aligned[np.newaxis, :], BIN_FRAMES)[0].astype(np.float32, copy=False)
    motion_binned_norm = normalize_motion(motion_binned)
    output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

iii. <Justification>

Step 5 Variable Mapping: "Reconstruct motion on full imaging-frame grid using timestamps, linearly interpolate missing frames, average in 10-frame bins, normalize within session, discretize into 5 equal-percentile bins, then one-hot encode into 5 binary channels and split into 2-minute trials". Step 5 Key Decision 6: "Normalize motion energy within session after 10-frame averaging, then discretize into five equal-percentile bins within that session". The normalisation step exists because the Decoder Output text the agent received (trajectory step 3) reads "Motion energy, **normalized and** discretized into five equal-percentile bins." Binning precedes discretisation so that class labels are never averaged.

---

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. <Decisions>

Quintile edges are computed **within each session** on the binned, normalised trace using `np.quantile(x, [0.2, 0.4, 0.6, 0.8])`, and each timepoint is assigned a class in `0…4` with `np.searchsorted(edges, x, side="right")`. This gives exactly 20% of timepoints per class in every session (`verification_full_out.txt` reports 0.200 positive fraction for each channel in all 41 sessions), and edges are re-derived per session so an overall shift in a mouse's activity level does not bias labels.

The AI then departs from the reference in **representation**: instead of exporting one categorical output of dimension 1 with 5 possible values, it expands the labels to a **5-row one-hot matrix** and declares five separate binary outputs, `motion_energy_q0 … motion_energy_q4`, each with `output_values = ["not_qi", "qi"]`. The exported dataset therefore has `doutput = 5` with chance = 0.5 per output, whereas the reference has `doutput = 1` with 5 classes and chance = 0.2.

ii. <Code snippets>

```python
N_OUTPUT_BINS = 5

def quintile_one_hot(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    classes = np.searchsorted(edges, x, side="right").astype(np.int64)
    one_hot = np.eye(N_OUTPUT_BINS, dtype=np.int64)[classes].T
    return one_hot, classes, edges
```

```python
        "output_names": [f"motion_energy_q{i}" for i in range(N_OUTPUT_BINS)],
        "output_values": [[f"not_q{i}", f"q{i}"] for i in range(N_OUTPUT_BINS)],
```

```python
        if output[trial_idx].shape[0] != N_OUTPUT_BINS:
            raise ValueError("Expected one-hot output with five rows.")
        if not np.all(output[trial_idx].sum(axis=0) == 1):
            raise ValueError("Each output timepoint must belong to exactly one motion quintile.")
```

iii. <Justification>

For the per-session quintiles: Step 5 Key Decision 6 and Step 4's last row, "User instructions explicitly require discretized 5-bin output ... Keep paper-like preprocessing and alignment, but discretize the final motion-energy target into 5 equal-percentile bins for exported `output`."

For the one-hot expansion, trajectory step 80: "I've got the validator behavior now: multi-class outputs need to be one-hot encoded for this decoder code, otherwise quintile labels would collapse incorrectly." This came from reading `decoder.py` lines 1436–1665 (trajectory step 79), where a *different* decoder function documents "output ... of dtype = bool ... Use a one-hot encoding for multi-class outputs." The function actually invoked by `train_decoder.py`, `train_decoder()` at `decoder.py:803`, documents the opposite: "output ... of dtype = int with categorical values for each timepoint. Each output[idx, t] is an integer from 0 to ncategories[idx]-1", and infers `ncategories` from the data. So the stated justification rests on a misread docstring.

---

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. <Decisions>

Imaging and videography were hardware-synchronised at 30 Hz, so alignment is one sample per imaging frame. Rather than assuming index identity, the AI re-derives the mapping from `tstamps.npy`: it divides each timestamp by the mean frame period and rounds to get an imaging-frame index, averages any colliding samples, and linearly interpolates any grid slot that receives no sample. After that the motion trace has exactly `F.shape[1]` entries, is binned with the identical `average_nonoverlapping` call as the neural data, and is sliced with the identical trial indices. The validator asserts matching time dimensions and no NaNs.

Verification I ran independently:
* On the 9 sessions that genuinely have dropped camera frames, the AI's result is **exactly** the reference's `interframe_int`-based insertion. On `jm031/2023-10-22_a` (116 dropped frames) the two full-length traces agree with `max abs diff = 0.0`.
* However, on 3 sessions where `len(motion_energy) == n_imaging_frames` (nothing is missing, so identity is trivially correct), the rounding re-grid still moves samples: `jm046/2024-09-05_a` (20 slots re-filled, shifts ≤ 2 frames), `jm046/2024-09-08_a` (36 slots, shifts ≤ 2 frames), and `jm046/2024-09-07_a` (33 slots, cumulative shift up to **10 frames = 0.33 s = one full time bin**). For that last session the binned trace correlates only r = 0.71 with the identity-aligned trace and **only 65% of quintile labels agree**. The AI logged these as `missing_motion_frames=20/33/36` and recorded them in `CONVERSION_NOTES.md` Step 10 as genuine "sessions with missing motion frames", so the artefact went unnoticed.

ii. <Code snippets>

```python
    frame_dt = (tstamps[-1] - tstamps[0]) / (n_imaging_frames - 1)
    if not np.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError(f"Invalid motion timestamp scale: frame_dt={frame_dt}")

    frame_idx = np.round((tstamps - tstamps[0]) / frame_dt).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, n_imaging_frames - 1)
```

```python
    info = {
        "n_motion_frames_raw": int(len(motion_energy)),
        "n_imaging_frames": int(n_imaging_frames),
        "n_missing_motion_frames": int((~valid).sum()),
        "motion_frame_dt": float(frame_dt),
    }
```

```python
        if neural[trial_idx].shape[1] != output[trial_idx].shape[1]:
            raise ValueError("Output time dimension does not match neural time dimension.")
```

iii. <Justification>

Step 3: "Behavior: videography at 30 Hz, triggered by microscope acquisition for simple synchronization with imaging." Step 4: "Temporal alignment | Code package itself does not align behavior; paper relies on synchronized 30 Hz acquisition | 9 sessions have missing behavior frames relative to imaging | Data README says missing camera frames should be treated as missing or interpolated | Align behavior to imaging frame index, explicitly detect missing behavior frames, and interpolate only where needed." Step 10 Check 2 reports an `np.allclose` spot-check of the full chain for `jm031/2023-10-22_a` (116 interpolated frames) against a fresh re-derivation — but that re-derivation used the same timestamp-regridding logic, so it could not detect the regridding artefact described above.

---

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

* **Dropped camera frames** (9/41 sessions, 1–148 frames): the grid slot is left NaN and filled by `np.interp` over the surrounding valid samples, so the behaviour trace always reaches full imaging length.
* **Degenerate inputs** raise rather than silently pass: non-1D motion arrays, mismatched `motion_energy`/`tstamps` lengths, `< 2` imaging frames, non-finite or non-positive `frame_dt`, zero valid motion frames, a session too short to bin, and a session too short for one trial all raise `ValueError`.
* **Constant motion trace**: `normalize_motion` returns zeros instead of dividing by zero; a single valid motion sample broadcasts to the whole trace.
* **Sub-trial remainders**: frames left over after 10-frame binning and after 120-s trial splitting are truncated (both are 0 for every session in this dataset).
* **Session duration heterogeneity** (36,000 vs 54,000 frames) is handled by deriving trial count per session rather than assuming a constant.
* **Post-hoc guard**: `validate_converted_session` runs on every session before it is accepted and rejects NaNs, shape mismatches, `< 2` trials, and non-one-hot outputs.

Not handled: the spurious re-gridding on the 3 no-drop sessions described in 4-d.

ii. <Code snippets>

```python
    if motion_energy.ndim != 1 or tstamps.ndim != 1:
        raise ValueError("Motion-energy inputs must be 1D arrays.")
    if len(motion_energy) != len(tstamps):
        raise ValueError("motion_energy and tstamps must have the same length.")
    if n_imaging_frames < 2:
        raise ValueError("Expected at least 2 imaging frames.")
    ...
    if not np.any(valid):
        raise ValueError("No valid motion frames after alignment.")

    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 1:
        full[:] = full[valid_idx[0]]
    else:
        missing = np.flatnonzero(~valid)
        if len(missing):
            full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
    usable = (n_frames // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError(f"Not enough frames to bin: got {n_frames}, need at least {bin_frames}")
```

```python
        if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
            raise ValueError("Converted arrays must not contain NaN values.")
```

iii. <Justification>

Step 4/Step 5 Key Decision 4 cite `data/README.md`: "In some recordings there might be some missing frames from the camera ... The indices of missing frames can be obtained by looking at `tstamps.npy` or `interframe_int.npy` and treated as missing values for motion energy or they can be interpolated over." The AI chose interpolation over masking because the decoder format requires a value at every timepoint. Step 10 Check 5 (edge cases): "Verified sessions with missing motion frames (1, 2, 3, 20, 33, 36, 116, 148) convert without NaNs" and "Verified both 20-minute and 30-minute sessions produce the expected 10 or 15 trials respectively."

---

## 6-a. What are the most time-consuming steps of the code?

i. <Decisions>

The AI instrumented per-session wall time (`processing_seconds` in each summary) and printed a full-run estimate. Full conversion of 41 sessions took **57.35 s**, mean **1.38 s/session** — well inside the 15-minute budget, so no optimisation was undertaken.

Breaking a representative 30-minute session (`jm046/2024-09-09_a`, 435 × 54,000) down by hand: `dcnv.preprocess` 0.76 s (≈ 58%), `np.load` of `F` + `Fneu` 0.18 s, neuropil subtraction 0.04 s, 10-frame binning 0.04 s. So the suite2p baseline correction dominates, followed by file I/O. Two further costs sit outside the per-session timer: the final `pickle.dump` of the ~420 MB dataset, and — when `--show-processing` is on — matplotlib rendering, which roughly doubled per-session time in the sample run (2.26 s/session with plots vs 1.38 s without).

One self-inflicted cost: `compute_fluorescence_signal` hard-codes `device=torch.device("cpu")`, so `dcnv.preprocess` never uses the GPU even though the container had CUDA available (`torch 2.6.0+cu124`, and the decoder ran on GPU).

ii. <Code snippets>

```python
def process_session(session: SessionRef, show_processing: bool = False) -> tuple[dict, dict]:
    start_time = time.time()
    ...
    elapsed = time.time() - start_time
    summary = { ..., "processing_seconds": float(elapsed) }
```

```python
        batch_size=min(512, max(32, Fc.shape[0])),
        device=torch.device("cpu"),
```

```python
    if session_summaries:
        mean_time = np.mean([summary["processing_seconds"] for summary in session_summaries])
        est_full = mean_time * 41
        print(f"Mean processing time per session: {mean_time:.2f}s")
        print(f"Estimated full-dataset time at this rate: {est_full / 60:.2f} minutes")
```

iii. <Justification>

Step 7 Run Time Estimates: "Sample conversion 2.26 s/session → 1.54 min for 41 sessions ... No optimization required before full conversion", and Step 6: "Need to benchmark Suite2p-style fluorescence preprocessing on sample data before deciding whether further optimization is required." The CPU pin is not discussed in the notes; the AI simply passed `torch.device("cpu")` to `suite2p_preprocess`. Given the measured 57 s total, the omission has no practical consequence here.

---

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

The AI explicitly vectorised the two places that would naturally be written as loops:
* **Motion-frame reconstruction**: scatter with `np.add.at` and gap-filling with a single `np.interp` call, instead of inserting frames one at a time. (The human reference does use a per-dropped-frame `np.insert` loop, which reallocates the 36,000-element array 116 times on `jm031/2023-10-22_a`; the AI's version avoids this.)
* **Bin averaging**: a single `reshape(...).mean(axis=-1)` for both streams.

What remains loop-shaped:
* `for start in range(0, usable_bins, bins_per_trial)` in `split_trials` — but the body is pure slicing, and the output format is an explicit *list* of per-trial arrays, so a vectorised reshape would have to be unpacked back into a list anyway. No meaningful gain.
* The per-session loop in `main()` — genuinely parallelisable (each session is independent and the dominant cost, `dcnv.preprocess`, is CPU-bound), which would cut the 57 s wall time by roughly the core count. The AI decided against it on measured-time grounds.
* The per-trial loop in `validate_converted_session` (545 trials × a handful of `np.isnan`/`sum` passes over full arrays). This re-scans the entire dataset and could be done once per session on the un-split arrays.
* `for edge in motion_edges: axhline(...)` in the plotting helper — 4 iterations, irrelevant.

ii. <Code snippets>

```python
    np.add.at(sums, frame_idx, motion_energy.astype(np.float64, copy=False))
    np.add.at(counts, frame_idx, 1)
    ...
            full[missing] = np.interp(missing, valid_idx, full[valid_idx]).astype(np.float32)
```

```python
    new_shape = x.shape[:-1] + (usable // bin_frames, bin_frames)
    return x.reshape(new_shape).mean(axis=-1)
```

```python
    for idx, session in enumerate(sessions, start=1):        # serial; independent per session
        converted, summary = process_session(session, show_processing=do_plot)
        validate_converted_session(converted)
```

iii. <Justification>

Step 6 "Code speedups added": "Vectorized motion-frame reconstruction using `np.add.at` and `np.interp`. Vectorized non-overlapping bin averaging via reshape/mean." Step 7: "Vectorized motion reconstruction + reshape-based binning | Full conversion estimated at 1.54 min; no further optimization needed." The instructions only required optimisation if the estimate exceeded 15 minutes, so leaving the session loop serial is a deliberate, stated trade-off.

---

## 6-c. What processing does the code repeat multiple times?

i. <Decisions>

Very little genuine recomputation. Each session's files are read exactly once, and each processing stage is applied once. The repeats that do exist are trivial:

* Redundant `.astype(np.float32, copy=False)` calls on arrays already `float32` — inside `compute_fluorescence_signal`, again on its return value, again on `neural_binned`, again on every trial slice in `split_trials`. These are no-ops (`copy=False`) but they clutter the flow.
* `validate_converted_session` re-traverses every trial of every session (NaN scan, one-hot sum) after the arrays were already checked for shape consistency by construction — an extra full pass over the ~370 MB of neural data.
* `motion_classes` is returned alongside `output_one_hot` even though it is recoverable from it by `argmax`; both are carried through `process_session`.
* Across the whole workflow (not the script itself) the pipeline was executed several times — `--sample` twice (the sample selection was changed from the first two to the last two sessions), the full run, plus five auxiliary subset pickles now in `/app/cache/` (`early_subset.pkl`, `late_subset.pkl`, `jm038/39/40/46_last2.pkl`) generated for the early-vs-late analysis in Step 12.

ii. <Code snippets>

```python
    Fc = F.astype(np.float32, copy=False) - float(ops["neucoeff"]) * Fneu.astype(np.float32, copy=False)
    ...
    return processed.astype(np.float32, copy=False)
```

```python
    neural_binned = average_nonoverlapping(neural_processed, BIN_FRAMES).astype(np.float32, copy=False)
    ...
        neural_trials.append(neural_binned[:, start:stop].astype(np.float32, copy=False))
```

```python
    for trial_idx in range(n_trials):
        ...
        if np.isnan(neural[trial_idx]).any() or np.isnan(input_[trial_idx]).any() or np.isnan(output[trial_idx]).any():
```

iii. <Justification>

Not discussed in `CONVERSION_NOTES.md` — the AI's efficiency discussion (Steps 6–7) is confined to the two vectorisations and the runtime estimate, and it concluded no further work was needed once the full run came in at under one minute. The `copy=False` guards and the extra validation pass are defensive-programming choices consistent with the instructions' "Validate data shapes and types at each step" and "Include sanity checks".

---

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

Several small items, none of them expensive:

* **`normalize_motion` is a no-op for the labels.** Min–max scaling is strictly monotonic, so `np.quantile`/`searchsorted` produce identical quintile assignments with or without it. It costs two full passes over the binned trace and, more importantly, discards the physical units of motion energy from everything that is exported. (It is done because the agent's Decoder Output text said "normalized and discretized".)
* **Dead computation in the plotting helper**: `vmax` is computed with `np.percentile(np.abs(trial_neural), 99)` over a full 435 × 360 trial and then never passed to `imshow`.
* **`ops.npy` is unpickled in full** (it includes `meanImg` and the rest of the suite2p option dict) to read six scalars.
* **`motion_classes`, `motion_edges` and the whole `motion_info` dict** are computed for every session but consumed only by the optional plots and the metadata summary.
* **`neural_processed`** (the full 30 Hz array, e.g. 435 × 54,000 float32 ≈ 94 MB) is kept alive for the whole session even though only `neural_binned` is exported; with `--show-processing` off it could be freed immediately after binning.
* **The one-hot expansion** stores five `int64` rows per trial where one small-int row would carry the same information (~7.8 MB of the 420 MB pickle — negligible next to the neural data, but 40× the necessary output bytes).
* **`session_info`**, a 41-entry list of per-session dicts including `motion_quintile_edges` and `processing_seconds`, is embedded in the exported metadata and never read by the decoder.

ii. <Code snippets>

```python
    motion_binned_norm = normalize_motion(motion_binned)      # monotonic -> no effect on quintiles
    output_one_hot, motion_classes, motion_edges = quintile_one_hot(motion_binned_norm)
```

```python
    vmax = np.percentile(np.abs(trial_neural), 99)
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
    ax[2, 0].imshow(trial_neural[: min(80, trial_neural.shape[0])], aspect="auto", cmap="viridis")   # vmax unused
```

```python
        "motion_quintile_edges": motion_edges.tolist(),
        "processing_seconds": float(elapsed),
    ...
            "session_ids": [f"{session.subject}/{session.session_id}" for session in session_refs],
            "session_info": [session["summary"] for session in converted_sessions],
```

iii. <Justification>

The normalisation is justified in Step 5 Key Decision 6 ("Normalize motion energy within session after 10-frame averaging, then discretize into five equal-percentile bins within that session") and by the agent's Decoder Output wording, though the notes never observe that it cannot change the labels. The rich `session_info`/`session_ids` metadata is justified by the target-format instruction to "Add other relevant fields, e.g. `session_info`". The dead `vmax` and the retained full-rate array are not discussed anywhere — they appear to be oversights rather than decisions.
