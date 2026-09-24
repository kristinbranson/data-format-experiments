# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the six subject IDs (`jm031, jm032, jm038, jm039, jm040, jm046`) and discovers sessions by listing the sorted, non-hidden subdirectories of each subject folder. For each session it loads four raw arrays directly with `np.load`: `suite2p/plane0/F.npy` (raw fluorescence), `suite2p/plane0/Fneu.npy` (neuropil fluorescence), `move_deve/motion_energy_glob.npy` (motion energy) and `move_deve/tstamps.npy` (camera timestamps). Nothing else (`spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`, `interframe_int.npy`) is read by the conversion path; `ops.npy` was inspected interactively to confirm `fs=30`, `neucoeff=0.7`, `baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`. All 41 sessions (7/7/7/7/6/7) are loaded in `--full` mode; `--sample` mode restricts to the first subject's first 2 sessions. There is no trial structure in the raw data, so trials are created later by segmenting the continuous recording.

ii.
```python
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']

def get_sessions(subject_dir):
    """Get sorted list of session directories for a subject."""
    sessions = []
    for d in sorted(os.listdir(subject_dir)):
        full = os.path.join(subject_dir, d)
        if os.path.isdir(full) and not d.startswith('.'):
            sessions.append(full)
    return sessions

def process_session(session_dir, device=None):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    me_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
```

iii. From CONVERSION_NOTES Step 1/2: the dataset README states that subject folders are named `jm*` (mouse A–F in alphabetical order) and that each session subfolder holds Track2p output in suite2p format; the reference `load_data.ipynb` also loads `F.npy` from `suite2p/plane0`. The AI documented the full file inventory per session and concluded that `F.npy`/`Fneu.npy` plus `motion_energy_glob.npy` are the only streams needed for the decoder task, with `tstamps.npy`/`interframe_int.npy` available for the documented camera frame drops.

## 1-b. How are the data split into subjects?

i. Each of the six hard-coded `jm*` directory names is one mouse. The subject list is stored verbatim as `data['subjects']`, and `subject_idx` is the enumeration index of the subject loop, so it is in alphabetical order (jm031→mouse A … jm046→mouse F). In `--sample` mode `data['subjects']` is truncated to `SUBJECTS[:1]` so the index stays valid.

ii.
```python
SUBJECTS = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for subj_idx, subject in enumerate(subjects_to_process):
    subject_dir = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_dir):
        print(f"Warning: Subject directory not found: {subject_dir}")
        continue
    ...
    subject_idx_list.append(subj_idx)
...
'subjects': SUBJECTS if not sample else SUBJECTS[:1],
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 records the six subject folders and their neuron counts, and cites the data README: "the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A, jm032 - mouse B … jm046 - mouse F)". The AI listed them explicitly to preserve that paper-matching order.

## 1-c. How are the data split into sessions?

i. One session = one dated subdirectory (`YYYY-MM-DD_a`) inside a subject folder, i.e. one recording day. Directories are sorted by name (chronological), hidden entries and non-directories are skipped. Each session becomes one entry of the `neural`/`input`/`output` session lists, giving 41 sessions total. Sessions are 36 000 frames (20 min, jm031/jm032) or 54 000 frames (30 min, the other four mice); both are kept as-is.

ii.
```python
sessions = get_sessions(subject_dir)      # sorted dated subfolders
if sample:
    sessions = sessions[:2]
for sess_dir in sessions:
    dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)
    session_data.append((subj_idx, sess_dir, dff_binned, me_binned, n_neurons))
```

iii. CONVERSION_NOTES Step 2/4: "Each subject folder contains a number of session folders, each corresponding to one recording day." Step 4 flags the 20- vs 30-minute discrepancy with the methods ("each session lasted 20 minutes") and resolves it as "jm038–jm046 have 30-min sessions; paper says 20 min but some mice recorded longer. Use all available data."

## 1-d. How are the data split into trials?

i. There is no trial structure in the continuous recording, so trials are defined as **non-overlapping 120-second (2-minute) blocks** of each session: `TRIAL_DURATION_S = 120.0`, i.e. 120 s × 30 Hz / 10 frames-per-bin = **360 bins per trial**. Splitting happens after binning. 20-min sessions yield 10 trials, 30-min sessions yield 15 trials, for 545 trials total. Any trailing bins that do not fill a whole trial are silently dropped by integer division (in practice zero, since 36 000/54 000 frames divide exactly by 3 600).

ii.
```python
TRIAL_DURATION_S = 120.0  # 2 minutes per trial (paper: "consecutive 2 minute blocks")

def split_into_trials(dff_binned, me_binned):
    bins_per_trial = int(TRIAL_DURATION_S * FS / BIN_SIZE)  # 360
    n_bins = dff_binned.shape[1]
    n_trials = n_bins // bins_per_trial
    neural_trials, me_trials = [], []
    for t in range(n_trials):
        start = t * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(dff_binned[:, start:end].astype(np.float32))
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "**Trial splitting**: 2-minute blocks (paper uses 'consecutive 2 minute blocks' for CV)", quoting the methods sentence "We used 5 fold splits for both the inner and outer loops, splits were done on consecutive 2 minute blocks of the recording." The AI therefore took the paper's cross-validation block length as the natural trial length. Note that the prompt recorded in the trajectory (step 1) did **not** contain the "Split sessions into 60-second trials" sentence that appears in `/tests/instruction_reference.md`, which is why the AI fell back on the paper.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. Every complete 2-minute block of every session is kept; no trials, sessions or mice are excluded. The only data loss is the (empty in practice) trailing remainder of each session.

ii.
```python
n_trials = n_bins // bins_per_trial      # remainder bins simply not emitted
# no other filtering anywhere in the script
```

iii. CONVERSION_NOTES Step 3 "Curation Steps": "**Trial curation**: No explicit trial curation mentioned - continuous recording." Step 10 Check 5 notes "Partial bins at end of sessions: discarded (consistent with integer division)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the two suite2p traces in `plane0`: `F.npy` (n_neurons × n_frames raw fluorescence) and `Fneu.npy` (neuropil fluorescence). Deconvolved `spks.npy` is deliberately not used. Neuron counts are 221/370/685/746/541/435 per mouse (2 998 unique tracked neurons, 20 445 neuron-sessions).

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
dff = compute_dff(F, Fneu, device=device)
```

iii. CONVERSION_NOTES Step 3/5: the methods say "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses", so fluorescence rather than deconvolved spikes is the paper's decoding substrate; `Fneu` is required for the suite2p neuropil correction with `neucoeff = 0.7` read from `ops.npy`.

## 2-b. How is the `neural` data processed?

i. Three steps: (1) neuropil subtraction `F_corr = F − 0.7·Fneu`; (2) suite2p's `preprocess()` (the `dcnv` maximin baseline routine, `win_baseline = 60 s`, `sig_baseline = 10`, `fs = 30`, `prctile_baseline = 8`, GPU when available) which returns the baseline-subtracted trace; (3) the AI then recovers the baseline itself (`baseline = F_corr − F_subtracted`) and divides, giving a ratiometric `dF/F = (F_corr − F0)/F0`, with the baseline clipped at `1e-6` before division. The result is cast to float32 and later averaged in 10-frame bins. No z-scoring or per-neuron normalisation is applied.

ii.
```python
F_corr = F - NEUCOEFF * Fneu
F_subtracted = preprocess(
    F_corr_copy, BASELINE, WIN_BASELINE, SIG_BASELINE, FS,
    prctile_baseline=PRCTILE_BASELINE, device=device
)
# F_subtracted = F_corr - baseline => baseline = F_corr - F_subtracted
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
return dff.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1 ("Use Suite2p's preprocess (baseline_maximin) with default params from ops.npy") and Step 10 Check 3, where the AI lists "dF/F | (F_corr - baseline) / baseline | 'baseline corrected fluorescence traces as our dF/F' | YES". Its stated rationale is that the paper's phrase plus suite2p defaults defines the pipeline, and that dividing by F0 gives the conventional dF/F units. (Verification of the saved file shows this division is not benign: because the maximin baseline of the neuropil-subtracted trace can be ≤ 0, the `1e-6` clip yields values up to 4.6 × 10⁸ and down to −9.1 × 10⁷ in 40 of 41 sessions, affecting 497 of 20 445 neuron-sessions; this is also why the reported training loss starts at 1.5 × 10⁶.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is performed. All rows of `F.npy` are kept, on the grounds that the released data already contains only the Track2p-tracked, `iscell`-thresholded cells. The AI verified that `iscell[:, 0]` is uniformly 1.0 in the released files. No neurons, sessions or mice are dropped for quality.

ii.
```python
# no iscell / SNR / activity filtering anywhere; every row of F.npy is kept
n_neurons, n_frames = F.shape
...
brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. CONVERSION_NOTES Step 1: "All neurons are already filtered to cells tracked across ALL days for each mouse; iscell.npy has all values = 1.0 (all cells marked as cells since they're pre-filtered)". Step 4 resolves the apparent discrepancy with the methods' "default threshold of 0.5": "Provided data already pre-filtered to tracked cells; no additional filtering needed". Step 9 cross-checks the resulting counts against the paper's "526 ± 190 neurons per mouse" (data gives 500 ± 198).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus event; trials are contiguous blocks measured from the start of the recording, so alignment is simply "session start plus k × 120 s". Metadata records `temporal_alignment_event = 'Start of recording session'`, `off_start = 0.0`, `off_end = None`. Neural, input and output for a trial are all taken from the same bin index range `[t·360, (t+1)·360)`, so the three streams are aligned bin-for-bin.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': None,
    'trial_duration_s': TRIAL_DURATION_S,
    ...
}
...
neural_trials.append(dff_binned[:, start:end].astype(np.float32))
me_trials.append(me_binned[start:end])
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "**Temporal alignment**: ME is already synced to neural (camera triggered by microscope)", and Step 3 processing note 5: "Camera sync: Video at 30 Hz triggered by microscope acquisition → 1:1 frame correspondence". Because the recording is continuous and un-cued, the AI treats the beginning of the recording as the only meaningful alignment point. (`off_end = None` with `off_start = 0.0` is internally inconsistent with a 120 s trial, but has no effect on the data arrays.)

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Raw acquisition is 30 Hz (33.3 ms/frame). Both the dF/F traces and the motion energy trace are re-binned by averaging **10 consecutive frames**, giving 3 Hz, i.e. a **333.33 ms** bin, written to `metadata['time_bin_size']`. Binning is done once per session on the full continuous trace, before trial splitting and before motion-energy discretization; a trailing partial bin would be dropped. Every trial is exactly 360 bins, identical across all trials and sessions.

ii.
```python
BIN_SIZE = 10   # paper: "bins of 10 consecutive timestamps"

def bin_traces(data, bin_size):
    if data.ndim == 1:
        n_bins = len(data) // bin_size
        trimmed = data[:n_bins * bin_size]
        return trimmed.reshape(n_bins, bin_size).mean(axis=1)
    else:
        n_features, n_frames = data.shape
        n_bins = n_frames // bin_size
        trimmed = data[:, :n_bins * bin_size]
        return trimmed.reshape(n_features, n_bins, bin_size).mean(axis=2)
...
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned = bin_traces(me_interp, BIN_SIZE)
...
time_bin_ms = BIN_SIZE / FS * 1000   # 333.33 ms
```

iii. CONVERSION_NOTES Step 3/5 quote the methods directly: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." The AI applies the same binning to both streams so that they remain the same length and so that discretization operates on the denoised trace rather than on labels.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. No raw variable is used. The single input channel is synthesised from the within-trial bin index and the known frame rate: `time = arange(n_timebins) · (BIN_SIZE / FS)`. Crucially the counter is **reset at the start of every trial**, so the input is the time since the start of that 2-minute block, not since the start of the session/experiment. Every trial in the dataset therefore carries the identical vector 0.0 … 119.667 s (confirmed in `verification_full_out.txt`: input range `[0.0, 119.7]` for all 41 sessions). It is named `time_elapsed_s`.

ii.
```python
def make_time_input(n_timebins):
    """
    Task: "Time elapsed from the beginning of the experiment. Time-varying."
    Time in seconds from start of trial.
    """
    time_bin_s = BIN_SIZE / FS  # seconds per bin
    return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
...
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Time index → input[0] → Time in seconds from start of trial — (bin_index * 10 / 30) seconds within trial". The justification given is that the frame rate is exactly 30 Hz so elapsed time can be computed arithmetically; no rationale is given for restarting the clock each trial, and the function's own docstring still quotes the task text "time elapsed from the beginning of the experiment".

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Only `arange(360) × 0.3333` cast to float32 and reshaped to `(1, 360)`. No offsetting by the trial's position within the session, no session-level or global offset, no normalisation. As a consequence the input is constant across trials and carries no information that distinguishes one trial or one session from another; the only information it supplies to the decoder is position within the 2-minute block. `input_names = ['time_elapsed_s']`.

ii.
```python
time_bin_s = BIN_SIZE / FS  # 0.3333 s
return (np.arange(n_timebins) * time_bin_s).astype(np.float32).reshape(1, -1)
# note: no `+ trial_start_seconds` term anywhere in the script
```

iii. CONVERSION_NOTES Step 7 lists "Input range | [0.0, 119.7] seconds" as an expected sample statistic and Step 10 Check 2 records the sanity check "Input check: Time values match expected (arange * 10/30)" — i.e. the AI checked its implementation against its own trial-relative formula rather than against the task's "time from the beginning of the experiment/session" definition, so the mismatch was never caught.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time vector is built with exactly `neural_t.shape[1]` entries for the trial it accompanies, and bin *k* of the input corresponds to bin *k* of the neural matrix, which corresponds to frames `[10k, 10k+10)` of that trial. So within a trial the input and neural arrays are aligned element-for-element and have identical shape in the time axis. (The value assigned to bin *k* is the left edge of the bin.)

ii.
```python
for neural_t, me_t in zip(neural_trials, me_trials):
    n_timebins = neural_t.shape[1]          # 360, taken from the neural array itself
    time_input = make_time_input(n_timebins)
    input_trials.append(time_input)         # (1, 360), same time axis as neural (n_neurons, 360)
```

iii. Implicit in the design: the input is generated from the neural trial's own length, so no resampling or interpolation is needed and no misalignment is possible. CONVERSION_NOTES Step 7 confirms "Time bins per trial | 360" for both streams.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `output` comes from `move_deve/motion_energy_glob.npy` — the pre-computed global motion energy of the behavioural video, one value per camera frame (uint64). `move_deve/tstamps.npy` is also loaded and passed into the frame-mismatch handler, but it is never actually read inside that function. `interframe_int.npy`, the array the dataset README points to for locating dropped frames, is not used by the conversion script.

ii.
```python
me = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))
...
me_interp = interpolate_motion_energy(me, n_frames, tstamps)   # tstamps unused inside
```

iii. CONVERSION_NOTES Step 2 documents the `move_deve` folder contents, Step 3 records that motion energy is "Pixel-wise difference of consecutive video frames, squared and summed across pixels", and Step 5 Key Decision 4 says "Missing ME frames: Interpolate to match neural frame count using timestamps" — the intent was to use `tstamps.npy`, which the code ultimately does not do.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. (1) If the motion energy array is shorter than the neural recording, it is **uniformly resampled** onto the neural frame grid with `np.interp` using evenly spaced source positions (`np.linspace(0, n_neural−1, n_me)`); if it is longer it is truncated; if equal it is passed through unchanged. (2) It is then averaged into the same 10-frame bins as the neural data. (3) The binned values are discretized into 5 levels using percentile edges computed over the concatenation of **all sessions of all mice**. No per-session or per-mouse normalisation/standardisation of the raw motion energy is performed despite "normalized" appearing in the task text and in the notes' mapping table.

ii.
```python
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    n_me = len(me)
    if n_me == n_neural_frames:
        return me.astype(np.float64)
    if n_me < n_neural_frames:
        neural_indices = np.arange(n_neural_frames)
        me_indices = np.linspace(0, n_neural_frames - 1, n_me)
        me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
        return me_interp
    else:
        return me[:n_neural_frames].astype(np.float64)
...
me_binned = bin_traces(me_interp, BIN_SIZE)
```

iii. CONVERSION_NOTES Step 3 quotes the README: "In some recordings there might be some missing frames from the camera … they can be interpolated over". Step 5 Key Decision 4 and Step 10 Check 5 ("Missing ME frames: handled via interpolation (affects 7 sessions)") record interpolation as the chosen policy; binning is justified by the same methods sentence as the neural binning.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five **global** equal-percentile bins. All binned motion-energy traces from all 41 sessions are concatenated, `np.percentile(all, [0, 25, 50, 75, 100]/…)` i.e. `linspace(0, 100, 6)` gives 6 edges, and every session's trace is digitized with those same edges (`np.digitize(me, edges[1:-1])`, clipped to 0…4). Edges are saved in metadata (`[5.3e5, 7.2e5, 8.2e5, 1.02e6, 1.64e6, 3.92e7]`) and used to build human-readable `output_values` labels. Globally the five classes are exactly 20 % each, but per session the distribution is strongly skewed: e.g. session 0 has no level-0 bins, several sessions put 72–79 % of their time in a single level, and the last seven sessions (jm046) contain only levels 2–4 (one session only levels 3 and 4).

ii.
```python
def compute_me_percentile_bins(all_me_values, n_bins=N_OUTPUT_BINS):
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(all_me_values, percentiles)

def discretize_me(me_values, bin_edges):
    n_bins = len(bin_edges) - 1
    labels = np.digitize(me_values, bin_edges[1:-1])   # 0 .. n_bins-1
    labels = np.clip(labels, 0, n_bins - 1)
    return labels.astype(np.int64)
...
all_me_concat = np.concatenate(all_me_binned_values)   # every session, every mouse
bin_edges = compute_me_percentile_bins(all_me_concat, N_OUTPUT_BINS)
...
me_disc = discretize_me(me_t, bin_edges)
output_trials.append(me_disc.reshape(1, -1))
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "**Motion energy normalization**: Compute percentile bins globally across all sessions", and Step 9 reports "Global output distribution | 20%/20%/20%/20%/20% | YES" as evidence the discretization is correct. The AI's stated reasoning is that a single global set of edges makes the five classes exactly equal-percentile over the whole dataset and keeps the class labels comparable across sessions; it did not examine or report the per-session distributions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Video and 2-photon acquisition are treated as frame-synchronous (the camera is triggered by the microscope), so motion energy frame *i* is assumed to correspond to imaging frame *i*. When the motion energy array is short (9 of 41 sessions: 1–3 missing frames in 7 sessions, 116 in jm031/2023-10-22 and 148 in jm032/2023-10-22), the AI stretches the whole trace uniformly onto the neural grid rather than inserting values at the specific dropped-frame indices recoverable from `interframe_int.npy`/`tstamps.npy`. After that, both streams are binned identically and sliced with the same bin indices, so shapes always match; there is no assertion or warning about the mismatch.

ii.
```python
me_interp = interpolate_motion_energy(me, n_frames, tstamps)   # n_frames from F.npy
...
me_indices = np.linspace(0, n_neural_frames - 1, n_me)         # uniform stretch
me_interp = np.interp(neural_indices, me_indices, me.astype(np.float64))
...
dff_binned = bin_traces(dff, BIN_SIZE)
me_binned  = bin_traces(me_interp, BIN_SIZE)
...
neural_trials.append(dff_binned[:, start:end])
me_trials.append(me_binned[start:end])
```

iii. CONVERSION_NOTES Step 3 processing note 5 ("Camera sync: Video at 30 Hz triggered by microscope acquisition → 1:1 frame correspondence") and Step 5 Key Decision 6 ("ME is already synced to neural"). The README's statement that dropped frames "can be … interpolated over" is cited as license for the interpolation; the AI did not verify that the interpolated trace places the drops at the correct times (the uniform stretch spreads the correction evenly, so in the two worst sessions the motion-energy trace is locally shifted relative to the neural trace by up to ~100–148 frames ≈ 3–5 s).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Four edge cases are handled, with varying success. (a) Missing camera frames: uniform resampling as above (no assertion, no error if the counts are wildly different). (b) Motion energy longer than the neural recording: silently truncated. (c) Trailing frames/bins that do not fill a whole 10-frame bin or a whole 120 s trial: silently dropped by integer division. (d) Non-positive maximin baselines during the dF/F division: clipped to `1e-6`. A missing subject directory would print a warning and be skipped (but would then shift `subject_idx` relative to `data['subjects']`). Case (d) is not really handled — it converts a division-by-zero into values up to 4.6 × 10⁸, present in 40 of 41 sessions and 497 of 20 445 neuron-sessions in the saved file. The AI's independent sanity check (Step 10 Check 2) was run only on jm031 session 0, the one session that happens to be unaffected, so the artefact went unreported.

ii.
```python
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
...
else:
    # ME has more frames than neural (shouldn't happen, but truncate)
    return me[:n_neural_frames].astype(np.float64)
...
n_bins = n_frames // bin_size          # trailing partial bin dropped
n_trials = n_bins // bins_per_trial    # trailing partial trial dropped
...
if not os.path.isdir(subject_dir):
    print(f"Warning: Subject directory not found: {subject_dir}")
    continue
```

iii. CONVERSION_NOTES Step 10 Check 5 "Edge cases": "Missing ME frames: handled via interpolation (affects 7 sessions); Partial bins at end of sessions: discarded (consistent with integer division); 30-min vs 20-min sessions: handled correctly (15 vs 10 trials)". The baseline clip is justified in a code comment only ("Avoid division by zero (clip baseline to small positive value)") and is not discussed in the notes.

## 6-a. What are the most time-consuming steps of the code?

i. Per-session wall-clock is printed by the script; the full conversion of 41 sessions took well under the 15-minute budget (~1.3 s/session in the sample run, ≈50 s estimated for the full run). The dominant costs are (1) suite2p's `preprocess` maximin baseline filter on the (n_neurons × up to 54 000) matrix — run on GPU when available, (2) `np.load` of `F.npy`/`Fneu.npy` (tens of MB per session), and (3) at the end, pickling the 395 MB output, which also means all 41 sessions of float32 dF/F are held in RAM simultaneously.

ii.
```python
t_sess = time.time()
dff_binned, me_binned, n_neurons, n_frames_raw = process_session(sess_dir, device=device)
...
print(f"  {sess_name}: {n_neurons} neurons, {n_frames_raw} frames -> "
      f"{dff_binned.shape[1]} bins ({elapsed:.1f}s)")
...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

iii. CONVERSION_NOTES Step 6: "Suite2p's preprocess() for dF/F computation with GPU acceleration; Vectorized binning (reshape + mean); Processing time: ~1s per session", and Step 7: "Sample: 2.5 s for 2 sessions (~1.25 s/session); Estimated full: ~50 s for 41 sessions" — i.e. the AI measured the bottleneck and judged no further optimisation necessary.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops are all cheap and operate on at most a few hundred items: the per-trial slicing loop in `split_into_trials`, the per-trial loop that calls `make_time_input` and `discretize_me` (both could be done once per session on the whole trace and then reshaped/sliced), and the per-session outer loops. The heavy operations — neuropil subtraction, baseline filtering, binning (`reshape(...).mean(axis=…)`), `np.digitize` — are already fully vectorized. Nothing in the script has quadratic or reallocating behaviour (there is no `np.insert`-style growth loop).

ii.
```python
for t in range(n_trials):                       # 10–15 iterations/session, pure slicing
    neural_trials.append(dff_binned[:, start:end].astype(np.float32))
...
for neural_t, me_t in zip(neural_trials, me_trials):
    time_input = make_time_input(n_timebins)    # identical every iteration
    me_disc = discretize_me(me_t, bin_edges)    # could be one call per session
```

iii. CONVERSION_NOTES Step 6 claims "Vectorized binning (reshape + mean)" and Step 7 concludes the ~50 s estimate needs no further work; the residual per-trial loops are not discussed because they are negligible next to the baseline filter.

## 6-c. What processing does the code repeat multiple times?

i. Three repetitions, all minor. (1) `make_time_input(360)` recomputes the identical 360-element vector for all 545 trials (and stores 545 separate copies of it in the pickle). (2) `discretize_me` is called once per trial instead of once per session. (3) In `--show-processing` mode, `plot_processing` re-loads `F.npy`, `Fneu.npy` and `motion_energy_glob.npy` from disk for sessions that were already fully loaded and processed. Additionally the conversion is a deliberate two-pass design — all sessions are preprocessed and held in memory in pass 1 so that global percentile edges can be computed before pass 2 assembles the trials — which avoids recomputing dF/F but at the cost of peak memory.

ii.
```python
# repeated per trial
time_input = make_time_input(n_timebins)
me_disc = discretize_me(me_t, bin_edges)

# re-loads raw arrays already read in process_session()
def plot_processing(session_dir, ...):
    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    me_raw = np.load(os.path.join(me_dir, 'motion_energy_glob.npy'))
```

iii. Not discussed in CONVERSION_NOTES beyond the general Step 6 statement that binning was vectorized and runtime was ~1 s/session. The two-pass structure is an intentional consequence of Key Decision 3 (global percentile edges), which cannot be applied until every session has been processed.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `tstamps.npy` is loaded for every session and threaded through `interpolate_motion_energy`, which never uses it — pure dead I/O. (2) The extra baseline reconstruction and division in `compute_dff` (`baseline = F_corr − F_subtracted`, then a full-size divide) is work beyond the baseline-subtracted trace the paper describes, and is the source of the extreme values. (3) `me_disc_trials` is accumulated for every session but only consumed by the first two plots in `--show-processing` mode. (4) Descriptive `output_values` label strings are formatted from the bin edges on every run. (5) 545 identical copies of the time input are stored in the pickle. None of these except (2) materially affect runtime or the downstream decoder.

ii.
```python
tstamps = np.load(os.path.join(me_dir, 'tstamps.npy'))   # never used
def interpolate_motion_energy(me, n_neural_frames, tstamps):
    ...                                                   # tstamps not referenced
...
baseline = F_corr - F_subtracted
baseline_safe = np.clip(baseline, 1e-6, None)
dff = F_subtracted / baseline_safe
...
me_disc_trials.append(me_disc)      # only used by plot_processing
```

iii. Not documented in CONVERSION_NOTES; the notes state the intent to use timestamps for interpolation (Step 5 Key Decision 4), which explains why the array is loaded, and the dF/F division is presented as the correct realisation of the paper's dF/F definition (Step 10 Check 3).
