# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the list of the six subject folders (`MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']`) after listing `/app/data` and confirming those are the six mouse folders. For each mouse it lists the subject directory and keeps every entry that is itself a directory (this drops the stray `ground_truth.csv` files present in `jm038`, `jm039`, `jm046`), sorted alphabetically (= chronologically, since folders are named `YYYY-MM-DD_a`). For each session it loads four arrays: `suite2p/plane0/F.npy`, `suite2p/plane0/Fneu.npy`, `move_deve/motion_energy_glob.npy` and `move_deve/tstamps.npy`. `iscell.npy`, `spks.npy`, `stat.npy`, `ops.npy` and `interframe_int.npy` are inspected during exploration but not used by the final script. Trials are not read from disk; they are cut from the continuous session (see 1-d). The result is 41 sessions (7+7+7+7+6+7) and 1090 trials.

ii.
```python
DATA_DIR = '/app/data'
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for mouse_i, mouse in enumerate(MICE):
    mouse_dir = os.path.join(DATA_DIR, mouse)
    sessions = sorted([
        d for d in os.listdir(mouse_dir)
        if os.path.isdir(os.path.join(mouse_dir, d))
    ])
    for sess in sessions:
        session_dir = os.path.join(mouse_dir, sess)
        ...
```
```python
def process_session(session_dir):
    s2p_dir = os.path.join(session_dir, 'suite2p', 'plane0')
    move_dir = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
    dff = compute_dff(F, Fneu)

    me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
    tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The AI read `/app/data/README.md`, which documents the layout (one folder per subject, one sub-folder per recording day, `suite2p/` holding the tracked-cell traces and `move_deve/` holding the behavioural data), and `load_data.ipynb`, whose `load_traces` helper loads `F.npy` from `suite2p/plane0` and notes that "for more proper analysis compute dF/F the way as described in the paper". It then enumerated the folders by hand and encoded that enumeration as a constant. It explicitly checked shapes across all 41 sessions before writing the script (`F=(221, 36000) ... F=(435, 54000)`), and noted "6 mice, each with 6-7 sessions; jm031, jm032: 36000 frames (20 min at 30Hz); jm038-jm046: 54000 frames (30 min at 30Hz)".

## 1-b. How are the data split into subjects (mice)?

i. One subject per top-level `jm*` folder; the six ids are hard-coded in the `MICE` constant and used directly as the `subjects` field. `subject_idx` is the index of the mouse in that list, appended once per session, so it is ordered consistently with `neural`/`input`/`output`.

ii.
```python
MICE = ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
...
for mouse_i, mouse in enumerate(MICE):
    ...
    subject_idx.append(mouse_i)
...
'subjects': MICE,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. From the data README: "For each subject there is a folder corresponding to the subject id ... the subjects are named in alphabetically increasing order (e.g. jm031 - mouse A ... jm046 - mouse F)". The AI listed the directory and adopted those six ids verbatim.

## 1-c. How are the data split into sessions?

i. One session per date sub-folder of a subject (`2023-10-18_a`, ...), sorted alphabetically, keeping only entries that are directories. Each session is treated as an independent entry in the `neural`/`input`/`output` lists: 41 sessions total, with 20-minute sessions (36000 frames) for `jm031`/`jm032` and 30-minute sessions (54000 frames) for the other four mice. Sessions are never pooled across days even though the same neurons are tracked across days.

ii.
```python
sessions = sorted([
    d for d in os.listdir(mouse_dir)
    if os.path.isdir(os.path.join(mouse_dir, d))
])

for sess in sessions:
    session_dir = os.path.join(mouse_dir, sess)
    print(f"Processing {mouse}/{sess}...")
    dff_binned, me_binned = process_session(session_dir)
```

iii. The README states "Each subject folder contains a number of session folders, each corresponding to one recording day". The `os.path.isdir` filter was needed because three subject folders also contain a `ground_truth.csv` file, which the AI saw when it listed all subject directories. Sorting gives chronological (P8→P14) order.

## 1-d. How are the data split into trials?

i. The dataset has no task/stimulus structure (continuous spontaneous-activity recording), so trials are artificial: after binning to 3 Hz, each session is cut into non-overlapping, contiguous 60-second blocks of `TRIAL_BINS = int(60 / (10/30)) = 180` bins. Any tail shorter than a full trial is dropped (in practice there is none: 36000/10/180 = 20 and 54000/10/180 = 30 exactly, giving 20 or 30 trials per session and 1090 trials overall). Trials are cut *after* binning, and the same cut is applied to neural, input, and output streams.

ii.
```python
TRIAL_DUR = 60  # trial duration in seconds
TRIAL_BINS = int(TRIAL_DUR / BIN_DUR)  # 180 bins per 60s trial

def split_trials(neural, me_binned, trial_bins=TRIAL_BINS):
    n_bins = neural.shape[1]
    n_trials = n_bins // trial_bins
    neural_trials = []
    me_trials = []
    for t in range(n_trials):
        start = t * trial_bins
        end = start + trial_bins
        neural_trials.append(neural[:, start:end])
        me_trials.append(me_binned[start:end])
    return neural_trials, me_trials
```

iii. The instructions say "Split sessions into 60-second trials". The AI worked the arithmetic out explicitly in its reasoning: "10 frames at ~30Hz gives roughly 333ms bins, so 3600 binned timepoints per session split into 180-point, 60-second trials yields 20 trials per session", satisfying the format requirement of at least two trials per session.

## 1-e. How are trials filtered based on quality controls?

i. No trial-, session-, or subject-level filtering is performed. Every session of every mouse is kept, and every complete 60 s block is kept. There is no rejection of trials with dropped camera frames, low activity, or degenerate motion-energy distributions.

ii. N/A — there is no filtering code. The only data ever discarded is a partial trailing trial (`n_trials = n_bins // trial_bins`), which never occurs in this dataset.

iii. Not discussed explicitly in the trajectory. The AI verified up front that all sessions are complete and uniform (all 36000 or 54000 frames, all cells flagged `iscell == 1`, motion energy present for every session), so it evidently saw no basis for exclusions. The paper likewise analyses all recorded days.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the Suite2p-format outputs written by Track2p: the raw ROI fluorescence `suite2p/plane0/F.npy` and the neuropil fluorescence `suite2p/plane0/Fneu.npy`, both shape (n_neurons, n_frames). The deconvolved `spks.npy` is explicitly considered and rejected in favour of dF/F. The neuron count per mouse is fixed across days because Track2p only exports cells tracked on all days (221/370/685/746/541/435 neurons).

ii.
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
dff = compute_dff(F, Fneu)
```

iii. The AI reasoned: "I'm also wondering whether spks.npy should be used instead, but the paper seems to stick with dF/F for downstream analyses". The paper states it used "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)", and the loader notebook comments that one should "compute dF/F the way as described in the paper (or alternatively use spks.npy)". Computing dF/F needs both F and Fneu.

## 2-b. How is the `neural` data processed?

i. Suite2p's standard dF/F pipeline, re-implemented directly with `scipy.ndimage` rather than called from the `suite2p` package: (1) neuropil subtraction `Fc = F - 0.7*Fneu`; (2) "maximin" baseline `Flow` = Gaussian smoothing along time with `sig_baseline=10`, then a running minimum filter, then a running maximum filter, both with a `win_baseline*fs = 60*30 = 1800`-frame window; (3) baseline *subtraction* (not division) `dff = Fc - Flow`; cast to float32. The traces are then averaged in non-overlapping 10-frame bins (see 2-e). No z-scoring, smoothing, or per-neuron normalisation beyond this is applied. I verified numerically that this reproduces `suite2p.extraction.dcnv.preprocess(baseline='maximin', ...)` on this data (r = 0.999990 over a full session; residuals ~0.03% of the trace SD, from edge handling in suite2p 1.1.0's torch-based filters).

ii.
```python
NEUCOEFF = 0.7
SIG_BASELINE = 10.0
WIN_BASELINE = 60.0

def compute_dff(F, Fneu, fs=FS):
    """Compute dF/F using Suite2p default parameters.
    1. Neuropil correction: Fc = F - 0.7 * Fneu
    2. Maximin baseline: gaussian smooth -> min filter -> max filter
    3. Baseline subtraction: dF/F = Fc - Flow
    """
    Fc = F - NEUCOEFF * Fneu
    win = int(WIN_BASELINE * fs)
    Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    dff = Fc - Flow
    return dff.astype(np.float32)
```

iii. The AI located the paper's own implementation in `/app/code/track2p/gui/data_management.py::F_processing` and copied its structure (gaussian → min → max → subtract). It noticed a conflict — that function defaults to `neucoeff=0.0` while the paper says "default Suite2p parameters" — and resolved it in favour of Suite2p's real default: "the paper says 'using the default Suite2p parameters' which would mean neucoeff=0.7 ... I'll go with 0.7 as the paper states". It also explicitly confirmed that Suite2p's normalisation "is confirmed to be pure baseline subtraction (F - Flow), not division", and read `fs = 30` out of `ops.npy` for both recording lengths rather than assuming it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering. Every row of `F.npy` is kept; `iscell.npy` is loaded during exploration but never used in the conversion script. All 20445 neurons across the 41 sessions are exported, and `brain_region_idx` labels all of them as `barrel_cortex`.

ii. N/A — no filtering code. All neurons flow through:
```python
dff = compute_dff(F, Fneu)      # (n_neurons, n_frames), no row selection
n_neurons = dff.shape[0]
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The AI initially planned to apply the paper's `iscell` probability threshold ("cells are classified as true if their ROI probability exceeds the default 0.5 threshold"), then tested whether it would do anything: it printed `iscell` for the first session of each mouse and found `iscell sum == n_neurons, iscell all 1s = True` for all six mice. Its conclusion — consistent with the README's statement that the exported data "only includes traces for the cells present across all days" — was that the Track2p export is already curated, so the filter is a no-op and was omitted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no external alignment event. Trials are contiguous blocks of the continuous recording, so the alignment reference is the start of the recording session: trial *t* covers bins `[t*180, (t+1)*180)` of the session, i.e. seconds `[60t, 60(t+1))`. Metadata records `temporal_alignment_event = 'start of recording session'`, `off_start = 0.0`, `off_end = None`. All three streams (neural, time input, motion-energy output) are cut with the same indices, so they are aligned to one another by construction.

ii.
```python
neural_trials, me_trials = split_trials(dff_binned, me_binned)  # same indices for both
...
'metadata': {
    ...
    'temporal_alignment_event': 'start of recording session',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. Implicit in the AI's framing of the dataset: the recordings are of spontaneous behaviour with no stimulus or task events, and the instructions ask only for a 60-second segmentation plus "time elapsed from the beginning of the session", so session start is the only meaningful reference. (Note that `off_start = 0.0` is literally accurate only for the first trial of each session; the offset is `60*t` for trial *t*.)

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the 30 Hz acquisition is rebinned to 3 Hz by averaging non-overlapping blocks of 10 consecutive frames, giving a bin size of 10/30 s = 333.33 ms, recorded in `metadata['time_bin_size']` as 333.33 (ms). The identical binning is applied to the neural traces and to the (already length-matched) motion-energy trace, and the binning is done *before* the motion energy is discretised and before trials are cut, so all streams have exactly 180 bins per trial. A trailing partial bin would be truncated (`n = len // bin_size * bin_size`), though none occurs here.

ii.
```python
BIN_SIZE = 10  # frames per bin (paper: "averaging in bins of 10 consecutive timestamps")
BIN_DUR = BIN_SIZE / FS  # duration of each bin in seconds

def bin_data(data, bin_size=BIN_SIZE):
    if data.ndim == 1:
        n = len(data) // bin_size * bin_size
        return data[:n].reshape(-1, bin_size).mean(axis=1)
    else:
        n = data.shape[1] // bin_size * bin_size
        return data[:, :n].reshape(data.shape[0], -1, bin_size).mean(axis=2)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me_aligned, BIN_SIZE)
...
'time_bin_size': BIN_DUR * 1000,  # in ms: 333.33 ms
```

iii. Taken directly from the paper's decoding analysis, which the AI quoted as the source of the constant: "the paper denoises dF/F and behavior traces by averaging across bins of 10 consecutive timestamps", and it confirmed `fs = 30` from `ops.npy` for both the 20-min and 30-min recordings so that one bin size applies to every session.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Not derived from any stored variable. It is generated analytically from the bin index and the known constant frame rate (30 Hz from `ops.npy`) and bin width (10 frames): `input_names = ['time_in_session']`, one input channel, values in seconds since the start of that session. The camera `tstamps.npy` is not used for this.

ii.
```python
for t in range(len(neural_trials)):
    start_bin = t * TRIAL_BINS
    time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
    input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```

iii. The AI verified `fs = 30` and `nframes` in `ops.npy` for both session lengths, and established that `tstamps.npy` is a camera-clock array in odd units (it computed "tstamps are in kiloseconds" after seeing a 1.2097 span for a 1200 s recording). With a fixed, verified 30 Hz imaging clock, time can be computed exactly from the bin index, so no stored timebase is needed.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Three details: (1) time is expressed in seconds, not frames or bins; (2) the value assigned to a bin is the **centre** of that bin, hence the `+ 0.5` (so the first bin is 0.1667 s, not 0); (3) the clock is continuous across trials within a session — the offset `start_bin = t * 180` is added, so trial 0 runs 0.167→59.83 s, trial 1 runs 60.167→119.83 s, and the last bin of a 30-minute session is 1799.83 s. It is stored as float32 with shape (1, 180) per trial. The clock restarts at 0 for each new session.

ii.
```python
start_bin = t * TRIAL_BINS
# Time of each bin center relative to session start
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
input_trials.append(time_points.reshape(1, -1).astype(np.float32))
```
Verified in the produced pickle: `input[0][0][:, :4] = [0.1667, 0.5, 0.8333, 1.1667]`, `input[0][1][:, :3] = [60.167, 60.5, 60.833]`; overall range reported by the verifier as `time_in_session: [0.2, 1799.8]`.

iii. The instruction is "Time elapsed from the beginning of the session in seconds. Time-varying." — so the AI made the input a per-timepoint ramp rather than a per-trial scalar, and kept the ramp continuous across trials so the decoder sees the animal's position within the whole session (which matters because motion energy drifts over a session). Using the bin centre is the natural representative time for a bin-averaged sample.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: the time vector is built from the same bin indices used to slice the neural matrix, so element *j* of `input[s][t]` is the centre time of the bin in column *j* of `neural[s][t]`. Both have exactly 180 columns per trial, with no offset or resampling.

ii.
```python
neural_trials, me_trials = split_trials(dff_binned, me_binned)
for t in range(len(neural_trials)):
    start_bin = t * TRIAL_BINS                      # same index used by split_trials
    time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR
```

iii. Not separately justified in the trajectory; it follows from computing time from the bin index rather than from an independent clock, which makes misalignment impossible.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Solely from `move_deve/motion_energy_glob.npy`, the pre-computed global motion-energy trace from the behavioural video (one value per camera frame). `move_deve/tstamps.npy` is loaded and passed into `align_motion_energy()` but is never read inside that function; `move_deve/interframe_int.npy` is examined during exploration but is not loaded by the final script. So, in effect, the only raw variable used is the motion-energy trace itself plus the neural frame count.

ii.
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
n_neural = F.shape[1]
me_aligned = align_motion_energy(me, n_neural, tstamps)
```
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    ...
    # tstamps gives the time of each camera frame   <-- tstamps never used below
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
```

iii. The data README says `move_deve` "contains the processed behavioural data (motion energy extracted from videography of spontaneous behaviour 'motion_energy_glob.npy')", so no recomputation from video was needed (the raw `.avi` files are not distributed). The AI inspected `tstamps.npy` and `interframe_int.npy` at length to understand dropped frames and intended to use the timestamps for alignment, but the final implementation does not.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Three steps, in order: (1) **length matching** — if the motion-energy trace is shorter than the neural trace (dropped camera frames), it is resampled onto the neural frame grid with a single global linear interpolation, `np.interp` from `linspace(0,1,len(me))` to `linspace(0,1,n_neural)`; if lengths already match, the trace is returned untouched; (2) **binning** — averaging in non-overlapping 10-frame bins, the same operation applied to the neural data; (3) **discretisation** — into five equal-percentile bins computed within each session (see 4-c). No smoothing, log transform, normalisation, or outlier clipping is applied. Binning precedes discretisation, so the class labels come from bin-averaged motion energy.

ii.
```python
me_aligned = align_motion_energy(me, n_neural, tstamps)
...
me_binned = bin_data(me_aligned, BIN_SIZE)
...
me_discrete = discretize_me(me_trials)
...
all_output.append([me.reshape(1, -1) for me in me_discrete])
```

iii. The AI stated the plan as "interpolate any missing motion energy frames and bin those the same way, split into 60-second trials, then discretize motion energy into five equal-percentile bins per session". The binning follows the paper, which denoises "the behaviour traces" exactly as it denoises dF/F; the discretisation follows the task instruction that outputs must be categorical.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Into `N_BINS_OUTPUT = 5` equal-percentile (quintile) bins whose boundaries are computed **per session**: the binned motion energy of all trials of that session is pooled, the four inner boundaries are taken at the 20th/40th/60th/80th percentiles of that pooled distribution, and each time point is assigned with `np.digitize` to a level 0–4 stored as int64 with shape (1, 180) per trial. Labels are `['bin_0', ..., 'bin_4']`. Because the edges are session-local and every session's length is an exact multiple of the trial length, each class occupies exactly 20% of every session (the verifier reported `{bin_0 (0.200), ..., bin_4 (0.200)}`).

ii.
```python
def discretize_me(me_trials, n_bins=N_BINS_OUTPUT):
    # Pool all ME values from this session
    all_me = np.concatenate(me_trials)
    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # inner boundaries
    boundaries = np.percentile(all_me, percentiles)
    discretized = []
    for me in me_trials:
        binned = np.digitize(me, boundaries)  # 0 to n_bins-1
        discretized.append(binned.astype(np.int64))
    return discretized
```

iii. Directly from the instruction "Motion energy, discretized into five equal-percentile bins, selected per session. Time-varying." Per-session edges also absorb the session-to-session differences in absolute motion-energy scale (illumination, camera position, animal age), which would otherwise make the classes incomparable across days, and they guarantee a balanced 5-class problem against a 0.2 chance level.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Video and 2-photon acquisition are synchronous at 30 Hz, so in 30 of 41 sessions the motion-energy array already has exactly the same length as the neural array and is used as-is, index for index. In the 11 sessions where the camera dropped frames (1–2 frames in most, but 116 in `jm031/2023-10-22_a` and 148 in `jm032/2023-10-22_a`), the AI stretches the whole trace uniformly onto the neural frame grid with `np.interp`, i.e. it assumes the dropped frames are spread evenly over the session rather than placing them where they actually occurred. The per-frame timestamps that record where the drops occurred are loaded but not used. After this, both streams are binned identically and cut with the same trial indices, so alignment is index-for-index thereafter.

ii.
```python
def align_motion_energy(me, n_neural_frames, tstamps):
    """Align motion energy to neural frames by interpolation.
    When camera drops frames, ME has fewer samples than neural data.
    Use tstamps to interpolate ME to the full neural frame count.
    """
    if len(me) == n_neural_frames:
        return me
    camera_pos = np.linspace(0, 1, len(me))
    neural_pos = np.linspace(0, 1, n_neural_frames)
    me_aligned = np.interp(neural_pos, camera_pos, me)
    return me_aligned
```

iii. The AI established from the README that "In some recordings there might be some missing frames from the camera ... The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over", and chose the interpolate-over option. Its reasoning explicitly proposed the timestamp-based version — "Since tstamps give the actual timing of each camera frame while neural frames sit at regular 30Hz intervals, the cleanest approach is interpolating ME values onto the neural frame times using those timestamps to identify gaps" — but the code it wrote performs a uniform stretch instead, leaving `tstamps` unused.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The only defect present in this dataset is dropped camera frames, handled by the uniform interpolation described in 4-d. There are no assertions, warnings, or logging around it — the fix is silent, and a session whose motion energy was *longer* than the neural trace, or grossly wrong in length, would be silently compressed onto the neural grid rather than flagged. No NaN handling is implemented (I confirmed none of the loaded arrays or the exported neural data contain NaNs). Partial trailing bins and partial trailing trials are silently truncated, which is a no-op for this dataset (every session is an exact multiple of 1800 frames).

ii.
```python
if len(me) == n_neural_frames:
    return me
# ... otherwise silently resample, no check on the size of the discrepancy
me_aligned = np.interp(neural_pos, camera_pos, me)
```
```python
n = data.shape[1] // bin_size * bin_size   # silent truncation of a partial bin
n_trials = n_bins // trial_bins            # silent drop of a partial trial
```

iii. The AI verified up front that the problem is bounded — it printed `F`, `ME`, `tstamps` and `interframe_int` shapes for every one of the 41 sessions and saw that mismatches are at most 148 frames out of 36000 (0.4%) — so it treated interpolation as sufficient and did not add guards.

## 6-a. What are the most time-consuming steps of the code?

i. `compute_dff` dominates. For each session it casts the (n_neurons × n_frames) corrected trace to float64 and runs a Gaussian filter plus a 1800-frame running-minimum and a 1800-frame running-maximum along time — on `jm039` that is 746 × 54000 samples per filter pass, over 41 sessions. Second is file I/O: `F.npy` and `Fneu.npy` are up to ~161 MB each (746 × 54000 float32) and are read in full for every session. Everything afterwards (binning, slicing, `np.percentile`, `np.digitize`, pickling) is cheap by comparison. The script is single-threaded, CPU-only, and processes sessions one at a time.

ii.
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])  # float64 copy
Flow = minimum_filter1d(Flow, win)   # win = 1800
Flow = maximum_filter1d(Flow, win)
```

iii. Not discussed in the trajectory; the AI never profiled or timed the script (the full run over 41 sessions completed in one call without any need for optimisation).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Three, all minor: (1) the trial loop in `split_trials`, which builds 20–30 slices one at a time when a single `reshape` of the binned array to (n_neurons, n_trials, 180) would produce them all at once; (2) the per-trial loop building the time input, which recomputes `np.arange(180)` for every trial of every session when one session-long ramp could be built once and reshaped; (3) the per-trial `np.digitize` loop in `discretize_me`, which could digitise the whole pooled session vector in one call and then split. All three iterate over at most 30 trials with ~180-element arrays, so the gain would be negligible relative to `compute_dff`.

ii.
```python
for t in range(n_trials):
    start = t * trial_bins
    end = start + trial_bins
    neural_trials.append(neural[:, start:end])
    me_trials.append(me_binned[start:end])
```
```python
for me in me_trials:
    binned = np.digitize(me, boundaries)
    discretized.append(binned.astype(np.int64))
```

iii. Not discussed. The per-trial list-of-arrays structure is what the target format requires, so the loops largely reflect the output format rather than avoidable work.

## 6-c. What processing does the code repeat multiple times?

i. (1) The motion energy is split into trials and then immediately re-concatenated inside `discretize_me` (`all_me = np.concatenate(me_trials)`) to compute the session percentiles — the pooled session vector `me_binned` was already in hand one line earlier. (2) The time ramp `np.arange(TRIAL_BINS) * BIN_DUR` is rebuilt for every trial although only the constant offset differs. (3) `np.zeros(n_neurons)` for `brain_region_idx` is rebuilt per session even though it is identical across the seven sessions of a mouse. All are trivially cheap.

ii.
```python
neural_trials, me_trials = split_trials(dff_binned, me_binned)
me_discrete = discretize_me(me_trials)
...
    all_me = np.concatenate(me_trials)   # reconstructs me_binned
```
```python
time_points = (np.arange(TRIAL_BINS) + start_bin + 0.5) * BIN_DUR  # per trial
```

iii. Not discussed in the trajectory. Splitting before discretising keeps `discretize_me` self-contained (it returns a per-trial list matching the output format), which appears to be the reason for the round trip.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `tstamps.npy` is loaded for every session (a 36000–54000 element float64 array, ~0.3–0.4 MB per session) and threaded through `process_session` into `align_motion_energy`, where it is never used — pure dead work, and more importantly the information it carries is exactly what the alignment step needed. (2) `compute_dff` promotes the full trace to float64 for the filtering and then casts back to float32, doubling peak memory and filter cost for a result that is stored at float32 precision. (3) dF/F is computed at the full 30 Hz resolution for the whole session and then immediately averaged down by a factor of 10, so 90% of the computed samples are discarded (this ordering is nonetheless correct — the 1800-frame maximin baseline must be estimated at native resolution, and averaging after baseline subtraction is what the paper does). (4) `print` of a progress line per session, negligible.

ii.
```python
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))   # never read
me_aligned = align_motion_energy(me, n_neural, tstamps)
```
```python
Flow = gaussian_filter(Fc.astype(np.float64), [0., SIG_BASELINE])
...
return dff.astype(np.float32)
```

iii. Not discussed. The unused `tstamps` load is a leftover from the timestamp-based alignment the AI described in its reasoning but did not implement.
