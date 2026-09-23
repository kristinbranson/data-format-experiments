# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the data directory for subdirectories starting with `jm` to identify subjects, then iterates over subdirectories within each subject folder for sessions. For each session, it loads `F.npy` and `Fneu.npy` from `suite2p/plane0/` for neural data, and also loads `iscell.npy` and `ops.npy` to validate and extract parameters. Motion energy is loaded from `move_deve/motion_energy_glob.npy` and inter-frame intervals from `move_deve/interframe_int.npy`. All sessions are processed via a `ProcessPoolExecutor` for parallelism.

ii.
```python
def load_session_raw(session_dir):
    p = os.path.join(session_dir, 'suite2p', 'plane0')
    F = np.load(os.path.join(p, 'F.npy'))
    Fneu = np.load(os.path.join(p, 'Fneu.npy'))
    iscell = np.load(os.path.join(p, 'iscell.npy'))
    ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
    assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROIs'
    assert F.shape == Fneu.shape
    assert F.shape[1] == ops['nframes'], f'{session_dir}: F length != ops["nframes"]'
    return F, Fneu, float(ops['fs']), int(ops['nframes']), float(ops['neucoeff'])
```

```python
def list_sessions(subject):
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())

SUBJECTS = list(SUBJECT_INFO.keys())  # ['jm031','jm032','jm038','jm039','jm040','jm046']
session_dirs = []
for s in SUBJECTS:
    session_dirs.extend(list_sessions(s))
```

iii. The AI uses the standard directory structure convention (jm* subject folders containing session subfolders). It additionally loads `iscell.npy` and `ops.npy` to validate that all ROIs are cells and to extract runtime parameters (`fs`, `neucoeff`, `nframes`), providing assertions for data integrity. The reference code does not load these extra files.

## 1-b. How are the data split into subjects?

i. Subjects are hardcoded in a `SUBJECT_INFO` dictionary mapping folder names to paper labels and postnatal day offsets. The order matches the paper's alphabetical labeling (A-F).

ii.
```python
SUBJECT_INFO = {
    'jm031': ('A', 7),
    'jm032': ('B', 7),
    'jm038': ('C', 8),
    'jm039': ('D', 8),
    'jm040': ('E', 9),
    'jm046': ('F', 8),
}
SUBJECTS = list(SUBJECT_INFO.keys())
```

iii. Hardcoding subjects provides additional metadata (paper labels, postnatal day offsets) and ensures a deterministic order. The reference dynamically discovers subjects by scanning for `jm*` directories.

## 1-c. How are the data split into sessions?

i. Sessions are subdirectories within each subject folder, sorted alphabetically (which corresponds to chronological order since folder names are dates like `YYYY-MM-DD_a`).

ii.
```python
def list_sessions(subject):
    d = os.path.join(DATA_DIR, subject)
    return sorted(f.path for f in os.scandir(d) if f.is_dir())
```

iii. Same approach as the reference. Sorting by directory name gives chronological order.

## 1-d. How are the data split into trials?

i. Trials are 60-second non-overlapping segments of the continuous recording. After 10-frame binning (30 Hz to 3 Hz), each trial is 180 bins. Remainder bins that don't fill a complete trial are discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC * fs / BIN_FRAMES))  # 60 * 30 / 10 = 180
n_trials = dff_binned.shape[1] // bins_per_trial
n_used = n_trials * bins_per_trial
dff_binned = dff_binned[:, :n_used]
me_binned = me_binned[:n_used]
```

iii. Per the task instructions, trials are defined as 60-second segments. This matches the reference approach.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All complete 60-second trials are included. The AI documents that there is no task structure and no basis for rejecting trials.

ii. N/A (no filtering code)

iii. The AI notes that every session has complete neural and behavioural coverage, so no trials need to be dropped. The only data quality issue (dropped camera frames) is handled by interpolation rather than trial exclusion. This matches the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `F.npy` (raw fluorescence) and `Fneu.npy` (neuropil fluorescence) from `suite2p/plane0/`. The neuropil coefficient is read from `ops.npy` (`ops['neucoeff']` = 0.7).

ii.
```python
F, Fneu, fs, n_frames, ops_neucoeff = load_session_raw(session_dir)
# ops_neucoeff comes from ops['neucoeff'] = 0.7
```

iii. Same source variables as the reference. The AI additionally reads the neuropil coefficient from the session's `ops.npy` rather than hardcoding it, though the value is the same (0.7).

## 2-b. How is the `neural` data processed?

i. The AI reimplements the suite2p maximin baseline correction using scipy functions directly, rather than calling `dcnv.preprocess`. The processing is: (1) neuropil subtraction `Fc = F - neucoeff * Fneu`, (2) Gaussian smoothing with sigma=10 frames, (3) minimum filter with 60s window, (4) maximum filter with 60s window, (5) subtract baseline: `Fc - Flow`. After baseline correction, the data is averaged into 10-frame bins. Additionally, the AI applies per-neuron mean subtraction and divides by a single scalar per session (the pooled standard deviation), termed "center" scaling.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff, baseline='maximin',
                 sig_baseline=SIG_BASELINE, win_baseline=WIN_BASELINE):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    return Fc - Flow, Flow

# Then in convert_session:
dff, F0 = f_processing(F, Fneu, fs, neucoeff)
dff_binned = bin_average(dff).astype(np.float32)

# Neural scaling (center mode):
scale = float(dff_binned.std())
neural_all = (dff_binned - dff_binned.mean(axis=1, keepdims=True)) / scale
```

iii. The baseline correction reimplements the same algorithm as `dcnv.preprocess` (which internally uses the same scipy functions). The AI justifies the additional centering/scaling as information-preserving for a linear readout with bias, arguing it improves SVD initialization conditioning. The reference code does not apply any neural scaling beyond the baseline correction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering is applied. The AI asserts that all ROIs in the released data have `iscell[:,0] == 1` and that the neuron count is consistent across days within a mouse, confirming that the Track2p curation was already applied.

ii.
```python
assert np.all(iscell[:, 0] == 1), f'{session_dir}: unexpected non-cell ROIs'
```

iii. The released data already contains only Track2p-tracked, iscell-curated neurons. The AI verifies this with assertions rather than re-filtering. This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to the start of the imaging session. The continuous recording is cut into consecutive 60-second segments starting from the first frame. No event-based alignment is needed since there is no stimulus.

ii.
```python
'temporal_alignment_event':
    'start of the imaging session (first 2-photon frame); the continuous '
    'session is cut into consecutive non-overlapping 60 s trials, and each '
    'trial is aligned to its own start',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

iii. There is no stimulus event in this spontaneous activity dataset. The reference sets `off_start: None` and `off_end: None`, while the AI sets `off_start: 0.0` and `off_end: 60.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Both neural and motion energy traces are averaged into non-overlapping bins of 10 consecutive frames, reducing 30 Hz to 3 Hz (333.33 ms per bin). Binning is applied before motion energy discretization.

ii.
```python
BIN_FRAMES = 10
dff_binned = bin_average(dff).astype(np.float32)
me_binned = bin_average(me)

def bin_average(x, k=None):
    k = BIN_FRAMES if k is None else k
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(*x.shape[:-1], n // k, k).mean(axis=-1)
```

iii. Matches the paper's description: "for all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps". Same as the reference.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is not derived from any raw data variable. It is computed from the bin index, the bin size (10 frames), and the sampling rate (30 Hz). The AI uses the center of each bin.

ii.
```python
bin_dt = BIN_FRAMES / fs  # 10/30 = 0.333... s
time_all = (np.arange(n_used) + 0.5) * bin_dt
```

iii. Since the sampling rate is constant, computing time from bin indices is equivalent to loading timestamps. The AI uses bin centers (`+ 0.5`), while the reference uses left bin edges (no `+ 0.5` offset).

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. Simple arithmetic: `time = (bin_index + 0.5) * (BIN_FRAMES / fs)` seconds. No further processing.

ii.
```python
time_all = (np.arange(n_used) + 0.5) * bin_dt
# For each trial:
inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. The `+ 0.5` places the time at the center of each bin rather than the left edge. This is a minor difference from the reference.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input uses the same bin indices as the neural data, so alignment is inherent. The time runs continuously across the session (not reset per trial), so trial 2's time starts at 60s, trial 3 at 120s, etc.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
    inputs.append(time_all[sl][None, :].astype(np.float32))
```

iii. Both the neural and time arrays share the same bin indexing, ensuring alignment. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motion_energy_glob.npy` in the `move_deve` subdirectory. Inter-frame intervals from `interframe_int.npy` are used to detect dropped camera frames and align motion energy to the imaging frame grid.

ii.
```python
me = np.load(os.path.join(md, 'motion_energy_glob.npy')).astype(np.float64)
itv = np.load(os.path.join(md, 'interframe_int.npy'))
```

iii. Same source files as the reference. The AI casts motion energy to float64 for precision during interpolation.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Align motion energy to imaging frame grid using cumulative sum of rounded inter-frame intervals to map each camera frame to its imaging frame index. (2) Treat `me[0]` as missing (it's a placeholder with no preceding video frame). (3) Linearly interpolate all missing frames (dropped triggers and me[0]) using `np.interp`. (4) Average into 10-frame bins. (5) Discretize into 5 quintile bins per session.

ii.
```python
nominal = np.median(itv)
steps = np.round(itv / nominal).astype(np.int64)
idx = np.concatenate([[0], np.cumsum(steps)])

out = np.full(n_frames, np.nan)
keep = idx < n_frames
out[idx[keep]] = me[keep]
out[0] = np.nan  # me[0] is a placeholder

missing = np.isnan(out)
out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), out[~missing])
```

iii. The cumulative-sum alignment method is more principled than the reference's threshold-based approach (`dt * 1000 > 0.04`). The AI's approach maps every camera frame to its correct imaging frame index based on the actual inter-frame intervals, rather than detecting individual dropped frames by a threshold. Additionally, the AI correctly identifies that `me[0]` is a placeholder (no preceding video frame for computing frame differences) and treats it as missing data. The reference does not handle `me[0]` specially.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The binned motion energy is discretized into 5 equal-percentile (quintile) bins. Thresholds are computed separately for each session using the interior quantile edges (20th, 40th, 60th, 80th percentiles). Values are assigned to bins 0-4 using `np.searchsorted`.

ii.
```python
def discretize_quantiles(x, n_bins=N_QUANTILES):
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]  # [20, 40, 60, 80]
    edges = np.percentile(x, qs)
    labels = np.searchsorted(edges, x, side='right').astype(np.int64)
    return labels, edges
```

iii. Per-session quintiles ensure each class holds approximately 20% of bins per session, as the task specifies. The reference uses `np.digitize(me, bin_edges[1:-1])` which is functionally equivalent to `np.searchsorted(edges, x, side='right')`.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The camera is triggered by the 2-photon microscope, so camera frame i corresponds to imaging frame i when no trigger is dropped. Dropped triggers are identified from the inter-frame intervals: the cumulative sum of rounded interval ratios gives the imaging-frame index for every camera frame. Missing frames are linearly interpolated. After alignment and interpolation, the motion energy array has the same length as the neural data, and both are binned together with the same 10-frame averaging, ensuring alignment.

ii.
```python
nominal = np.median(itv)
steps = np.round(itv / nominal).astype(np.int64)
idx = np.concatenate([[0], np.cumsum(steps)])

out = np.full(n_frames, np.nan)
keep = idx < n_frames
out[idx[keep]] = me[keep]
# ... interpolation ...

# Both streams binned together:
me_binned = bin_average(me)
dff_binned = bin_average(dff)
```

iii. The AI's alignment method reconstructs the imaging-frame index for every camera frame rather than detecting individual dropped frames by a threshold. The AI also handles the edge case where three jm046 sessions have camera frames that extend past the last 2-photon frame (`idx < n_frames` guard). The reference uses a simpler threshold-based detection (`dt * 1000 > 0.04`) and loop-based insertion.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data issues are handled: (1) Dropped camera frames are detected via inter-frame interval analysis and linearly interpolated. (2) `me[0]` is treated as missing (placeholder) and interpolated. (3) Camera frames past the last imaging frame are discarded. (4) Remainder bins that don't fill a complete trial are discarded. (5) Assertions check data integrity throughout (iscell, array shapes, nframes, interframe_int length).

ii.
```python
# me[0] placeholder handling:
out[0] = np.nan

# Camera frames past last imaging frame:
keep = idx < n_frames

# Interpolation of all missing values:
out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), out[~missing])

# Assertions:
assert np.all(iscell[:, 0] == 1)
assert F.shape == Fneu.shape
assert F.shape[1] == ops['nframes']
assert len(itv) == len(me) - 1
```

iii. The AI handles more edge cases than the reference (me[0] handling, camera-past-imaging guard). The reference handles dropped frames with a threshold-based approach and an assertion that lengths match post-interpolation.

## 6-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the maximin baseline correction (Gaussian smoothing + min/max filtering over the full session for all neurons). Loading `ops.npy` (85 MB per session due to mean images) is also noted as an I/O cost. The AI mitigates this with 8-way parallel processing.

ii. N/A

iii. The CONVERSION_NOTES document the dF/F computation as taking ~0.3-0.8s per session, with the full dataset converting in ~6s thanks to parallelism. The reference identifies the same bottleneck (`dcnv.preprocess`).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is already well-vectorized. The main remaining loop is the per-trial slicing loop (`for tr in range(n_trials)`) which could potentially be replaced with array reshape operations, though the cost is negligible. The AI does not have the `np.insert` loop that appears in the reference code, since it uses `np.interp` for interpolation.

ii.
```python
for tr in range(n_trials):
    sl = slice(tr * bins_per_trial, (tr + 1) * bins_per_trial)
    neural.append(np.ascontiguousarray(neural_all[:, sl]))
    inputs.append(time_all[sl][None, :].astype(np.float32))
    outputs.append(labels_all[sl][None, :])
```

iii. The AI documents having already vectorized the binning (`reshape(...).mean(-1)`) and uses parallel processing across sessions. The trial-cutting loop is unavoidable since the output format requires a list of per-trial arrays.

## 6-c. What processing does the code repeat multiple times?

i. When `--show-processing` is enabled, the AI processes sessions twice: once in `plot_processing()` (which calls `convert_session()` internally) and once in the main conversion loop. This doubles the work for up to 2 sessions.

ii.
```python
if args.show_processing:
    for sd in session_dirs[:2]:
        sid = f'{os.path.basename(os.path.dirname(sd))}_{os.path.basename(sd)}'
        plot_processing(sd, args.neural_scaling, f'/app/processing_{sid}.png')

# Then later, all sessions (including the plotted ones) are converted again:
sessions = list(ex.map(_worker, [(sd, args.neural_scaling, args.neucoeff) for sd in session_dirs]))
```

iii. The double processing only affects the `--show-processing` mode and at most 2 sessions, so the practical impact is minimal.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ops.npy` (approximately 85 MB per session due to stored mean images and registration info) but only uses three fields (`fs`, `nframes`, `neucoeff`). The remaining data (mean images, registration metadata) is loaded into memory and immediately discarded. Additionally, the neural scaling (centering and division by session std) is extra processing not present in the reference.

ii.
```python
ops = np.load(os.path.join(p, 'ops.npy'), allow_pickle=True).item()
# Only uses: ops['fs'], ops['nframes'], ops['neucoeff']
```

iii. Loading the full `ops.npy` is an I/O inefficiency. The neural scaling computation (mean subtraction, std division) is an additional processing step that goes beyond the reference paper's description.
