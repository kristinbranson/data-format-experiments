# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every top-level directory under `/app/data` as a subject, every subdirectory under each subject as a session, and then loads each session's neural and behavioral files inside `load_session()`. For each session it reads `F.npy`, `Fneu.npy`, `ops.npy`, `iscell.npy`, `motion_energy_glob.npy`, and `tstamps.npy`, then later slices the processed session arrays into trials.

ii.
```python
def load_session(session_dir):
    s2p = os.path.join(session_dir, 'suite2p', 'plane0')
    mov = os.path.join(session_dir, 'move_deve')

    F = np.load(os.path.join(s2p, 'F.npy'))
    Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
    ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(s2p, 'iscell.npy'))
    ...
    me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
    ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0

def main():
    subjects = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)))
    ...
    for si, subj in enumerate(subjects):
        subj_dir = os.path.join(DATA_DIR, subj)
        sessions = sorted(d for d in os.listdir(subj_dir)
                          if os.path.isdir(os.path.join(subj_dir, d)))
```

iii. In trajectory step 49, the AI said the shipped `suite2p` folders are the Track2p outputs, that all 6 mice and all 6-7 daily recordings should be kept, and that motion energy should be realigned using camera timestamps before trialization.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all top-level directories inside `/app/data`, sorted alphabetically. The script does not explicitly filter for names beginning with `jm`; it assumes every directory there is a subject.

ii.
```python
subjects = sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))
```

iii. In trajectory step 49, the AI justified keeping "All 6 mice × 6-7 days" and treated each mouse directory as one subject.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory, sorted alphabetically. Each session directory is processed independently and becomes one session in the output lists.

ii.
```python
for si, subj in enumerate(subjects):
    subj_dir = os.path.join(DATA_DIR, subj)
    sessions = sorted(d for d in os.listdir(subj_dir)
                      if os.path.isdir(os.path.join(subj_dir, d)))
    for di, sess in enumerate(sessions):
        dff, t, me, dt = load_session(os.path.join(subj_dir, sess))
```

iii. In trajectory step 49, the AI described these as the daily recordings for each mouse and said session order within each mouse is chronological.

## 1-d. How are the data split into trials?

i. After per-session preprocessing and 10-frame temporal binning, each session is split into consecutive non-overlapping 60-second blocks. The code uses `180` binned samples per trial (`60 s * 30 Hz / 10 frames per bin`), takes `n_bins // 180` full trials, and ignores any leftover tail that does not fill a full block.

ii.
```python
TRIAL_SECONDS = 60.0
NOMINAL_FS = 30.0
FRAMES_PER_BIN = 10
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * NOMINAL_FS / FRAMES_PER_BIN))  # 180

...
n_bins = dff.shape[1]
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
    ninp.append(t[sl][None, :].astype(np.float32))
    nout.append(cls[sl][None, :])
```

iii. In trajectory step 49, the AI explicitly justified "consecutive non-overlapping 1800-frame blocks (60 s at the nominal 30 Hz) = 180 bins; 20 trials per 20-min session, 30 per 30-min session."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit per-trial quality-control filter. It only requires each session to contain at least two full 60-second trials and implicitly discards any trailing bins that do not complete a full trial because the loop only iterates over `range(n_trials)`.

ii.
```python
n_trials = n_bins // BINS_PER_TRIAL
assert n_trials >= 2

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ...
```

iii. In trajectory step 49, the AI said there was "no further neuron/session/mouse curation" because the released dataset was already the curated set used in the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural output is derived from suite2p fluorescence arrays `F.npy` and `Fneu.npy`. The script also reads `ops.npy` to get the frame rate used by the baseline filter and `iscell.npy` to assert that all ROIs are cells, but the signal itself is computed from `F` and `Fneu`.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
fs = float(ops['fs'])
...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
```

iii. In trajectory step 49, the AI said it reimplemented `track2p/gui/data_management.py::F_processing` and interpreted the paper's "baseline corrected fluorescence traces" as the neural signal to use.

## 2-b. How is the `neural` data processed?

i. The AI applies a custom Track2p-style baseline correction. It computes `Fc = F - neucoeff * Fneu` with `neucoeff=0`, so there is no neuropil subtraction in practice. It then smooths with a Gaussian filter over time, applies a minimum filter followed by a maximum filter over a 60-second window, and returns `Fc - Flow`. After that it averages the signal in non-overlapping bins of 10 frames and stores the result as `float32`.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow

...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
...
dff_b = bin_mean(dff, FRAMES_PER_BIN)
...
return dff_b.astype(np.float32), t_b, me_b, float(np.median(np.diff(tstamps)))
```

iii. In trajectory step 49, the AI justified this by saying the paper described "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" and that it therefore reimplemented Track2p's `F_processing` exactly with `neucoeff=0`, maximin baseline correction, and no z-scoring.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out of the session arrays. Instead, it asserts that every ROI in `iscell.npy` is already marked as a cell and then keeps all rows in `F.npy`.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
...
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in Track2p output'
```

iii. In trajectory step 49, the AI said the released `suite2p` folders already contain only tracked neurons that passed suite2p's default `iscell` threshold, so it asserted this rather than applying another filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to the start of each artificial 60-second trial block, with trial blocks beginning at imaging onset and then continuing consecutively through the session. This is reflected in the metadata rather than by a separate event array.

ii.
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))

...
'metadata': {
    'temporal_alignment_event':
        'start of the 60 s trial block; each session is cut into '
        'consecutive non-overlapping 1800-frame (60 s) blocks starting '
        'at imaging onset',
    'off_start': 0.0,
    'off_end': float(np.mean(bin_durations) * BINS_PER_TRIAL),
}
```

iii. In trajectory step 49, the AI justified this as "Sessions are cut into consecutive, non-overlapping 60 s trials" and described the alignment event as the start of those blocks at imaging onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use non-overlapping 10-frame bins for both neural and behavioral streams. The stored time-bin size is based on the measured camera timestamp spacing rather than exactly `10 / 30` seconds, so the metadata reports an average bin size of about `335.88 ms`.

ii.
```python
FRAMES_PER_BIN = 10

def bin_mean(x, k):
    n = (x.shape[-1] // k) * k
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // k, k).mean(axis=-1)

...
dff_b = bin_mean(dff, FRAMES_PER_BIN)
me_b = bin_mean(me, FRAMES_PER_BIN)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)

...
time_bin_size_ms = float(np.mean(bin_durations) * 1000.0)
```

iii. In trajectory step 49, the AI cited the paper's statement that decoding analyses average "10 consecutive timestamps" and said this yields `time_bin_size = 335.88 ms` when using the recovered camera timing.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is derived from `tstamps.npy`, after mapping the recorded camera timestamps back onto the imaging-frame grid. It is not computed from a simple frame index.

ii.
```python
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
...
idx = camera_frame_index(ts_cam, n_frames)
...
tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)
...
t_b = bin_mean(tstamps, FRAMES_PER_BIN)
```

iii. In trajectory step 49, the AI said "`tstamps.npy` is in units of 1000 s; ×1000 gives the true frame interval of 33.59 ms (29.78 Hz), used for the time input."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI reconstructs dropped camera frames from timestamp gaps, creates a full imaging-frame timestamp vector with `NaN` placeholders, linearly interpolates missing entries, and then averages timestamps within each 10-frame bin. The result is the bin-center time series.

ii.
```python
idx = camera_frame_index(ts_cam, n_frames)

tstamps = np.full(n_frames, np.nan)
tstamps[idx] = ts_cam
tstamps = interp_nan(tstamps)

t_b = bin_mean(tstamps, FRAMES_PER_BIN)  # bin-centre time, seconds
```

iii. In trajectory step 49, the AI justified this by saying it recovered the true imaging-frame index from cumulative gap counts, verified the reconstructed indices, and used the repaired timestamps as the basis of the time input.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The time input is aligned by being reconstructed on the same imaging-frame grid as the neural data, then binned with the same 10-frame averaging, and finally sliced with the same per-trial index ranges as the neural matrix.

ii.
```python
dff_b = bin_mean(dff, FRAMES_PER_BIN)
t_b = bin_mean(tstamps, FRAMES_PER_BIN)

...
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
    ninp.append(t[sl][None, :].astype(np.float32))
```

iii. In trajectory step 49, the AI's alignment justification was that the camera was hardware-triggered by the microscope, so the repaired timestamp trace could be put directly onto the imaging frame grid and sliced together with the neural data.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion-energy output is derived from `motion_energy_glob.npy` and `tstamps.npy`. The motion-energy vector provides the behavioral signal, and the timestamps are used to place that signal back onto imaging-frame indices when camera frames were dropped.

ii.
```python
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
ts_cam = np.load(os.path.join(mov, 'tstamps.npy')).astype(np.float64) * 1000.0
...
idx = camera_frame_index(ts_cam, n_frames)
```

iii. In trajectory step 49, the AI said the camera was hardware-triggered by the microscope and that dropped frames should be recovered from the camera timestamps.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI maps the recorded motion-energy samples onto the full imaging-frame grid using recovered frame indices, fills missing frames by linear interpolation, forces the first frame to be missing because motion energy at frame 0 is degenerate, averages the repaired series in 10-frame bins, and only then discretizes it.

ii.
```python
me_cam = np.load(os.path.join(mov, 'motion_energy_glob.npy')).astype(np.float64)
...
idx = camera_frame_index(ts_cam, n_frames)

me = np.full(n_frames, np.nan)
me[idx] = me_cam
me[0] = np.nan
me = interp_nan(me)

me_b = bin_mean(me, FRAMES_PER_BIN)
...
cls = discretize_quintiles(me)
```

iii. In trajectory step 49, the AI justified this by saying dropped camera frames were identified from exact multiples of the nominal interval, missing motion-energy values were linearly interpolated, and both streams were binned together in 10-frame windows.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized separately within each session into five equal-percentile bins. The code computes the 20th, 40th, 60th, and 80th percentile edges and uses `np.digitize` to assign class labels `0` through `4`.

ii.
```python
def discretize_quintiles(x, nbins=N_OUTPUT_BINS):
    edges = np.percentile(x, np.linspace(0, 100, nbins + 1)[1:-1])
    return np.digitize(x, edges).astype(np.int64)
```

iii. In trajectory step 49, the AI explicitly said the output is "motion energy in 5 per-session quintile bins (exactly 20% per class, as intended)."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to neural data by reconstructing the imaging-frame index of every recorded camera sample, filling any missing behavioral frames on that same grid, averaging both streams in identical 10-frame bins, and then slicing both arrays with the same trial boundaries.

ii.
```python
idx = camera_frame_index(ts_cam, n_frames)

me = np.full(n_frames, np.nan)
me[idx] = me_cam
me = interp_nan(me)

dff_b = bin_mean(dff, FRAMES_PER_BIN)
me_b = bin_mean(me, FRAMES_PER_BIN)

for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
    nout.append(cls[sl][None, :])
```

iii. In trajectory step 49, the AI justified this as hardware-triggered camera/imaging synchronization plus dropped-frame repair from timestamp gaps, followed by shared 10-frame binning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles dropped or missing camera frames by reconstructing the intended imaging-frame indices from timestamp gaps and linearly interpolating missing motion-energy and timestamp values on the full frame grid. It treats the first motion-energy sample as missing because it has no preceding video frame, asserts the recovered frame indexing is self-consistent, and drops any partial tail shorter than a full 60-second trial by not iterating over it.

ii.
```python
def camera_frame_index(tstamps, n_frames):
    ifi = np.diff(tstamps)
    dt = np.median(ifi)
    n_missing = np.round(ifi / dt).astype(int) - 1
    idx = np.concatenate([[0], np.cumsum(1 + n_missing)])
    assert idx[-1] == n_frames - 1, 'could not reconstruct dropped camera frames'
    assert n_missing.sum() == n_frames - ncam
    return idx

def interp_nan(x):
    bad = np.isnan(x)
    ...
    x[bad] = np.interp(t[bad], t[~bad], x[~bad])
    return x

me[0] = np.nan
me = interp_nan(me)
```

iii. In trajectory step 49, the AI said it verified dropped-frame recovery in affected sessions, then linearly interpolated the missing motion-energy values, and treated frame 0 as missing because motion energy is undefined there by construction.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive computation in the AI's code is the per-session fluorescence baseline correction inside `f_processing()`, especially the Gaussian smoothing and the large 60-second min/max filtering across every neuron and every frame. The rest of the work is mainly array loading, interpolation, binning, and trial slicing.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    Flow = gaussian_filter(Fc, [0., sig_baseline])
    Flow = minimum_filter1d(Flow, win)
    Flow = maximum_filter1d(Flow, win)
    return Fc - Flow
```

iii. The trajectory does not contain an explicit performance discussion. The closest justification is in step 49, where the AI emphasized reimplementing Track2p's `F_processing`, which is the heaviest operation in the script.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python-level loop that could be reduced is the per-trial assembly loop, which repeatedly slices and appends one trial at a time. The subject/session loops are structurally necessary. The AI already avoided a slower frame-insertion loop by using vectorized timestamp reconstruction and `np.interp`.

ii.
```python
for k in range(n_trials):
    sl = slice(k * BINS_PER_TRIAL, (k + 1) * BINS_PER_TRIAL)
    ntr.append(np.ascontiguousarray(dff[:, sl]))
    ninp.append(t[sl][None, :].astype(np.float32))
    nout.append(cls[sl][None, :])
```

iii. The trajectory does not explicitly discuss vectorization. Implicitly, step 49 shows the AI chose an array-based dropped-frame repair strategy rather than repeated insertions.

## 6-c. What processing does the code repeat multiple times?

i. The AI recomputes the motion-energy percentile edges a second time when filling `session_info`, even though the first computation already happened inside `discretize_quintiles()`. It also traverses the trial boundaries separately to build neural, input, and output trial lists.

ii.
```python
cls = discretize_quintiles(me)
...
'motion_energy_quintile_edges': [
    float(v) for v in np.percentile(
        me, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1])],
```

iii. The trajectory does not provide an explicit justification for this repeated work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra validation and metadata work that the downstream decoder does not use: it loads `iscell.npy` only to assert all ROIs are cells, computes and stores detailed `session_info`, computes `bin_durations` only to summarize an average bin size, and carries timestamp-repair details into metadata even though the decoder only consumes `neural`, `input`, and `output`.

ii.
```python
iscell = np.load(os.path.join(s2p, 'iscell.npy'))
assert np.all(iscell[:, 0] == 1), 'unexpected non-cell ROI in Track2p output'

subject_idx, brain_region_idx, session_info = [], [], []
bin_durations = []
...
bin_durations.append(dt * FRAMES_PER_BIN)
...
session_info.append({
    'subject': subj,
    'date': sess.split('_')[0],
    'day_index': di,
    'n_neurons': int(dff.shape[0]),
    'n_trials': int(n_trials),
    'session_duration_s': float(t[-1]),
    'motion_energy_quintile_edges': [
        float(v) for v in np.percentile(
            me, np.linspace(0, 100, N_OUTPUT_BINS + 1)[1:-1])],
})
```

iii. The trajectory does not explicitly justify this extra bookkeeping; step 49 presents it as documentation and validation metadata rather than decoder-critical computation.
