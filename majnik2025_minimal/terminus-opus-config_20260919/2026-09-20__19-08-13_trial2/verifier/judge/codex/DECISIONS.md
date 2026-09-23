# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads subjects by globbing all `/app/data/jm*` directories, then loads every subdirectory under each subject as a session. Within each session it loads Suite2p fluorescence (`F.npy`, `Fneu.npy`), session metadata (`ops.npy`), and behavior files (`motion_energy_glob.npy`, `interframe_int.npy`). Trials are not loaded directly; they are created later by cutting each processed session into 60 s blocks.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))

for si, sub in enumerate(subjects):
    sub_dir = os.path.join(DATA_DIR, sub)
    sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
    ...
    for sd in sess_dirs:
        s2p = os.path.join(sd, 'suite2p', 'plane0')
        F = np.load(os.path.join(s2p, 'F.npy'))[keep]
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
        ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
        ...
        mv = os.path.join(sd, 'move_deve')
        me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
        ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
```

iii. In the trajectory, the AI first explored `/app/data`, identified 6 `jm*` mouse folders, listed their session subdirectories, and inspected the available files and array shapes before writing the converter. It explicitly concluded that the dataset structure was “subjects -> session directories -> suite2p + move_deve files” and then implemented loading around that structure.

## 1-b. How are the data split into subjects?

i. Subjects are defined as all directories in `/app/data` whose names start with `jm`, sorted lexicographically.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))
```

iii. The trajectory shows the AI listing `/app/data`, seeing the `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046` folders, and deciding that each `jm*` folder represented one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory, sorted lexicographically. Each subject-session pair becomes one output session.

ii.
```python
sub_dir = os.path.join(DATA_DIR, sub)
sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
```

iii. In the trajectory, the AI listed the session folders for each mouse and noted that each dated subdirectory corresponded to one daily recording. It then used that same directory boundary as the session boundary in the output.

## 1-d. How are the data split into trials?

i. The AI treats each session as a continuous recording and divides it into consecutive, non-overlapping 60 s trials after temporal binning. The number of bins per trial is computed from the imaging bin duration, and only full trials are kept.

ii.
```python
bin_sec = BIN_FRAMES / fs
...
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
neural_sess, input_sess, output_sess = [], [], []
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
    input_sess.append(tvec[sl][None, :].astype(np.float32))
    output_sess.append(me_cat[sl][None, :])
```

iii. The trajectory states that there is “no natural trial structure” and that sessions should be chunked into 60 s blocks for the decoder. The AI also inspected session lengths and noted that sessions were 20 or 30 min long, which made this fixed-length split straightforward.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply any explicit trial-quality filtering. It keeps all full 60 s blocks produced from each session.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    ...
```

iii. The trajectory contains no discussion of trial exclusion criteria. The AI focused its quality control on neuron filtering and dropped camera frames, not on removing individual trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from Suite2p `F.npy` and `Fneu.npy`, and also reads `ops.npy` to get the sampling rate and frame count needed for downstream alignment and trial construction.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))[keep]
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
fs = float(ops['fs'])
nframes = int(ops['nframes'])
```

iii. In the trajectory, the AI inspected Suite2p outputs, checked that `iscell.npy` was already curated, and then decided that `F.npy`/`Fneu.npy` were the raw fluorescence sources while `ops.npy` supplied timing metadata.

## 2-b. How is the `neural` data processed?

i. The AI applies a custom Track2p-style baseline correction function `f_processing` to `F` and `Fneu`, using `neucoeff=0.0` and a `maximin` baseline built from Gaussian smoothing plus minimum/maximum filters. After that it averages traces into 10-frame bins and z-scores each neuron within each session.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    win = int(win_baseline * fs)
    if baseline == 'maximin':
        Flow = gaussian_filter(Fc, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    else:
        Flow = 0.
    return Fc - Flow

...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
dff_b = bin_time(dff, BIN_FRAMES)
dff_b = (dff_b - dff_b.mean(axis=1, keepdims=True)) / dff_b.std(axis=1, keepdims=True)
```

iii. This was one of the AI’s clearest explicit decisions in the trajectory. It searched the Track2p repository, found `track2p/gui/data_management.py:F_processing`, concluded that the paper’s intended processing was Track2p’s baseline correction rather than Suite2p’s `dcnv.preprocess`, and additionally justified per-session z-scoring by citing Track2p raster preprocessing and cross-day scale comparability.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI drops neurons whose `F.npy` trace is identically zero or has zero standard deviation on any day for a given mouse. Because rows are treated as the same tracked neuron across days, once a neuron is marked bad for one session it is removed from all sessions of that mouse.

ii.
```python
# --- first pass: find ROIs that are all-zero on any day of this mouse ---
bad = None
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
    bad = z if bad is None else (bad | z)
keep = ~bad

...
F = np.load(os.path.join(s2p, 'F.npy'))[keep]
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
```

iii. The trajectory shows the AI explicitly checking `iscell.npy`, zero-variance traces, and the Track2p GUI logic for removing “zero rows” across days. It then justified this as cross-day curation of tracked neurons rather than session-by-session ROI filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats each 60 s block boundary as the temporal alignment event for a trial. Neural data are aligned by slicing each session’s binned trace into consecutive 60 s windows; metadata describes the alignment event as the start of each 60 s block.

ii.
```python
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))

...
'metadata': {
    ...
    'temporal_alignment_event': (
        'start of each 60 s block of the continuous recording (blocks are cut '
        'consecutively from the session onset; there is no trial structure in this '
        'spontaneous-activity experiment)'),
    'off_start': 0.0,
    'off_end': 60.0,
```

iii. In the trajectory, the AI reasoned that there was no real event structure in the experiment and therefore used artificial 60 s blocks as the trial alignment unit. It framed the dataset as a continuous spontaneous recording rather than an event-locked task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI averages both neural and motion-energy traces into non-overlapping bins of 10 imaging frames. With 30 Hz data this produces 3 Hz signals, or 333.33 ms per time bin.

ii.
```python
BIN_FRAMES = 10

def bin_time(x, bin_size):
    T = x.shape[-1]
    nb = T // bin_size
    x = x[..., :nb * bin_size]
    return x.reshape(*x.shape[:-1], nb, bin_size).mean(axis=-1)

...
dff_b = bin_time(dff, BIN_FRAMES)
me_b = bin_time(me, BIN_FRAMES)

...
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The trajectory cites the methods text stating that decoding used averages over bins of 10 consecutive timestamps. The AI adopted that language directly and treated 10-frame averaging as the paper-mandated denoising step.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. The time input is not read from an explicit timestamp variable. It is derived from the number of post-binning samples and the imaging frame rate `fs` read from `ops.npy`.

ii.
```python
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
fs = float(ops['fs'])
...
nbins = dff_b.shape[1]
bin_sec = BIN_FRAMES / fs
tvec = (np.arange(nbins) + 0.5) * bin_sec
```

iii. The trajectory shows the AI inspecting `ops.npy` to confirm 30 Hz acquisition and then deciding to synthesize time from the bin index because no dedicated experiment-time variable was needed once all streams were placed on the imaging frame grid.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The AI computes one scalar time value per binned sample using the center of each 10-frame bin: `(bin index + 0.5) * bin_sec`. It does not use raw camera timestamps.

ii.
```python
bin_sec = BIN_FRAMES / fs

# time from session start (bin centres), in seconds
tvec = (np.arange(nbins) + 0.5) * bin_sec
```

iii. The trajectory does not contain a long separate justification for bin-center timing, but it does show the AI reasoning that signals should share a common binned time base after 10-frame averaging. Choosing bin centers is consistent with that interpretation.

## 3-c. How is `input` *Time from start of experiment* aligned with the neural data?

i. The AI first constructs a session-wide time vector on the same binned grid as the neural data, then slices it with the exact same trial boundaries as `neural`, so each time series sample lines up one-for-one with the corresponding neural bin.

ii.
```python
tvec = (np.arange(nbins) + 0.5) * bin_sec

for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
    input_sess.append(tvec[sl][None, :].astype(np.float32))
```

iii. The trajectory repeatedly emphasizes a shared imaging-frame grid: the AI first aligns motion energy to imaging frames, then bins both streams together, then slices all modalities with identical trial windows. That is the AI’s justification for temporal alignment.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `move_deve/motion_energy_glob.npy`, with `move_deve/interframe_int.npy` used to locate missing camera frames before interpolation.

ii.
```python
mv = os.path.join(sd, 'move_deve')
me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
me = align_motion_energy(me, ifi, nframes)
```

iii. In the trajectory, the AI inspected the motion-energy files, checked sessions where the motion trace was shorter than the neural recording, and concluded that `interframe_int.npy` must be used to repair dropped video frames before decoding.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns motion energy to the imaging frame count by inserting missing samples at positions inferred from unusually long interframe intervals, linearly interpolates those inserted samples, averages the repaired signal into 10-frame bins, and finally discretizes the binned motion energy per session.

ii.
```python
def align_motion_energy(me, ifi, nframes):
    me = me.astype(np.float64)
    gap = int(nframes - len(me))
    if gap <= 0:
        return me[:nframes]
    med = np.median(ifi)
    nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
    ...
    out = [me[0]]
    for i in range(len(ifi)):
        for _ in range(int(nmiss[i])):
            out.append(np.nan)
        out.append(me[i + 1])
    arr = np.array(out, dtype=np.float64)
    nanmask = np.isnan(arr)
    if nanmask.any():
        idx = np.arange(len(arr))
        arr[nanmask] = np.interp(idx[nanmask], idx[~nanmask], arr[~nanmask])
    if len(arr) > nframes:
        arr = arr[:nframes]
    elif len(arr) < nframes:
        arr = np.concatenate([arr, np.full(nframes - len(arr), arr[-1])])
    return arr

...
me_b = bin_time(me, BIN_FRAMES)
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The trajectory documents that the AI first tried a simpler dropped-frame rule, then later patched `align_motion_energy` after noticing that some sessions would be shifted if it inserted frames whenever interframe intervals were long even when the total lengths already matched. The resulting justification was: use the microscope-triggered camera assumption, only repair actual length gaps, and interpolate missing frames.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI discretizes binned motion energy into 5 equal-percentile categories within each session by computing the 20th, 40th, 60th, and 80th percentile thresholds and passing the values through `np.digitize`.

ii.
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The trajectory states that the decoder output should be “motion energy discretized into 5 equal-percentile bins (quintiles) computed per session,” and the AI implemented exactly that decision.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes that camera frames and imaging frames are synchronized one-for-one because the camera was triggered by the microscope. It repairs missing camera frames to restore the imaging-frame length, bins motion energy with the same 10-frame averaging used for neural data, and slices both modalities with identical trial boundaries.

ii.
```python
me = align_motion_energy(me, ifi, nframes)

# denoise by averaging bins of 10 frames
dff_b = bin_time(dff, BIN_FRAMES)
me_b = bin_time(me, BIN_FRAMES)

for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
    output_sess.append(me_cat[sl][None, :])
```

iii. The trajectory explicitly says the camera was “triggered by the microscope,” so the AI treated motion energy as living on the same frame grid as imaging except for dropped frames. That assumption drove both the interpolation and the shared binning/slicing scheme.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly handles missing behavior samples. If motion energy is shorter than the neural frame count, it infers how many samples to insert from the length mismatch, chooses insertion points using interframe intervals, interpolates the missing values, and then truncates or pads the repaired trace to exactly `nframes`. It also removes neurons with zero or constant fluorescence traces.

ii.
```python
gap = int(nframes - len(me))
if gap <= 0:
    return me[:nframes]
...
if len(arr) > nframes:
    arr = arr[:nframes]
elif len(arr) < nframes:
    arr = np.concatenate([arr, np.full(nframes - len(arr), arr[-1])])

...
z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
bad = z if bad is None else (bad | z)
```

iii. The trajectory shows the AI actively investigating sessions with missing camera frames, validating that interframe intervals exposed frame drops, and then revising the interpolation logic once it noticed a possible misalignment edge case. It also inspected zero-variance neurons and decided to remove them across all days of a mouse.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the AI code are likely the repeated loading of full `F.npy` arrays, the per-session baseline correction in `f_processing` (Gaussian smoothing plus min/max filters over long traces), and the extra first pass over every session for cross-day zero-ROI detection.

ii.
```python
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
    bad = z if bad is None else (bad | z)

...
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
```

iii. The trajectory does not contain an explicit runtime analysis, so this is inferred from the code structure. The closest justification is that the AI deliberately added the first-pass ROI scan because it believed cross-day neuron curation was important enough to justify the extra pass.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested insertion loops in `align_motion_energy` could be vectorized or replaced with direct index construction. The trial-assembly loop also repeatedly appends slices one trial at a time, and the first-pass neuron screen reloads each session separately.

ii.
```python
for i in order:
    if remaining <= 0:
        break
    take = min(nmiss[i], remaining)
    keepmask[i] = take
    remaining -= take

out = [me[0]]
for i in range(len(ifi)):
    for _ in range(int(nmiss[i])):
        out.append(np.nan)
    out.append(me[i + 1])

...
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
```

iii. The trajectory does not explicitly discuss vectorization, but it does show the AI revisiting `align_motion_energy` as a fragile part of the code. That same section is also the clearest candidate for vectorization.

## 6-c. What processing does the code repeat multiple times?

i. The AI repeats loading `F.npy` for every session twice: once in the subject-level first pass to detect bad neurons and again in the main per-session conversion pass. It also recomputes `sess_dirs.index(sd)` inside the inner session loop when building metadata.

ii.
```python
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
    bad = z if bad is None else (bad | z)

...
for sd in sess_dirs:
    s2p = os.path.join(sd, 'suite2p', 'plane0')
    F = np.load(os.path.join(s2p, 'F.npy'))[keep]
    ...
    session_info.append({
        ...
        'day_index': sess_dirs.index(sd),
```

iii. The trajectory justifies the repeated first pass as part of the cross-day neuron curation strategy. It does not mention the repeated `sess_dirs.index(sd)` lookup, which appears to be incidental rather than intentional.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `Fneu.npy` even though its chosen `f_processing` call uses `neucoeff=0.0`, so neuropil data do not affect the result. It also computes and stores detailed `session_info` metadata that are not used by the downstream decoder, and it spends time on cross-day ROI curation that the human reference solution does not require.

ii.
```python
def f_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin',
                 sig_baseline=10.0, win_baseline=60.0):
    Fc = F - neucoeff * Fneu
    ...

...
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]

...
session_info.append({
    'subject': sub,
    'session': os.path.basename(sd),
    'day_index': sess_dirs.index(sd),
    'n_neurons': int(dff_b.shape[0]),
    'n_trials': int(ntrials),
    'imaging_rate_hz': fs,
})
```

iii. The trajectory makes the rationale for the extra ROI curation explicit, but it does not acknowledge that loading `Fneu.npy` with `neucoeff=0.0` is effectively unnecessary for the final numerical result. The extra metadata appear to be added for documentation rather than decoding.
