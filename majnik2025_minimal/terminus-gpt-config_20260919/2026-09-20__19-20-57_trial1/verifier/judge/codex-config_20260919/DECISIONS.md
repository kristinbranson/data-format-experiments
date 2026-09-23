# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every `jm*` subject directory and every directory immediately below each subject, in sorted order. For each daily session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `motion_energy_glob.npy`, and motion timestamps. It processes all sessions unless the script itself is edited; there is no sample mode.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(md / 'tstamps.npy').astype(np.float64)
```

iii. The trajectory says the data README identified six mice, daily session directories, supplied Track2p-curated neurons, and processed global motion energy. The agent explicitly decided to include “all six subjects and all daily sessions.”

## 1-b. How are the data split into subjects?

i. Each sorted directory whose name matches `jm*` is one subject. Its enumeration index is recorded once per session in `subject_idx`.

ii.
```python
subjects = sorted(p.name for p in ROOT.glob('jm*') if p.is_dir())
for si, subject in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The agent relied on the documented directory convention and observed six such mouse directories.

## 1-c. How are the data split into sessions?

i. Every immediate subdirectory of a subject is treated as a separate daily recording session; paths are sorted deterministically. Each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
sessions = sorted(p for p in (ROOT / subject).iterdir() if p.is_dir())
for session in sessions:
    ...
    neural.append(ns); inputs.append(ins); outputs.append(outs)
```

iii. The trajectory states that the README described daily sessions, and validation found 41 sessions across the six mice.

## 1-d. How are the data split into trials?

i. A continuous session is divided into consecutive, non-overlapping, complete 60-second blocks after 10-frame averaging. At 3 Hz this is 180 bins per trial. An incomplete final block is omitted.

ii.
```python
out_fs = fs / AVG_FRAMES
trial_bins = int(round(TRIAL_SECONDS * out_fs))
ntrials = dff.shape[1] // trial_bins
for tr in range(ntrials):
    a, b = tr * trial_bins, (tr + 1) * trial_bins
```

iii. The task explicitly requests 60-second trials. The agent noted that this continuous recording has no natural trials and chose complete non-overlapping blocks, yielding 20 or 30 trials for 20- or 30-minute sessions.

## 1-e. How are trials filtered based on quality controls?

i. There is no trial-level quality filter. All complete 60-second blocks are retained, while only the incomplete tail is discarded.

ii.
```python
ntrials = dff.shape[1] // trial_bins
for tr in range(ntrials):
```

iii. The trajectory reports consistent frame counts and does not identify any paper-defined trial exclusion criterion; these are artificial blocks rather than behavioral trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p plane-0 raw fluorescence `F.npy`, neuropil fluorescence `Fneu.npy`, and preprocessing parameters in `ops.npy`.

ii.
```python
F = np.load(plane / 'F.npy', mmap_mode='r')
Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
dff = maximin_dff(F, Fneu, ops)
```

iii. The notebook inspection convinced the agent that `F.npy` was raw fluorescence, not spikes or ready-made dF/F, and that the paper called for neuropil-corrected, baseline-corrected fluorescence.

## 2-b. How is the `neural` data processed?

i. The agent subtracts neuropil using `ops['neucoeff']` (default 0.7), Gaussian-smooths in time, estimates a maximin baseline with minimum then maximum filters, and computes `(corrected fluorescence - baseline) / baseline`. It then averages every 10 frames and casts trial matrices to contiguous `float32`.

ii.
```python
x = np.asarray(F, dtype=np.float32) - np.float32(ops.get('neucoeff', .7)) * np.asarray(Fneu, dtype=np.float32)
smooth = gaussian_filter1d(x, float(ops.get('sig_baseline', 10.0)), axis=1, mode='reflect')
win = int(round(float(ops.get('win_baseline', 60.0)) * float(ops['fs'])))
base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
x = (x - base) / denom
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
```

iii. The agent interpreted the methods as requiring dF/F and the paper’s decoding analysis as averaging 10 timestamps. It intended to reproduce Suite2p’s stored maximin-baseline settings rather than use deconvolved spikes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional ROI filtering is performed. Every row supplied in `F.npy` is used, based on the assumption that these files already contain only Track2p-curated cells detected with Suite2p probability above 0.5 and tracked across all days.

ii.
```python
dff = maximin_dff(F, Fneu, ops)
region_idx.append(np.zeros(F.shape[0], dtype=np.int64))
```

iii. The trajectory says inventory checks found `iscell[:,0]` was one for all supplied rows and the README described the matrices as restricted to neurons tracked across every day. Thus reapplying `iscell` or tracking filters was deemed unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are indexed from session start, then cut into consecutive 60-second blocks. Metadata describes alignment to the start of each block, with offsets 0 to 60 seconds; the input clock nevertheless remains absolute session elapsed time.

ii.
```python
a, b = tr * trial_bins, (tr + 1) * trial_bins
ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
'temporal_alignment_event': 'start of each consecutive 60-second recording block',
'off_start': 0.0, 'off_end': 60.0,
```

iii. The agent observed no stimulus-alignment event. Near completion it explicitly refined the metadata to say trials align to each 60-second block start while time input remains session-absolute.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The original 30 Hz streams are averaged in non-overlapping groups of 10 frames, producing 3 Hz data and 333.333 ms bins. A tail shorter than 10 source frames would be dropped.

ii.
```python
AVG_FRAMES = 10
usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
'time_bin_size': 1000.0 * AVG_FRAMES / 30.0,
```

iii. The agent cites the paper’s decoding method: both dF/F and behavior were denoised by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is generated from the temporally averaged bin index and the session’s Suite2p sampling rate, not read from a raw timestamp array.

ii.
```python
out_fs = fs / AVG_FRAMES
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
```

iii. The agent found a constant 30 Hz imaging rate, so a bin index divided by 3 Hz provides elapsed session time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based sequence over all rebinned session samples is divided by output sampling frequency and stored in seconds as `float32`. It is then sliced per trial without resetting at trial boundaries.

ii.
```python
elapsed = np.arange(dff.shape[1], dtype=np.float32) / np.float32(out_fs)
ins.append(elapsed[None, a:b].copy())
```

iii. This directly implements the requested time elapsed from the beginning of the session and keeps the clock continuous across artificial trials.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Time and neural activity use identical rebinned indices and identical `[a:b]` trial slices, yielding one time value per neural column.

ii.
```python
ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
ins.append(elapsed[None, a:b].copy())
```

iii. The trajectory emphasizes consistent 180-bin trial dimensions; deriving time from the neural bin index avoids cross-stream drift.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses the supplied global motion-energy trace `motion_energy_glob.npy` and camera timestamps `tstamps.npy`; the neural frame count supplies the target length/index.

ii.
```python
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
stamps = np.load(md / 'tstamps.npy').astype(np.float64)
motion = aligned_motion(session, F.shape[1])
```

iii. The README and notebook indicated that global motion energy was precomputed and that timestamp gaps reveal missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The script validates equal motion/timestamp lengths, reconstructs nominal camera-frame slots from timestamp differences, interpolates onto every imaging index if frames are missing, averages 10 consecutive values, and converts the session trace to quintile labels.

ii.
```python
dt = np.median(np.diff(stamps))
slots = np.rint((stamps - stamps[0]) / dt).astype(np.int64)
unique, idx = np.unique(slots, return_index=True)
return np.interp(np.arange(nframes), unique, motion[idx])
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
labels = quintile_labels(motion)
```

iii. The agent found occasional timestamp gaps approximately twice the nominal interval and chose interpolation at those missing slots, followed by the paper’s same 10-frame denoising.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20th/40th/60th/80th percentile edges are computed independently for each complete rebinned session. `np.searchsorted(..., side='right')` assigns integer categories 0 through 4.

ii.
```python
edges = np.quantile(x, [.2, .4, .6, .8])
return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. This follows the decoder request for five equal-percentile bins “selected per session.” The agent independently checked that the resulting session labels were balanced.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is interpolated onto the neural imaging-frame index, both streams are truncated to the same multiple of 10 and averaged over corresponding 10-frame groups, then both are sliced with the same trial boundaries.

ii.
```python
motion = aligned_motion(session, F.shape[1])
usable = (F.shape[1] // AVG_FRAMES) * AVG_FRAMES
dff = dff[:, :usable].reshape(F.shape[0], -1, AVG_FRAMES).mean(2, dtype=np.float32)
motion = motion[:usable].reshape(-1, AVG_FRAMES).mean(1)
ns.append(np.ascontiguousarray(dff[:, a:b], dtype=np.float32))
outs.append(labels[None, a:b].copy())
```

iii. The agent regarded the streams as synchronous except for dropped camera frames; timestamp-derived interpolation restores one behavior value per imaging frame before common rebinning.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. A motion/timestamp length mismatch raises an error. Missing camera slots are detected from timestamp gaps and linearly interpolated. A fallback rescales timestamp positions to the known neural index range when clock drift makes the final slot unexpected. Slots are clipped and deduplicated. Unexpected imaging rate or mismatched `F`/`Fneu` shapes also raise errors; incomplete temporal tails are discarded.

ii.
```python
if len(motion) != len(stamps):
    raise ValueError(f'motion/timestamp mismatch in {session}')
if slots[-1] not in (nframes - 1, nframes):
    slots = np.rint((stamps - stamps[0]) * (nframes - 1) / (stamps[-1] - stamps[0])).astype(np.int64)
slots = np.clip(slots, 0, nframes - 1)
unique, idx = np.unique(slots, return_index=True)
return np.interp(np.arange(nframes), unique, motion[idx])
if fs != 30 or F.shape != Fneu.shape:
    raise ValueError(...)
```

iii. The agent used timestamp evidence and the README’s permission to interpolate missing video frames, while adding explicit integrity checks instead of silently accepting malformed arrays.

## 6-a. What are the most time-consuming steps of the code?

i. The expensive computation is full-session neural preprocessing, especially Gaussian smoothing and the 1,800-frame minimum and maximum filters for every neuron. Loading large arrays and serializing the roughly 414 MB pickle are also substantial. The trajectory confirms full conversion ran long enough to require waiting and progress monitoring.

ii.
```python
smooth = gaussian_filter1d(x, ..., axis=1, mode='reflect')
base = minimum_filter1d(smooth, size=win, axis=1, mode='reflect')
base = maximum_filter1d(base, size=win, axis=1, mode='reflect')
with OUT.open('wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent’s trajectory identifies baseline correction over every neuron and full-session trace as the central preprocessing operation; it also records processing progress by subject and a 414 MB output.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Session and subject loops are inherently tied to separate files and variable shapes. The per-trial loop could be replaced by reshaping each complete session into a trial axis, though list conversion/copies would still be needed for the required format. `session_info` also reloads the motion file merely to query its length.

ii.
```python
for si, subject in enumerate(subjects):
    for session in sessions:
        ...
        for tr in range(ntrials):
            a, b = tr * trial_bins, (tr + 1) * trial_bins
            ns.append(...)
```

iii. The trajectory does not discuss vectorizing loops. The implementation already vectorizes the costly filtering, rebinning, time generation, interpolation, and thresholding; only structural assembly remains loop-based.

## 6-c. What processing does the code repeat multiple times?

i. Each session’s motion-energy file is loaded once for processing and again only to record `motion_frames`. Trial-by-trial slicing and copying repeats for neural, input, and output. Otherwise preprocessing occurs once per session.

ii.
```python
motion = np.load(md / 'motion_energy_glob.npy').astype(np.float64)
...
'motion_frames': int(np.load(session/'move_deve'/'motion_energy_glob.npy', mmap_mode='r').shape[0]),
```

iii. The trajectory does not justify this duplicate load; it is a minor metadata convenience rather than a methodological choice.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and saves extensive `session_info` metadata that the decoder need not use, reloads motion energy to obtain one metadata shape, and creates copies/contiguous arrays during trial assembly. It also processes rebinned remainder samples that do not form a full 60-second trial; those samples affect session-wide quintile edges but are not stored as trials.

ii.
```python
labels = quintile_labels(motion)
ntrials = dff.shape[1] // trial_bins
...
session_info.append({...})
```

iii. The trajectory focuses on validation and traceability, so detailed metadata and defensive copies aid auditing, but it gives no explicit justification for processing the final incomplete-trial remainder before discarding it from decoder arrays.
