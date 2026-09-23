# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `/app/data/jm*` subject directories and every sorted subdirectory as a session. It loads `F.npy`, `Fneu.npy`, and `ops.npy` from each session's Suite2p `plane0` directory, and `motion_energy_glob.npy` plus `interframe_int.npy` from `move_deve`. It processes all six mice and 41 sessions.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))
for si, sub in enumerate(subjects):
    sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
    for sd in sess_dirs:
        F = np.load(os.path.join(s2p, 'F.npy'))[keep]
        Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
        ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
        me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
        ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
```

iii. The trajectory says the agent inspected the directory tree, notebook, README, array shapes, and `ops` fields and concluded there were six mice and 41 daily sessions using the standard subject/session layout.

## 1-b. How are the data split into subjects?

i. Each sorted directory whose basename matches `jm*` is treated as one mouse. Its enumeration index is stored once for every session in `subject_idx`.

ii.
```python
subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, 'jm*')))
for si, sub in enumerate(subjects):
    ...
    subject_idx.append(si)
```

iii. The agent inferred from the data layout and README that the six `jm*` folders are mouse identifiers.

## 1-c. How are the data split into sessions?

i. Every directory immediately under a mouse folder is a session. Sessions are sorted, and each becomes one outer-list element in `neural`, `input`, and `output`.

ii.
```python
sess_dirs = sorted([f.path for f in os.scandir(sub_dir) if f.is_dir()])
for sd in sess_dirs:
    ...
    neural_all.append(neural_sess)
    input_all.append(input_sess)
    output_all.append(output_sess)
```

iii. The trajectory identifies these as daily recordings and uses sorting for stable chronological/name order.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided from session onset into consecutive, non-overlapping 60-second blocks. At the effective 3 Hz sampling rate, each trial has 180 bins; any incomplete tail is implicitly discarded.

ii.
```python
bins_per_trial = int(round(TRIAL_SEC / bin_sec))
ntrials = nbins // bins_per_trial
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
    neural_sess.append(dff_b[:, sl].astype(np.float32))
```

iii. The agent noted that this spontaneous-activity experiment has no natural trials and followed the explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. No complete 60-second trial is quality-filtered. Only the trailing bins that cannot form a complete trial are omitted through floor division.

ii.
```python
ntrials = nbins // bins_per_trial
for t in range(ntrials):
    sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
```

iii. The agent found no trial-level QC in the paper or repository and retained all complete blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`; `ops.npy` supplies the frame rate and expected frame count. With the chosen coefficient of zero, `Fneu` is loaded but has no numerical effect.

ii.
```python
F = np.load(os.path.join(s2p, 'F.npy'))[keep]
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
ops = np.load(os.path.join(s2p, 'ops.npy'), allow_pickle=True).item()
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
```

iii. The agent traced the repository's `F_processing` helper and interpreted the released `F` rows as tracked ROIs already classified as cells.

## 2-b. How is the `neural` data processed?

i. It computes `Fc = F - 0.0*Fneu`, estimates a maximin baseline after Gaussian smoothing (`sig_baseline=10`) with a 60-second window, subtracts that baseline, averages non-overlapping groups of 10 frames, then z-scores each neuron across the whole session and casts trial slices to `float32`.

ii.
```python
Fc = F - neucoeff * Fneu
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
return Fc - Flow
...
dff_b = bin_time(dff, BIN_FRAMES)
dff_b = (dff_b - dff_b.mean(axis=1, keepdims=True)) / dff_b.std(axis=1, keepdims=True)
```

iii. The agent said coefficient 0 and maximin reproduce Track2p's GUI helper, 10-frame averaging follows the paper's decoder denoising, and z-scoring follows Track2p raster preprocessing and makes arbitrary fluorescence scales comparable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A neuron is removed from every session of a mouse if its raw `F` trace is all zero or has zero standard deviation in any session. The same cross-session mask preserves tracked-cell correspondence.

ii.
```python
bad = None
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
    z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
    bad = z if bad is None else (bad | z)
keep = ~bad
```

iii. The agent justified this as matching the Track2p GUI's removal of zero rows across days and preventing empty ROIs from entering the decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event. Blocks are indexed consecutively from session onset; metadata describes the alignment event as the start of each 60-second block, with offsets 0 to 60 seconds.

ii.
```python
sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
neural_sess.append(dff_b[:, sl].astype(np.float32))
...
'temporal_alignment_event': 'start of each 60 s block ...',
'off_start': 0.0,
'off_end': 60.0,
```

iii. The agent reasoned that continuous spontaneous recordings have no natural trial event, so artificial block onset is the meaningful alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural and motion traces are averaged in non-overlapping 10-frame bins. At 30 Hz this produces 333.33 ms bins (3 Hz).

ii.
```python
BIN_FRAMES = 10
dff_b = bin_time(dff, BIN_FRAMES)
me_b = bin_time(me, BIN_FRAMES)
'time_bin_size': 1000.0 * BIN_FRAMES / 30.0,
```

iii. The agent cites the paper's statement that decoding analyses denoised dF/F and behavior by averaging 10 consecutive timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the binned sample index and the session-specific imaging rate from `ops.npy`; it is not read from behavioral timestamps.

ii.
```python
fs = float(ops['fs'])
bin_sec = BIN_FRAMES / fs
tvec = (np.arange(nbins) + 0.5) * bin_sec
```

iii. The agent determined that imaging is regularly sampled at 30 Hz and therefore bin indices and `fs` provide session elapsed time.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The center of every 10-frame bin is converted to seconds: `(index + 0.5) * 10/fs`. The vector remains continuous across trials and is cast to `float32` when sliced.

ii.
```python
tvec = (np.arange(nbins) + 0.5) * bin_sec
input_sess.append(tvec[sl][None, :].astype(np.float32))
```

iii. The agent explicitly chose bin centers, which naturally timestamp averages over each 10-frame interval.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. A time value is created for every binned neural column, and identical trial slices are applied to both arrays.

ii.
```python
sl = slice(t * bins_per_trial, (t + 1) * bins_per_trial)
neural_sess.append(dff_b[:, sl].astype(np.float32))
input_sess.append(tvec[sl][None, :].astype(np.float32))
```

iii. The shared bin grid and slice ensure one-to-one temporal correspondence.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output comes from each session's precomputed `motion_energy_glob.npy`; `interframe_int.npy` identifies missing camera samples and `ops['nframes']` defines the target imaging grid length.

ii.
```python
me = np.load(os.path.join(mv, 'motion_energy_glob.npy'))
ifi = np.load(os.path.join(mv, 'interframe_int.npy'))
me = align_motion_energy(me, ifi, nframes)
```

iii. The agent relied on the README and acquisition design: the camera was microscope-triggered, and interval gaps reveal dropped video frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Missing samples are inserted and linearly interpolated onto the imaging frame grid, the signal is averaged over 10-frame bins, and session-specific percentile cut points are used to form five classes.

ii.
```python
me = align_motion_energy(me, ifi, nframes)
me_b = bin_time(me, BIN_FRAMES)
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
```

iii. The trajectory says interpolation follows the dataset README, while averaging before categorization follows the paper and avoids averaging categorical labels.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20th/40th/60th/80th percentile thresholds are computed separately within each session after temporal averaging. `np.digitize` assigns integer categories 0–4, named very low through very high.

ii.
```python
edges = np.percentile(me_b, np.linspace(0, 100, N_QUANTILES + 1)[1:-1])
me_cat = np.digitize(me_b, edges).astype(np.int64)
...
'output_values': [['very low', 'low', 'medium', 'high', 'very high']],
```

iii. This directly implements the requested five equal-percentile bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Because camera and microscope are nominally synchronous, motion samples are placed onto the imaging grid. The algorithm infers missing counts from interval/median ratios, limits insertions to the observed length gap, linearly interpolates inserted NaNs, truncates or edge-pads to `nframes`, then applies the same 10-frame binning and trial slices as neural data.

ii.
```python
gap = int(nframes - len(me))
nmiss = np.maximum(np.round(ifi / med).astype(int) - 1, 0)
...
arr[nanmask] = np.interp(idx[nanmask], idx[~nanmask], arr[~nanmask])
...
output_sess.append(me_cat[sl][None, :])
```

iii. The agent refined the implementation after testing: it inserts exactly the actual length gap at the largest inferred intervals so sessions with no length mismatch are not shifted.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Dropped video frames are interpolated. If motion is longer it is truncated; if still shorter after inferred insertion it is padded with the last value. Empty/constant neural ROIs are removed consistently across a mouse, and partial terminal trials are discarded.

ii.
```python
if len(arr) > nframes:
    arr = arr[:nframes]
elif len(arr) < nframes:
    arr = np.concatenate([arr, np.full(nframes - len(arr), arr[-1])])
...
z = np.all(F == 0, axis=1) | (F.std(axis=1) == 0)
ntrials = nbins // bins_per_trial
```

iii. The agent used the README's dropped-frame guidance, preserved matched-neuron dimensions across days, and preferred complete equal-length trials.

## 6-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading every full fluorescence array twice and applying Gaussian, minimum, and maximum filters over every neuron and frame. Converting all arrays and z-scoring also traverse the full data.

ii.
```python
for sd in sess_dirs:
    F = np.load(...)
...
Flow = gaussian_filter(Fc, [0., sig_baseline])
Flow = minimum_filter1d(Flow, win)
Flow = maximum_filter1d(Flow, win)
```

iii. The trajectory investigated preprocessing and then ran the full conversion repeatedly; it did not provide an explicit runtime profile. The baseline filters are the computationally heavy full-session operations.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The missing-frame reconstruction uses Python loops over every interframe interval and each inferred missing sample. Preallocating the final frame grid and filling/interpolating in bulk would be faster. Trial slicing loops are also mechanically vectorizable, though lists are required by the output format.

ii.
```python
out = [me[0]]
for i in range(len(ifi)):
    for _ in range(int(nmiss[i])):
        out.append(np.nan)
    out.append(me[i + 1])
```

iii. The agent did not explicitly discuss vectorization; its trajectory focused instead on making missing-frame insertion numerically correct.

## 6-c. What processing does the code repeat multiple times?

i. Each session's `F.npy` is loaded once to build the mouse-wide bad-ROI mask and again for actual processing. The entire conversion was also rerun after the interpolation patch, but that is development-time rather than final-script repetition.

ii.
```python
for sd in sess_dirs:
    F = np.load(os.path.join(sd, 'suite2p', 'plane0', 'F.npy'))
...
for sd in sess_dirs:
    F = np.load(os.path.join(s2p, 'F.npy'))[keep]
```

iii. The repeated load is a consequence of first establishing one consistent cross-session neuron mask before processing any session.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Fneu.npy` is loaded, converted to float64, and multiplied by zero, so it contributes nothing. Full `F` arrays are loaded during the QC pass and discarded before being loaded again. Rich `session_info` and descriptive metadata are saved but are not decoder features.

ii.
```python
Fneu = np.load(os.path.join(s2p, 'Fneu.npy'))[keep]
dff = f_processing(F.astype(np.float64), Fneu.astype(np.float64), fs)
# f_processing default: neucoeff=0.0
Fc = F - neucoeff * Fneu
```

iii. The agent retained `Fneu` to mirror the Track2p helper signature and added metadata for provenance; it did not explicitly identify these as discarded downstream work.
