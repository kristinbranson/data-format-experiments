# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `jm*` subject directory, then dated session directories whose names begin with four digits. Unless `--sample` is supplied, it processes every discovered session. For each session it loads `F.npy`, `Fneu.npy`, `ops.npy`, `stat.npy`, `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`. `--full` is accepted but is effectively a no-op; full processing is the default.

ii.
```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
...
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. The notes identify six `jm*` mice and 41 dated recordings and state that the native layout is one Suite2p and one motion-data directory per daily session. They justify retaining observed 20- and 30-minute durations despite the paper's simplified 20-minute description.

## 1-b. How are the data split into subjects?

i. Each sorted directory beginning with `jm` is a subject. The complete discovered subject list is saved, while each retained session receives the index of its subject in that list.

ii.
```python
subject_names = [p.name for p in subjects]
...
data['subject_idx'].append(subject_names.index(item['subject']))
```

iii. The notes say this is a direct mapping from the native organization and matches the paper's six mice.

## 1-c. How are the data split into sessions?

i. Each dated subdirectory is treated as one session. A session is retained in the output only if at least two valid blocks remain.

ii.
```python
for subj in subject_names:
    for sess in sessions_by_subject[subj]:
        selected.append((subj, sess))
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The notes describe one recording day as one session and cite the continuous daily-recording organization. The two-block check satisfies the decoder format's minimum-trial requirement.

## 1-d. How are the data split into trials?

i. Continuous sessions are divided into non-overlapping 120-second blocks after 10-frame binning. Only complete blocks are kept, so trailing data shorter than 120 seconds are discarded.

ii.
```python
BLOCK_SECONDS = 120.0
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES
n_blocks = n_bins // BLOCK_BINS
for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
```

iii. The agent chose two-minute blocks because the paper used consecutive two-minute blocks for decoding cross-validation. Its notes call these the natural trial units, overlooking the task's explicit instruction to split sessions into 60-second trials.

## 1-e. How are trials filtered based on quality controls?

i. A block is discarded if it has fewer than 80% valid motion labels (with an absolute floor of 10 valid bins). Sessions with fewer than two retained blocks are then discarded. Complete-block segmentation also drops the final partial block.

ii.
```python
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
```

iii. The notes emphasize preserving missingness and the decoder's two-trial minimum, but do not justify the 80% threshold. They report that an earlier alignment attempt dropped nine sessions and that the final sample/full outputs passed format validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from Suite2p `F.npy` and `Fneu.npy`, with baseline parameters and neuropil coefficient read from `ops.npy`. `stat.npy` is loaded but not used.

ii.
```python
F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
dff = compute_dff(F, Fneu, ops)
```

iii. The notes identify these as released raw fluorescence traces and say dF/F should be computed for analysis. They also state that rows are already tracked and matched across days.

## 2-b. How is the `neural` data processed?

i. The code subtracts neuropil (`F - neucoeff*Fneu`), computes a per-neuron reflected moving average over the configured baseline window, takes one percentile of that smoothed trace as a constant baseline for the whole neuron, clamps the baseline to `1e-3`, and calculates `(Fcorr-baseline)/baseline`. It then averages every 10 frames.

ii.
```python
Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
baseline = np.maximum(baseline, 1e-3)
dff = (Fcorr - baseline) / baseline
...
return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)
```

iii. The agent describes this as “Suite2p-consistent” neuropil-subtracted, baseline-normalized fluorescence and acknowledges in its notes that it is a simplified approximation instead of Suite2p internals. Ten-frame averaging is justified by the paper's decoding method.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROI filtering is performed. All rows of `F`/`Fneu` are retained; `stat.npy` is unused and `iscell.npy` is not loaded.

ii.
```python
dff = compute_dff(F, Fneu, ops)
...
'n_neurons': dff.shape[0],
```

iii. Although earlier notes discuss an `iscell` probability threshold, the final plan says the released matrices already contain cells tracked across all days in matched row order, so it uses those matrices as provided.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins are sliced into contiguous 120-second blocks aligned to each block's start. Metadata calls the alignment event the start of each two-minute block, with offsets 0 to 120 seconds.

ii.
```python
neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
...
'temporal_alignment_event': 'start of each 2-minute continuous recording block',
'off_start': 0.0,
'off_end': BLOCK_SECONDS,
```

iii. The agent relied on the paper's two-minute decoder splits and noted that the recordings have no native trial events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Ten 30-Hz frames are averaged per bin for both neural and motion data, producing 333.33-ms bins. Short tails that do not complete a ten-frame bin are discarded.

ii.
```python
FS = 30.0
BIN_FRAMES = 10
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS,
```

iii. The paper explicitly says neural and behavioral traces were denoised by averaging 10 consecutive timestamps; the notes use this as the rationale.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. Time is synthesized from the within-block bin index, `BIN_FRAMES`, and `FS`; it is not derived from recorded timestamps and does not include the block's session offset.

ii.
```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. The notes say the requested elapsed time was operationalized as time within the session/block, and sample checks report a repeated range of 0.0–119.7 seconds.

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. A zero-based sequence of 360 binned samples is multiplied by 10/30 seconds. It resets to zero for every two-minute block and is stored as float32 with shape `(1, time)`.

ii.
```python
np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS)
```

iii. The agent viewed block start as the temporal alignment event and constructed the time input to satisfy the requested decoder-input shape.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. Each generated 360-bin time vector is appended alongside the neural slice from the same block, so bin positions correspond exactly, though the values describe time within block rather than time from session/experiment start.

ii.
```python
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    sess_neural.append(nb.astype(np.float32))
    sess_input.append(tb.astype(np.float32))
```

iii. The notes report recomputing converted time vectors and matching them with `np.allclose`, validating implementation of the chosen within-block convention.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Output is based on `move_deve/motion_energy_glob.npy`. `tstamps.npy` and `interframe_int.npy` are loaded, but neither affects the final alignment or values.

ii.
```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
```

iii. The notes identify global motion energy as the requested behavioral signal and say timestamps/interframe intervals reveal missing camera frames.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Motion samples are copied by order into an imaging-length array, with any length deficit represented as trailing NaNs and excess truncated. Ten-frame NaN-aware means are computed, the trace is split into blocks, and categories are assigned. Missing labels in otherwise retained blocks are forward-filled, then backward-filled, with any remaining values set to class 0.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]
...
out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
...
yy[yy < 0] = 0
```

iii. The final notes say an earlier timestamp interpretation wrongly dropped nine sessions, so the agent switched to preserving sample order and padding/truncating per the data README. It describes missing frames as preserved before binning, although this places all missingness at the tail rather than at actual dropped-frame locations.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Four 20/40/60/80% quantile edges are computed once from all finite binned motion values across all complete blocks in the entire converted dataset. `np.digitize` maps values to five zero-based categories.

ii.
```python
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
...
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

iii. The plan explicitly chooses full-dataset equal-percentile bins to make global class frequencies balanced. The agent notes that per-session fractions may vary, despite the task explicitly requiring bins selected per session.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion is assumed to correspond by sample order to imaging frames. It is padded/truncated to neural length, independently binned in the same 10-frame groups, clipped to the shorter binned stream, and sliced with identical block indices. Actual internal dropped-frame positions are not reconstructed.

ii.
```python
n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
neural_binned = neural_binned[:, :n_bins]
motion_binned = motion_binned[:n_bins]
...
motion_blocks.append(motion_binned[s:e].astype(np.float32))
```

iii. The agent reasons that microscope-triggered acquisition makes the streams framewise and that `tstamps.npy` contains seconds rather than frame indices. After a failed approach, it considered order-preserving tail padding/truncation the appropriate resolution.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Short motion arrays are tail-padded with NaNs and long ones truncated. Bins use available finite samples; blocks below 80% valid labels are dropped. Remaining missing labels are nearest-filled (previous first, next second) and finally class 0. Partial 10-frame bins and partial 120-second blocks are discarded. There is no explicit handling of NaNs in neural data.

ii.
```python
aligned = np.full(nframes, np.nan, dtype=np.float32)
aligned[:m] = motion[:m]
...
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
yy[yy < 0] = 0
```

iii. Notes document missing camera frames and an alignment revision after nine sessions were lost. Decoder verification and raw-versus-converted checks are cited as validation, but no justification is given for treating all missing samples as a tail deficit.

## 6-a. What are the most time-consuming steps of the code?

i. The custom baseline computation is the principal compute cost: it loops over every neuron and convolves each full trace. Loading large arrays, holding all processed sessions, concatenating motion values, and optional plotting also add I/O, memory, and runtime costs.

ii.
```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
```

iii. The notes explicitly identify the per-neuron baseline loop as inefficient and estimate roughly 1.5–3 minutes for 41 sessions; plotting is limited to two sessions.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron smoothing/percentile loop is the clearest vectorization candidate. The loops creating blocks and filling missing labels could also be replaced by reshaping/index-based operations, though they are comparatively small.

ii.
```python
for i in range(Fcorr.shape[0]):
    ...
for b in range(n_blocks):
    ...
for i in range(len(yy)):
    ...
for i in range(len(yy)-1, -1, -1):
    ...
```

iii. The agent only calls out the baseline loop in its efficiency notes; it says neural and motion frame binning were vectorized.

## 6-c. What processing does the code repeat multiple times?

i. Session loading, neural baseline processing, alignment, binning, and block construction repeat for each of 41 sessions. Identical time vectors are regenerated for every block. Motion blocks are traversed once to collect quantile data and again to discretize/assemble output.

ii.
```python
for idx, (subj, sessdir) in enumerate(selected):
    ...
for mb in motion_blocks:
    all_motion_values.append(...)
...
for nb, mb, tb in zip(...):
    y = np.digitize(mb, edges, right=False)
```

iii. The notes acknowledge that all sessions are prepared before global discretization, a two-pass design required by the agent's dataset-global thresholds, but do not otherwise discuss repeated processing.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `stat.npy`, `tstamps.npy`, and `interframe_int.npy` are loaded but unused. Raw arrays and diagnostics are processed for optional plots only when requested. Data in incomplete 10-frame bins, incomplete 120-second blocks, and rejected blocks are computed/read but discarded. `math` is imported but unused.

ii.
```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

iii. The notes initially intended timestamps/interframe intervals to locate missing frames and `stat`/cell metadata to support curation, but the final implementation instead assumes sample order and already-curated tracked rows. These unused loads are not acknowledged in the final efficiency review.
