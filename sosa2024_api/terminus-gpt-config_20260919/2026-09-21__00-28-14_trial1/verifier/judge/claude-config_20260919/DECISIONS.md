# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `.nwb` files under `/app/data` are discovered recursively with `Path.rglob('*.nwb')` and sorted, giving 152 files (11 subject directories `sub-m3` … `sub-m19`). Each file is one session and is opened exactly once with `pynwb.NWBHDF5IO` (no `h5py`). From each file the AI reads `nwb.subject.subject_id`, `nwb.identifier`, `nwb.session_id`, the eight behaviour series in `processing/behavior/BehavioralTimeSeries` (`trial_start`, `teleport`, `trial number`, `environment`, `position`, `speed`, `lick`, `reward_zone`), the `Reward` timestamps, `position.timestamps`, and every `RoiResponseSeries` in `processing/ophys/Deconvolved` (one per imaging plane) together with its `rois` `DynamicTableRegion` and the `iscell` column of the PlaneSegmentation table. Neural data are read lazily, per trial (`rs.data[start:start+n, :]`), rather than as one whole-session array. All 152 sessions / 12,216 trials / 138,678 accepted cells are retained; nothing is subsampled. `--sample` selects two files (one single-plane, one two-plane); `--full` (default) processes all of them.

ii.
```python
DATA_ROOT = Path('/app/data')

def choose_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    if not sample:
        return files
    single = files[0]
    multi = next(p for p in files if p.parent.name in ('sub-m17', 'sub-m18'))
    return [single, multi]
```
```python
def process_session(path: Path):
    t0 = time.time()
    with NWBHDF5IO(str(path), 'r') as io:
        nwb = io.read()
        subject = str(nwb.subject.subject_id)
        bts = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
        ...
        names = ['trial_start', 'teleport', 'trial number', 'environment',
                 'position', 'speed', 'lick', 'reward_zone']
        beh = {k: np.asarray(bts[k].data[:]) for k in names}
```
```python
def load_neural_trial(series_info, start: int, end: int, factor: int) -> np.ndarray:
    """Read accepted cells from all planes and return cells x common-rate time."""
    planes = []
    n = ((end - start) // factor) * factor
    for rs, accepted_cols in series_info:
        # pynwb exposes the NWB Data object; no direct h5py loading is used.
        block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
        block = block[:, accepted_cols]
        ...
        planes.append(block.T)
    return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES Step 2/Step 4: "Data are 152 NWB files … organized as `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`"; "All inspection used `pynwb.NWBHDF5IO`; h5py was not used." The AI decided to keep all 152 available sessions rather than only the paper's 77 switch sessions, arguing "Decoder outputs are defined on switch and stay sessions; full conversion means all provided data. Report switch-only checks separately." It reconciled the paper's 12,376-trial denominator with the 12,216 trials found by noting the two missing m11 stay sessions (2 × 80 trials). It also fixed an early bug where it inspected only `plane0` of the two-plane animals and undercounted cells: "Looking only at `plane0` undercounted m17/m18 cells. Fixed by iterating all Deconvolved series; paper per-animal switch totals then matched exactly" (73,512 switch-session cells).

## 1-b. How are the data split into subjects?

i. The subject label for a session is read from the NWB metadata field `nwb.subject.subject_id` (not from the directory or file name). The unique set is then sorted numerically on the digits of the id, producing `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']` (11 subjects), and `subject_idx` is the index of each session's subject into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
...
subjects = sorted(set(x['subject'] for x in infos), key=lambda z: int(re.sub(r'\D','',z)))
subject_idx = np.array([subjects.index(x['subject']) for x in infos], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2: "Subjects | 11 (m3, m4, m7, m11--m15, m17--m19)", cross-checked against the paper's "n = 11 mice" for the switch task (Step 3/Step 4 consistency table: "Subjects | 11 switch mice | IDs m3,m4,m7,m11--m15,m17--m19 | 11 | 11 | Exact"). Reading the id from the NWB subject object rather than parsing the path is justified as using the file's own authoritative metadata.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are kept in sorted-path order (subject directory, then `ses-NN`); `session_id` and `identifier` are stored per session in `metadata['session_info']`. No pooling or alignment of cells across days is attempted (each session's neurons are treated as an independent population). All 152 sessions are kept — the 77 "switch" sessions and the 75 available "stay" sessions.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for i, p in enumerate(files, 1):
    ns, xs, ys, info = process_session(p)
    neural.append(ns); inputs.append(xs); outputs.append(ys); infos.append(info)
...
info = {'path': str(path), 'subject': subject, 'session_id': str(nwb.session_id),
        'identifier': str(nwb.identifier), ...}
```

iii. CONVERSION_NOTES Step 4: "77 switch + 75 stay = 152 … All 77 switch sessions are present. m11 lacks two stay files; use all 152 available sessions for the requested full dataset because outputs are defined for both switch and stay trials." Step 9 confirms 14 sessions for every mouse except m11 (12).

## 1-d. How are the data split into trials?

i. Trial boundaries come from the two behaviour marker channels. Trial starts are the sample indices where `trial_start > 0`; trial ends are the sample indices where `teleport > 0`. Each start is paired with the next unused teleport index by `pair_bounds()`, and the trial is the **half-open** interval `[start, teleport)` — the teleport frame itself (the inter-trial-interval / position-reset frame) is excluded. The script hard-fails if the counts of starts, ends and pairs do not all agree. This yields 12,216 trials, with T mean 216.78, median 197.49, min 96, max 3,359 samples — identical to the human reference's trial statistics.

ii.
```python
def pair_bounds(starts: np.ndarray, ends: np.ndarray) -> list[tuple[int, int]]:
    """Pair each start with the next unused teleport; return half-open [start,end)."""
    pairs, j = [], 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j >= len(ends):
            break
        pairs.append((int(s), int(ends[j])))
        j += 1
    return pairs
```
```python
starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
pairs = pair_bounds(starts, ends)
if len(pairs) != len(starts) or len(starts) != len(ends):
    raise ValueError(f'{path}: unmatched trial markers {len(starts)}/{len(ends)}/{len(pairs)}')
...
for qi, (s, e) in enumerate(pairs):
    ...
    neural = load_neural_trial(series_info, s, e, factor)
    pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. CONVERSION_NOTES Step 2: "NWB `trials` tables are absent. Valid trial bounds are explicit framewise `trial_start > 0` through the next `teleport > 0`, matching reference code. All 12,216 starts paired uniquely with 12,216 ends." Step 5 key decision 2: "Use `[trial_start, teleport)`. Teleport is explicitly ITI entry and its frame can contain reset position; exclusion avoids an off-by-one reset artifact." Step 10 further audited the longest trial (m4 ses-04 trial 39, 3,359 frames / 216.6 s) by reloading the raw NWB and confirmed it is genuine behaviour (a ~2 min pause near 9 cm) rather than a mispaired marker.

## 1-e. How are trials filtered based on quality controls?

i. **No trials are removed.** All 12,216 start/teleport pairs are converted. The only trial-level quality control is diagnostic: the AI reimplements the paper's lick-circuit-artifact criterion (>30% of a trial's samples with a cumulative lick count > 2), which flags 81 trials — exactly matching the paper's "81 erroneous-lick trials out of 12,376". Those trials are **retained**, the flag is recorded per trial in `metadata['session_info'][…]['trial_info']` and counted in the log. There is no minimum-trial-length filter (the shortest trial found is 96 samples). Hard errors (not filters) are raised for unmatched trial markers, a non-constant or out-of-range `environment`, zero accepted cells in a plane, non-finite converted values, or a zero-length trial.

ii.
```python
raw_lick = beh['lick'][s:e]
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
lick_artifacts += int(lick_artifact)
...
trial_info.append({'trial_number': trial_num, 'start': s, 'end': e,
                   'environment': env, 'zone': zone, 'rewarded': rewarded,
                   'lick_artifact': lick_artifact, 'T': T})
```
```python
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(f'{path}: invalid/nonconstant environment in trial {qi}')
...
if not np.isfinite(neural).all() or not np.isfinite(inp).all():
    raise ValueError(f'{path}: non-finite converted data')
```

iii. CONVERSION_NOTES Step 4/Step 5 decision 7: "Detect and report the paper criterion (>30% native trial samples with lick count >2). It reproduces 81 available trials. Retain trials and binary lick because target format has no missing-output mask and deleting entire trials would discard valid neural and other outputs." Step 3 records the paper's "81/12,376 = ~0.65% trials" as the target. Step 10: "81 trials meet paper criterion … the count is recorded in metadata/logs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are taken **directly from the NWB `processing/ophys/Deconvolved` `RoiResponseSeries`** (one per imaging plane), together with the `iscell` column of the PlaneSegmentation table reached through each series' `rois` `DynamicTableRegion`. The raw `Fluorescence` (F) and `Neuropil` (Fneu) series that are also present in every file are **not read at all**. No dF/F is computed. Note that this NWB `Deconvolved` array is Suite2p's own deconvolution of raw fluorescence (values on the fluorescence scale, up to ~9.6e3, non-zero throughout the inter-trial/teleport periods), which is a different signal from the paper's `events`, which the reference code derives from F and Fneu via `preprocessing.dff(..., deconvolve=True)`.

ii.
```python
deconv = nwb.processing['ophys']['Deconvolved'].roi_response_series
series_info = []
rates = []
for key in sorted(deconv.keys()):
    rs = deconv[key]
    region = np.asarray(rs.rois.data[:], dtype=np.int64)
    labels = np.asarray(rs.rois.table['iscell'][:])
    keep = labels[region, 0] > 0.5
    accepted_local_cols = np.flatnonzero(keep)
    if not len(accepted_local_cols):
        raise ValueError(f'{path}: no accepted cells in {key}')
    series_info.append((rs, accepted_local_cols))
    rates.append(float(rs.rate))
```

iii. CONVERSION_NOTES Step 1: "The primary neural representation for paper analyses is deconvolved calcium `events` (rather than raw fluorescence). dF/F is also available and used for visualization/other analyses. **Conversion should prefer the corresponding processed neural series already exported in NWB when available rather than recomputing from raw fluorescence.**" Step 4 discrepancy table resolves: "canonical cleaned analysis uses `events`, maximin baseline | NWB exports Deconvolved, Fluorescence, Neuropil | paper mostly uses deconvolved calcium activity | **Use NWB Deconvolved series; no dF/F recomputation.**" Step 5: `metadata['neural_signal'] = 'Suite2p deconvolved calcium events, accepted cells only'`. The trajectory contains no reasoning step that compares the NWB `Deconvolved` array against the paper's `dff()` output.

## 2-b. How is the `neural` data processed?

i. Essentially no signal processing is applied. For each trial the per-plane `Deconvolved` block `[start:end, :]` is read, cast to `float32`, restricted to the accepted (`iscell`) columns, transposed to (cells × time), and the planes are concatenated along the cell axis. There is no neuropil subtraction (`F - 0.7·Fneu`), no per-trial maximin baseline (Gaussian σ=15 smoothing, 300-sample min then max filter = the Methods' 20 s window), no `(F − baseline)/|baseline|` normalisation, no 2-sample Gaussian smoothing, and no OASIS deconvolution at τ = 0.7 with `frame_rate/n_planes`. The per-mouse/per-day `keep_teleports` distinction from the reference's `teleport_metadata.py` is likewise not used. The only other transform is dtype (`float32`).

ii.
```python
block = np.asarray(rs.data[start:start+n, :], dtype=np.float32)
block = block[:, accepted_cols]
if factor > 1:
    block = block.reshape(n // factor, factor, block.shape[1]).sum(axis=1)
planes.append(block.T)
return np.concatenate(planes, axis=0).astype(np.float32, copy=False)
```
(`factor` is always 1 in the shipped code, so the `reshape/sum` branch never executes.)

iii. CONVERSION_NOTES Step 4: "Use NWB Deconvolved series; no dF/F recomputation." Step 10 Check 3: "Neuron filtering: reference/paper use Suite2p cells; conversion maps every plane's RoiResponseSeries region then applies `iscell[:,0]==1`. Exact paper total 73,512 on switch sessions." The AI's justification is entirely that the paper analyses "deconvolved activity" and the NWB already exports a series called `Deconvolved`; it never checked whether that array equals the paper's `events`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter only: Suite2p's manual curation label. For each `Deconvolved` plane series, the column→PlaneSegmentation-row mapping in `rs.rois.data` is applied, and only columns whose `iscell[:,0] > 0.5` are retained. Cells from all planes are then pooled. This leaves 138,678 cells overall (155–2,341 per session, mean 912.36), and exactly 73,512 cells over the 77 switch sessions. The paper's **second** curation step — excluding putative interneurons whose dF/F correlates with running speed at Pearson r > 0.5 (reference `is_putative_interneuron`) — is **not** applied. No place-cell selection and no speed-threshold sample exclusion are applied (deliberately, since the decoder must predict speed including <2 cm/s).

ii.
```python
region = np.asarray(rs.rois.data[:], dtype=np.int64)
labels = np.asarray(rs.rois.table['iscell'][:])
keep = labels[region, 0] > 0.5
accepted_local_cols = np.flatnonzero(keep)
```

iii. CONVERSION_NOTES Step 3 neuron-curation rules: "Use Suite2p `iscell[:,0] == 1` through each RoiResponseSeries region. Paper totals confirm 73,512 cells across the 77 switch sessions exactly when both planes are included. Do not restrict to place cells because the requested general decoder requires full neural populations." Step 1 lists `is_putative_interneuron` in the function table but labels it "**Optional** cell-class identification used in paper analyses", and Step 1 concludes "No generic additional ROI filtering should be invented: use the curated ROIs represented in the NWB neural series/ROI table." Step 4 explicitly rejects place-cell filtering and >2 cm/s speed filtering as "analysis-specific", but contains no row for interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The requested alignment event is trial start. Because the ophys rows and the behaviour samples are one-to-one on the same clock, alignment is achieved purely by slicing every stream with the same sample indices `[s, e)`, where `s` is the `trial_start` frame. Sample 0 of each trial's neural matrix is therefore the trial-start frame, and `input[0]` (time from trial start) begins at exactly 0.0 s. `metadata['temporal_alignment_event'] = 'trial start (entry to linear track)'`, `off_start = 0.0`, `off_end = None`. No pre-event window is included. The script asserts that neural, input and output all have the same T for every trial.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    ...
    neural = load_neural_trial(series_info, s, e, factor)
    pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
    speed = aggregate_behavior(beh['speed'][s:e], factor, 'mean').astype(np.float32)
    ...
    elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
    ...
    if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
        raise AssertionError('Trial stream lengths differ')
```
```python
'temporal_alignment_event': 'trial start (entry to linear track)',
'off_start': 0.0, 'off_end': None,
```

iii. CONVERSION_NOTES Step 10 Check 3: "Temporal alignment: both use imaging-frame-aligned VR/ophys samples. Explicit behavior timestamps are authoritative." Step 7 processing-plot review: "Neural events and all output streams span the same trial-start-aligned time axis. No teleport/reset frame is present." Step 10 Check 2 ran independent raw-NWB `np.testing.assert_allclose` spot checks of the neural matrix and all inputs/outputs for m11 ses-03 trial 5 (155×266), m17 ses-01 trial 30 (591×166) and m18 ses-08 trial 79 (2314×214).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is the native imaging frame interval: 1000 / 15.5078125 = **64.4836 ms** (`metadata['time_bin_size']`), identical for every trial and session. **No rebinning, resampling or downsampling is performed** (`factor` is hard-coded to 1 and the `aggregate_behavior`/`reshape-sum` machinery is therefore inert). The AI determined the effective rate from `1 / median(diff(position.timestamps))` and hard-fails if it is not within 0.2% of 15.5078125 Hz, deliberately ignoring the `rate` attribute stored on the `RoiResponseSeries`, which reads 31.015625 Hz on the two-plane animals (m17, m18) even though those series have one row per behaviour sample.

ii.
```python
TARGET_RATE = 15.5078125
BIN_MS = 1000.0 / TARGET_RATE
...
# NWB two-plane RoiResponseSeries metadata report 31.015625 Hz, but their
# rows are already one-to-one with behavior timestamps at 15.5078125 Hz.
# Use the synchronized behavior clock, not the inconsistent series rate.
pos_clock = np.asarray(bts['position'].timestamps[:], dtype=np.float64)
effective_rate = 1.0 / float(np.median(np.diff(pos_clock)))
if not np.isclose(effective_rate, TARGET_RATE, rtol=2e-3):
    raise ValueError(f'{path}: unsupported behavior-clock rate {effective_rate}')
factor = 1
```

iii. CONVERSION_NOTES Step 4: "Although two-plane RoiResponseSeries metadata report 31.015625 Hz, their row count equals the behavior row count and behavior timestamps are 15.5078125 Hz. Treat every row as one synchronized 64.484 ms sample; do not downsample." Step 7 records this as a bug the AI found and fixed: "Initial sample code trusted RoiResponseSeries `rate=31.015625` and downsampled m17/m18, producing implausibly short trials … Fixed by using behavior timestamps as authoritative." Cross-checked against the Methods' "0.0645 s imaging frame samples" (Step 3). (Note: the Step 4 narrative paragraph still contains a stale sentence, "Two-plane sessions are downsampled to the common paper frame interval", contradicting the corrected code.)

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is **not** read from any timestamp array. It is synthesised from the sample index within the trial and the constant `TARGET_RATE = 15.5078125` Hz. The behaviour timestamps (`bts['position'].timestamps`) are read, but only to verify the sampling rate and to define the reward-time window; they are not used to build the elapsed-time vector.

ii.
```python
pos_clock = np.asarray(bts['position'].timestamps[:], dtype=np.float64)
effective_rate = 1.0 / float(np.median(np.diff(pos_clock)))
...
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "elapsed samples from `trial_start` → `input[0]` time from trial start | `arange(T)/15.5078125` seconds | Continuous time-varying." Justified by the verified uniformity of the behaviour clock (Step 4: "Behavior timestamps align to neural samples … Rates are 15.5078125 Hz (124 sessions) or 31.015625 Hz (28 sessions)" — the latter being the mislabelled scanner rate).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. `arange(T) / 15.5078125`, cast to float32. The first sample of every trial is exactly 0.0 s and successive samples increment by 64.4836 ms. This is arithmetically equivalent to subtracting the first behaviour timestamp from the trial's timestamps, since the clock is uniform. The realised range across the full dataset is [0.0, 216.5] s.

ii.
```python
elapsed = np.arange(T, dtype=np.float32) / np.float32(TARGET_RATE)
inp = np.vstack([
    elapsed,
    np.full(T, env, dtype=np.float32),
    np.full(T, trial_num, dtype=np.float32),
    np.full(T, prev, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. Step 5 as above. Step 9 verification reports the input range `time from trial start: [0.0, 216.5]`, consistent with the longest trial (3,359 samples × 0.0644836 s = 216.6 s).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: `elapsed` has length `T = neural.shape[1]` and index 0 corresponds to the same frame as neural column 0 (the `trial_start` frame). The script asserts `neural.shape[1] == inp.shape[1] == out.shape[1]` on every trial, and asserts that the behaviour-derived lengths equal the neural length before that. Session-level length mismatches between ophys and behaviour are removed up front by truncating every stream to `common_n`.

ii.
```python
common_n = min([len(v) for v in beh.values()] +
               [int(rs.data.shape[0]) for rs, _ in series_info])
...
T = neural.shape[1]
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
...
if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
    raise AssertionError('Trial stream lengths differ')
```

iii. CONVERSION_NOTES Step 2: "Behavior timestamps align to neural samples. Neural/behavior lengths are equal in 142 sessions and differ by one in 10; truncate to common length." Step 10 Check 2 confirmed alignment independently against raw NWB reads.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` time series in `processing/behavior/BehavioralTimeSeries`, which takes values 0 (ENV 1) and 1 (ENV 2).

ii.
```python
names = ['trial_start', 'teleport', 'trial number', 'environment',
         'position', 'speed', 'lick', 'reward_zone']
beh = {k: np.asarray(bts[k].data[:]) for k in names}
...
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
```

iii. CONVERSION_NOTES Step 5 mapping table: "behavior `environment` → `input[1]` environment type | Source 0→ENV1/0, 1→ENV2/1; repeat trial value | Binary per trial." Step 2 confirms the balance: "Environment balance: code 0 = 6,226 trials (50.97%); code 1 = 5,990 trials (49.03%)", matching the paper's two-environment design.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The trial's first `environment` sample is taken, rounded to int, validated to be 0 or 1 **and** to be constant across the whole trial (hard error otherwise), then broadcast to all T timepoints as a float32 row of the input matrix. No recoding is applied — the native 0/1 already means ENV1/ENV2.

ii.
```python
env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
env = int(round(float(env_native[0])))
if env not in (0, 1) or not np.all(env_native == env_native[0]):
    raise ValueError(f'{path}: invalid/nonconstant environment in trial {qi}')
...
inp = np.vstack([elapsed, np.full(T, env, dtype=np.float32), ...])
```

iii. Step 5 decision 8: "Repeat all per-trial variables over time because the supplied decoder requires `(d,T)` arrays." Step 9 reports the realised input range `environment type: [0.0, 1.0]`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The `trial number` behaviour time series, sampled at the trial's start frame — i.e. the value the acquisition software recorded for that lap, not the Python loop counter.

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
if trial_num < 0:
    raise ValueError(f'{path}: negative trial number at start {s}')
```

iii. CONVERSION_NOTES Step 5 mapping table: "behavior `trial number` at start → `input[2]` trial number | Preserve zero-based source value; repeat across T." Step 10 Check 5: "trial IDs are unique/monotonic". (Independently verified while writing this report: across a 1-in-6 sample of the 152 sessions, `trial number` at the `trial_start` frames is exactly `0…n-1`, so it is identical to the within-session loop index.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Round to integer, check non-negative, and broadcast the scalar across all T timepoints of the trial as a float32 row. No normalisation or re-basing. The realised range over the dataset is [0, 99].

ii.
```python
trial_num = int(round(float(beh['trial number'][s])))
...
inp = np.vstack([
    elapsed,
    np.full(T, env, dtype=np.float32),
    np.full(T, trial_num, dtype=np.float32),
    np.full(T, prev, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. Step 5 decision 8 (per-trial variables repeated over time so the decoder receives `(d, T)`); Step 9 verification reports `trial number: [0.0, 99.0]`, consistent with the paper's 80–100 trials per session. The same `trial_num` is reused to decide the pre/post reward-switch zone (see 7-a), which is why the AI preferred the recorded value over a loop index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` time series in `BehavioralTimeSeries` — specifically its `timestamps` — compared against the behaviour clock `position.timestamps`. The previous-trial outcome is simply the previous element of the per-trial `outcomes` list that is built from those reward timestamps (see 11-a).

ii.
```python
pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
...
# Reward timestamps are exactly aligned, but timestamp comparison is robust.
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
```

iii. CONVERSION_NOTES Step 5 mapping table: "prior trial Reward timestamp → `input[3]` previous outcome | First trial 0; otherwise 1 iff reward delivered in immediately preceding valid trial; repeat | Binary per trial." Step 2 notes `Reward` "is a sparse timestamped reward-delivery series", hence the timestamp-window rather than index-based test.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Trials are processed in order and each trial's binary outcome is appended to `outcomes`. For trial *k* > 0 the input is `outcomes[k-1]` (accessed as `outcomes[-2]`, since the current trial's outcome has already been appended); for the first trial of a session it is 0. The scalar is broadcast across all T timepoints as float32. "Previous" is defined within the session only — it never crosses a session boundary.

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
...
prev = outcomes[-2] if len(outcomes) > 1 else 0
...
np.full(T, prev, dtype=np.float32),
```

iii. Step 5 mapping table as quoted above. The instructions specify "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"; the first trial has no predecessor and is assigned the "omitted" code 0. Step 9 reports `previous trial outcome: [0.0, 1.0]`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behaviour time series, and the active reward zone for the trial. The active zone is **not** inferred from the `reward_zone` time series. It is read from the session's NWB `identifier` string (e.g. `/data/InVivoDA/GCAMP11/25_02_2023/Env1_LocationA_to_C`), parsed by `parse_zone_sequence()` into a source and destination zone letter, and then selected per trial by the paper's protocol rule: zone = source for `trial number` < 30, destination for `trial number` ≥ 30. On non-switch ("stay"/fixed) sessions the identifier holds a single letter and source = destination. Zone extents are the paper's nominal ones: A = [80, 130], B = [200, 250], C = [320, 370] cm.

ii.
```python
ZONE_START = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_END = ZONE_START + 50.0

def parse_zone_sequence(identifier: str) -> tuple[int, int]:
    """Return source/destination A/B/C indices parsed from the NWB identifier."""
    name = identifier.rstrip('/').split('/')[-1]
    letters = re.findall(r'(?:Location)?([ABC])', name)
    if '_to_' in name and len(letters) >= 2:
        return ord(letters[-2]) - 65, ord(letters[-1]) - 65
    if letters:
        z = ord(letters[-1]) - 65
        return z, z
    raise ValueError(f'Cannot parse reward location from identifier: {identifier}')

def zone_for_trial(src: int, dst: int, trial_number: int) -> int:
    return src if trial_number < 30 else dst
```
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
is_switch = src_zone != dst_zone
...
zone = zone_for_trial(src_zone, dst_zone, trial_num)
_, dist_cls = distance_classes(pos, zone)
```

iii. CONVERSION_NOTES Step 5 decision 5: "Use nominal 50 cm zones [80,130], [200,250], [320,370] cm (A/B/C), with membership distance zero. **Zone identity comes from identifier because omission trials may have no positive `reward_zone` samples.**" The 30-trial rule comes from Step 3: "Reward switch | after first 30 trials" (Methods: "Each switch occurred after 30 trials"; "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm"). Step 10 Check 5 reports: "switch zones change exactly at source trial number 30". (Independently verified while writing this report: across 16 sessions / 1,102 trials that had any `reward_zone > 0` sample, the identifier+trial-30 assignment agreed with the empirically nearest zone on 100% of trials.)

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the signed distance to the nearest point of the 50 cm zone interval is computed as a linear (not circular) quantity: `position − zone_start` when before the zone, `position − zone_end` when past it, and exactly 0.0 while inside. The raw distance is then discretised (see 7-c); the continuous value itself is discarded.

ii.
```python
def distance_classes(position: np.ndarray, zone: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance to nearest point in the 50-cm reward-zone interval."""
    lo, hi = float(ZONE_START[zone]), float(ZONE_END[zone])
    d = np.where(position < lo, position - lo,
                 np.where(position > hi, position - hi, 0.0)).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: "position + nominal reward-zone interval → `output[0]` distance to reward zone | Signed distance to nearest point in interval: `pos-start` before zone, 0 inside, `pos-end` after; discretize task bins | **Linear signed distance, not circular wrap, because requested bins distinguish before/after on 450 cm corridor.**" Note the input to `distance_classes` is the raw (unclipped) position, whereas `position_classes` clips — a deliberate asymmetry so that pre-track samples give large negative distances.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks that follow the instruction text literally: 0 for d < −50; 1 for −50 ≤ d < −10; 2 for −10 ≤ d < 0; 3 for d exactly 0 (inside the zone); 4 for 0 < d ≤ 10; 5 for 10 < d ≤ 50; 6 for d > 50. Realised fractions on the full dataset: [0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242].

ii.
```python
# 0:<-50; 1:[-50,-10); 2:[-10,0); 3:0; 4:(0,10]; 5:(10,50]; 6:>50
c = np.empty(d.shape, dtype=np.int64)
c[d < -50] = 0
c[(d >= -50) & (d < -10)] = 1
c[(d >= -10) & (d < 0)] = 2
c[d == 0] = 3
c[(d > 0) & (d <= 10)] = 4
c[(d > 10) & (d <= 50)] = 5
c[d > 50] = 6
```
```python
'output_values': [
    ['< -50 cm', '-50 to < -10 cm', '-10 to < 0 cm', '0 cm (inside reward zone)',
     '> 0 to +10 cm', '> +10 to +50 cm', '> +50 cm'], ...]
```

iii. CONVERSION_NOTES Step 5 decision 6: "Implement specified strict inequalities exactly: distance values exactly -50/-10 use classes 1/2 respectively; exactly 0 class 3; +10/+50 use classes 4/5." Step 10 Check 5: "Exact discretizer boundary tests passed for distance, position, and speed." The class-3 "0 cm" category is well-defined here because the distance is clamped to exactly 0.0 inside the zone, so "inside the reward zone" and "distance 0" are the same event.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance class series is computed from `pos`, which is the `position` behaviour array sliced with exactly the same `[s, e)` indices used for the neural matrix, so index *t* of the output corresponds to the same imaging frame as column *t* of `neural`. No lag, shift, or interpolation is introduced, and the per-trial shape assertion enforces equality of T.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
...
zone = zone_for_trial(src_zone, dst_zone, trial_num)
_, dist_cls = distance_classes(pos, zone)
...
out = np.vstack([dist_cls, pos_cls, speed_cls, lick,
                 np.full(T, zone, dtype=np.int64),
                 np.full(T, rewarded, dtype=np.int64)]).astype(np.int64, copy=False)
if not (neural.shape[1] == inp.shape[1] == out.shape[1]):
    raise AssertionError('Trial stream lengths differ')
```

iii. Step 10 Check 2 verified this by reloading the raw NWB independently and `assert_allclose`-ing all six outputs for three trials in single- and two-plane sessions. Step 7 processing plots (`processing_m11_ses-03.png`, `processing_m17_ses-01.png`) overlay the neural raster and all six output traces on the same trial-start-aligned time axis: "Neural events and all output streams span the same trial-start-aligned time axis."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series only (cm along the 450 cm virtual linear track).

ii.
```python
beh = {k: np.asarray(bts[k].data[:]) for k in names}   # includes 'position'
...
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
```

iii. CONVERSION_NOTES Step 2: "Raw inter-trial position can be -500 cm; valid track is 0--450 cm … Position raw global range -500 to 452.47 cm." Step 5 mapping: "behavior `position` → `output[1]` absolute position".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial slice is taken, cast to float32, then **clipped to [0, 450] cm** inside `position_classes` before binning. Clipping absorbs the small amount of numerical/edge noise just outside the track (raw min −500 cm occurs only in the excluded teleport period; within trials the observed excursions are tiny). The clipped value is then discretised; the continuous position is not stored.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    p = np.clip(position, 0.0, 450.0)
    c = np.zeros(p.shape, dtype=np.int64)
    c[p >= 90] = 1
    c[p >= 180] = 2
    c[p >= 270] = 3
    c[p > 360] = 4
    return c
```

iii. CONVERSION_NOTES Step 5 mapping table: "behavior `position` → `output[1]` absolute position | **Clip numerical edge noise to [0,450]**, bins `<90`, `90–180`, `180–270`, `270–360`, `>360`". Justified by Step 2's observation that out-of-range position values belong to the teleport/ITI period, which is already excluded by the half-open trial interval.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes over the 450 cm track using 90 cm bins, assigned by cascading boolean masks: 0 for p < 90; 1 for 90 ≤ p < 180; 2 for 180 ≤ p < 270; 3 for 270 ≤ p ≤ 360; 4 for p > 360. Because the cascade ends with a strict `p > 360`, a sample at exactly 360 cm falls in class 3, matching the literal wording "3: 270 to 360 cm / 4: > 360 cm". Realised fractions: [0.211, 0.178, 0.231, 0.227, 0.154].

ii.
```python
c[p >= 90] = 1
c[p >= 180] = 2
c[p >= 270] = 3
c[p > 360] = 4
```
```python
['< 90 cm', '90 to < 180 cm', '180 to < 270 cm', '270 to 360 cm', '> 360 cm'],
```

iii. CONVERSION_NOTES Step 5 decision 6: "Position exactly 90/180/270/360 enters the higher bin **except task wording `>360`, for which exactly 360 remains bin 3**." Step 10 Check 5 confirms "Exact discretizer boundary tests passed for … position".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identically to 7-d: the same `[s, e)` sample indices index both `position` and the ophys rows, so no alignment step is required, and the per-trial length assertion guarantees the streams stay in register.

ii.
```python
pos = aggregate_behavior(beh['position'][s:e], factor, 'mean').astype(np.float32)
pos_cls = position_classes(pos)
...
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
```

iii. Same justification as 7-d — Step 10's independent raw-NWB `assert_allclose` spot checks and the Step 7 processing plots.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series, which the AI established is a per-frame **cumulative lick count**, not a binary flag.

ii.
```python
raw_lick = beh['lick'][s:e]
```

iii. CONVERSION_NOTES Step 2: "Lick and reward-zone streams are cumulative counts per frame, so event presence is `>0`."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised with `raw_lick > 0`, then (nominally) max-aggregated over the resampling factor and re-thresholded — with `factor = 1` this reduces to a plain `> 0` binarisation cast to int64. Separately, and without changing the lick output, the paper's artifact criterion (>30% of the trial's frames with cumulative count > 2) is evaluated and stored as a per-trial diagnostic flag; 81 trials are flagged across the dataset. Realised class fractions: [0.770, 0.230].

ii.
```python
raw_lick = beh['lick'][s:e]
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)
lick_artifacts += int(lick_artifact)
...
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping: "behavior `lick` → `output[3]` lick | `lick > 0` on each synchronized frame | paper converts cumulative counts to binary." Step 5 decision 7 explains keeping the artifact trials: "Retain trials and binary lick because target format has no missing-output mask and deleting entire trials would discard valid neural and other outputs." Step 4 corroborates the 81/12,376 figure from the Methods.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same-index slicing `[s, e)` as the neural data, with no shift; the per-trial assertion checks `len(lick) == T`.

ii.
```python
raw_lick = beh['lick'][s:e]
lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0).astype(np.int64)
...
if not (T == len(pos) == len(speed) == len(lick) == n_complete // factor):
    raise AssertionError('Resampling length mismatch')
```

iii. Step 10 Check 2's independent raw-data comparison included the lick output; Step 7's plots show lick co-varying with position within the reward zone.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The NWB `identifier` string of the session (which encodes the environment and the reward-zone schedule, e.g. `Env1_LocationA_to_C` or `Env2_LocationB`) combined with the per-trial `trial number`. The `reward_zone` behaviour series is read into `beh` but is **not** used to assign the label (it is only referenced during exploration). See 7-a.

ii. See 7-a — `parse_zone_sequence(nwb.identifier)` and `zone_for_trial(src, dst, trial_num)`.
```python
src_zone, dst_zone = parse_zone_sequence(nwb.identifier)
...
zone = zone_for_trial(src_zone, dst_zone, trial_num)
out = np.vstack([dist_cls, pos_cls, speed_cls, lick,
                 np.full(T, zone, dtype=np.int64),
                 np.full(T, rewarded, dtype=np.int64)])
```

iii. See 7-a. Step 5 decision 5: "Zone identity comes from identifier because omission trials may have no positive `reward_zone` samples" — i.e. the identifier gives a label on every trial, including omission trials and trials where the mouse never entered/licked in the zone, whereas the `reward_zone` stream is silent on some of them (198 of 1,300 trials in my 16-session spot check had no `reward_zone > 0` sample at all).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Map the parsed letter to an integer (A = 0, B = 1, C = 2) by `ord(letter) - 65`; pick source vs destination by `trial_number < 30`; broadcast the scalar across all T timepoints of the trial as int64. `output_values[4] = ['A','B','C']`. The same zone index drives the distance-to-reward-zone output (7-b), so the two outputs are guaranteed mutually consistent. Realised fractions: [0.329, 0.337, 0.335].

ii.
```python
def zone_for_trial(src: int, dst: int, trial_number: int) -> int:
    return src if trial_number < 30 else dst
...
np.full(T, zone, dtype=np.int64),
...
'reward_zone_intervals_cm': {'A':[80.0,130.0], 'B':[200.0,250.0], 'C':[320.0,370.0]},
```

iii. Step 5 mapping table: "session identifier + trial index → `output[4]` reward-zone location | Parse source/destination letters; trials 0–29 source, trial ≥30 destination on `_to_` sessions; fixed sessions use sole letter. A=0, B=1, C=2; repeat". Step 10 Check 5 verified "switch zones change exactly at source trial number 30", and the near-uniform 1/3 class fractions are consistent with the paper's counterbalancing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' `timestamps` (a sparse event stream with its own timestamps) tested against the behaviour clock `position.timestamps` at the trial's start and end frames.

ii.
```python
pos_ts = np.asarray(bts['position'].timestamps[:common_n], dtype=np.float64)
reward_ts = np.asarray(bts['Reward'].timestamps[:], dtype=np.float64)
...
# Reward timestamps are exactly aligned, but timestamp comparison is robust.
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
```

iii. CONVERSION_NOTES Step 2: "`Reward` is a sparse timestamped reward-delivery series." Step 5 mapping: "sparse `Reward.timestamps` → `output[5]` reward outcome | 1 iff a reward timestamp lies in `[start, teleport)`; repeat | Per-trial binary."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labelled rewarded (1) if at least one reward timestamp falls in the half-open time window `[t(start), t(end))` — the same window as the trial's samples — and omitted (0) otherwise; the scalar is broadcast across all T timepoints as int64. Comparing in the *time* domain rather than the *index* domain avoids having to snap reward times onto frame indices. Across the dataset 10,342 / 12,216 trials (84.66%) are rewarded, i.e. ~15% omissions, matching the paper's "reward was randomly omitted on ~15% of trials"; time-weighted the classes are [0.157, 0.843].

ii.
```python
rewarded = int(np.any((reward_ts >= pos_ts[s]) & (reward_ts < pos_ts[e])))
outcomes.append(rewarded)
...
np.full(T, rewarded, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2: "Rewarded trials: 10,342 / 12,216 = 84.66% using Reward timestamps within trial bounds", cross-checked in Step 9's consistency table. Step 12 re-verified individual labels against the raw NWB: "Loaded raw NWB Reward timestamps and checked m11 session 03 trials 1 (omitted), 0 and 79 (rewarded); converted labels exactly matched."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms, and a deliberate fail-fast policy elsewhere:
 - **Ophys/behaviour length mismatch** (10 of 152 sessions differ by one sample): every behaviour array and every plane's ophys array is truncated to `common_n = min(all lengths)` *before* trial markers are located, so no trial can index past the end of any stream.
 - **Multi-plane column mapping**: each `RoiResponseSeries` has its own `DynamicTableRegion`, so the `iscell` labels are gathered through `labels[region, 0]` rather than by positional assumption (this was an actual bug the AI found and fixed).
 - **Mislabelled sampling rate** on two-plane sessions: the stored `rate` (31.015625 Hz) is ignored in favour of the measured behaviour clock, with a hard check that it is 15.5078125 Hz.
 - **Out-of-range behaviour values**: position is clipped to [0, 450] and speed clamped at ≥ 0 before binning; cumulative lick counts are binarised.
 - Everything else raises: unmatched trial markers, a plane with zero accepted cells, differing plane rates, a negative trial number, a non-constant or non-{0,1} environment, a zero-length trial, mismatched stream lengths, or any non-finite value in the converted neural/input arrays. No missing-value imputation is performed, and no trial or session is silently dropped.

ii.
```python
common_n = min([len(v) for v in beh.values()] +
               [int(rs.data.shape[0]) for rs, _ in series_info])
starts = np.flatnonzero(beh['trial_start'][:common_n] > 0)
ends = np.flatnonzero(beh['teleport'][:common_n] > 0)
pairs = pair_bounds(starts, ends)
if len(pairs) != len(starts) or len(starts) != len(ends):
    raise ValueError(f'{path}: unmatched trial markers {len(starts)}/{len(ends)}/{len(pairs)}')
```
```python
if not np.allclose(rates, rates[0]):
    raise ValueError(f'{path}: plane rates differ: {rates}')
...
if not np.isclose(effective_rate, TARGET_RATE, rtol=2e-3):
    raise ValueError(f'{path}: unsupported behavior-clock rate {effective_rate}')
...
if not np.isfinite(neural).all() or not np.isfinite(inp).all():
    raise ValueError(f'{path}: non-finite converted data')
```

iii. CONVERSION_NOTES Step 2: "Neural/behavior lengths are equal in 142 sessions and differ by one in 10; truncate to common length." Step 10 "Issues Found and Resolved" documents the two real data quirks (two-plane rate metadata; plane-0-only cell counting) and their fixes, plus the lick-artifact decision. Step 10 Check 5: "Every start has one later teleport; trial IDs are unique/monotonic; … every stream is finite and shape-matched." The full conversion ran to completion over all 152 files with none of the guards tripping.

## 13-a. What are the most time-consuming steps of the code?

i. The full conversion takes **85.07 s** for 152 sessions (0.2–0.9 s per session), so nothing is a serious bottleneck. Within that, the dominant costs are (1) HDF5 reads of the `Deconvolved` arrays — issued once per trial per plane, ~12,216 × 1–2 slice reads totalling roughly 13 GB decompressed; (2) writing the 9.84 GB output pickle; and (3) opening each NWB file and building the pynwb object graph. The behaviour arrays, discretisation and per-trial bookkeeping are negligible. The AI instruments this with per-session timing (`info['seconds']`) printed to the log and a total elapsed time.

ii.
```python
def process_session(path: Path):
    t0 = time.time()
    ...
    info = {..., 'seconds': time.time() - t0}
```
```python
print(f"[{i}/{len(files)}] {p.name}: neurons={info['n_neurons']} trials={info['n_trials']} "
      f"rewarded={info['n_rewarded']} factor={info['downsample_factor']} "
      f"lick_bad={info['lick_artifact_trials']} time={info['seconds']:.2f}s", flush=True)
...
print(f'Wrote {out} ({out.stat().st_size/1e9:.3f} GB), elapsed {time.time()-t0:.2f}s', flush=True)
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Loading full uncurated fluorescence arrays would be wasteful; only Deconvolved trial slices are loaded. Two-plane sessions require both plane series but should not materialize full-session matrices. Repeated NWB opens would dominate I/O." Step 9: "Conversion time: 85.07 s, well below the 15-minute threshold."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain, all cheap:
 - `pair_bounds()` iterates over trial starts with an inner `while` to advance the teleport pointer. This is a pure interval-matching problem and could be done in one `np.searchsorted(ends, starts)` call.
 - The main `for qi, (s, e) in enumerate(pairs)` loop in `process_session` recomputes discretisation per trial. `distance_classes`, `position_classes` and `speed_classes` are already fully vectorised inside, but they could have been applied once to the whole session array and then split — except that the reward zone changes at trial 30, so distance would need a per-trial zone vector first.
 - `load_neural_trial` loops over planes (1 or 2 iterations) and issues one HDF5 slice read per trial; reading each plane's full array once per session and slicing in memory would trade ~1–2 GB of peak RAM for fewer, larger reads.
 The AI did not enumerate these; it documented only the vectorisation it *did* do.

ii.
```python
pairs, j = [], 0
for s in starts:
    while j < len(ends) and ends[j] < s:
        j += 1
    ...
```
```python
for qi, (s, e) in enumerate(pairs):
    ...
    _, dist_cls = distance_classes(pos, zone)
    pos_cls = position_classes(pos)
    speed_cls = speed_classes(speed)
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Each NWB is opened once and processed trial-by-trial. Neural reads are sliced to trial bounds before cell curation/resampling. **Vectorized discretization and pairwise aggregation are used.** Float32 neural/input and int64 categorical output arrays avoid unnecessary float64 storage." Step 7 concluded the estimated full run was "<2 minutes for 152 sessions plus pickle I/O", so no further optimisation was pursued.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly **once** and read in a single pass — there is no separate survey/statistics pass over the dataset. The repeated work that does exist is minor: (a) the `iscell`/region lookup and the behaviour arrays are read once per session, but `rs.data` is hit once per trial per plane rather than once per session, so the HDF5 chunk/dataset access path is re-traversed ~12,216 times; (b) `aggregate_behavior` is called four times per trial with `factor = 1`, in which case it slices and returns the array unchanged; (c) per-trial constants (`env`, `trial_num`, `prev`, `zone`, `rewarded`) are re-materialised as full-length T vectors via `np.full` for both the input and output matrices, which is required by the target format rather than redundant computation.

ii.
```python
for qi, (s, e) in enumerate(pairs):
    ...
    neural = load_neural_trial(series_info, s, e, factor)     # HDF5 read per trial per plane
    pos = aggregate_behavior(beh['position'][s:e], factor, 'mean')
    speed = aggregate_behavior(beh['speed'][s:e], factor, 'mean')
    lick = (aggregate_behavior((raw_lick > 0).astype(np.int8), factor, 'max') > 0)
    env_native = aggregate_behavior(beh['environment'][s:e], factor, 'first')
```
```python
def aggregate_behavior(x: np.ndarray, factor: int, mode: str) -> np.ndarray:
    n = (len(x) // factor) * factor
    x = np.asarray(x[:n])
    if factor == 1:
        return x
```

iii. CONVERSION_NOTES Step 6: "Repeated NWB opens would dominate I/O … Each NWB is opened once and processed trial-by-trial." This single-pass design is what keeps the full conversion at 85 s.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, none expensive:
 - **Dead resampling machinery.** `factor` is hard-coded to 1 after the two-plane-rate bug was fixed, so `aggregate_behavior`'s `mean`/`max`/`first` branches and `load_neural_trial`'s `reshape(...).sum(axis=1)` branch are unreachable. `n_complete = ((e - s) // factor) * factor` is likewise always `e - s`.
 - **Discarded continuous distance.** `distance_classes` builds and returns the float32 signed-distance array `d`, which the caller throws away (`_, dist_cls = ...`); only the class labels are stored.
 - **Unused lick-artifact flag.** `lick_artifact` is computed for all 12,216 trials and stored per trial, but never affects any converted value — it is diagnostic only (a deliberate choice, see 1-e).
 - **Unused metadata.** `effective_rate`, `native_rate`, `downsample_factor`, `is_switch` and the full per-trial `trial_info` list are written into the pickle; `metadata['session_info']` contributes a non-trivial amount of the file but is not read by the decoder. `off_start = 0.0` is also set even though the alignment event *is* the trial start.
 - Conversely, the code performs none of the paper's expensive-but-required neural preprocessing (see 2-b), which is why it is so fast.

ii.
```python
factor = 1
...
n_complete = ((e - s) // factor) * factor
...
_, dist_cls = distance_classes(pos, zone)     # continuous distance `d` discarded
...
lick_artifact = bool(np.mean(raw_lick > 2) > 0.30)   # recorded, never applied
```
```python
'downsample_factor': factor, ...
'lick_artifact_trials': lick_artifacts, 'is_switch': is_switch,
'trial_info': trial_info, 'seconds': time.time() - t0,
```

iii. The AI does not identify any of this as unnecessary. CONVERSION_NOTES Step 6 frames all of it as deliberate ("Vectorized discretization and pairwise aggregation are used"), and Step 13 declares the directory clean. The retained diagnostics are defended in Step 5 decision 7 and Step 10 ("the count is recorded in metadata/logs") as provenance for the paper's 81-trial lick-artifact statistic.
