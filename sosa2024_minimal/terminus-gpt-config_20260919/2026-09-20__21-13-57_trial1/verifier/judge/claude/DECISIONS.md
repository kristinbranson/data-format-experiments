# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every NWB file matching `sub-*/*_behavior+ophys.nwb` under `/app/data`, sorts them, and processes each one as a session. All 152 files across the 11 `sub-*` directories are loaded. It reads the HDF5 files directly with `h5py` rather than with `pynwb`, and reads lazily: behavioral streams are read whole, but the (large) ophys matrices are sliced per trial straight off the HDF5 dataset. Within a session all trials found between a `trial_start` pulse and the following `teleport` sample are kept.

ii.
```python
DATA_ROOT = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))
if not files:
    raise FileNotFoundError('No NWB files found')
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
...
for i,p in enumerate(files):
    nt, it, ot, nc, scene, rate, block = convert_session(p)
```
```python
def convert_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        ...
        pos = b['position/data'][:]
        speed = b['speed/data'][:]
        lick = b['lick/data'][:]
        env = b['environment/data'][:]
        trial_num = b['trial number/data'][:]
        trial_start = b['trial_start/data'][:]
        teleport = b['teleport/data'][:]
        timestamps = b['position/timestamps'][:]
```

iii. From the trajectory, the AI first enumerated the directory tree and the DANDI metadata (`find /app/data ...`, `dandiset.yaml`), then walked one NWB with `h5py.visititems` to learn the group layout, and finally scanned *every* NWB in one pass to tabulate frame rate, number of frames, ROI counts, trials, environments and reward counts ("files 152 subjects [...]"). It chose `h5py` explicitly because "NWB files are large, metadata and HDF5 structure can be inspected lazily without loading imaging arrays"; the per-trial HDF5 slicing was then kept in the converter to avoid ever holding a full session's deconvolved matrix in memory.

## 1-b. How are the data split into subjects?

i. One subject per `sub-<id>` directory; the subject id is taken from the parent directory name of each NWB file. `subjects` is the sorted unique set (11 mice: m11–m15, m17–m19, m3, m4, m7) and `subject_idx` stores the index of the owning subject for every session, appended in the same order as the session lists.

ii.
```python
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
...
subj = os.path.basename(os.path.dirname(p)).replace('sub-','')
data['subject_idx'].append(subj_to_idx[subj])
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int16)
```

iii. The AI's first scan printed `subjects sorted(set(...))` from the directory names and confirmed 11 mice, consistent with the paper's cohort. No further justification is given in the trajectory — the directory naming (`sub-m11`, ...) is treated as self-evident.

## 1-c. How are the data split into sessions?

i. One session per NWB file. Sessions are not merged across days or subjects, and nothing is done about tracking ROIs across days (the paper's cross-day ROI alignment is ignored). Session provenance (file name, scene string, source frame rate, frames per bin, trial and neuron counts) is recorded in `metadata['session_info']`. A session is dropped only if fewer than 2 trials survive (never triggered: all 152 sessions are kept).

ii.
```python
for i,p in enumerate(files):
    nt, it, ot, nc, scene, rate, block = convert_session(p)
    if len(nt) < 2:
        print('SKIP (<2 trials)', p, flush=True)
        continue
    ...
    data['metadata']['session_info'].append({
        'file': os.path.relpath(p, DATA_ROOT), 'scene':scene,
        'source_frame_rate_hz':rate, 'frames_per_bin':block,
        'n_trials':len(nt), 'n_neurons':nc})
```

iii. The AI's whole-dataset scan showed one `ses-NN` NWB per experiment day with its own ROI set and its own `iscell` table, so each file is treated as an independent recording. The `<2 trials` guard is there to satisfy the instruction that "there needs to be at least two trials within each session in order to evaluate the decoder performance."

## 1-d. How are the data split into trials?

i. A trial starts at each `trial_start` pulse (the AI established that this pulse coincides with the position crossing 0 cm) and ends at the first sample at which `teleport` is non-zero; the teleport sample itself and everything after it, up to the next `trial_start`, is excluded. The AI explicitly rejected slicing by the `trial number` stream because the trial-number transition happens earlier, during the inter-trial teleport period.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
...
for s in starts:
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    if e <= s:
        continue
    # Ignore a clipped final trial if it never reaches teleport.
    if not len(tq):
        continue
```

iii. From the trajectory (step 14): "Trials should be aligned to `trial_start` pulses, which occur as position crosses 0 cm. Trial-number transitions occur earlier during the inter-trial teleport period, so slicing by trial-number alone would misalign trials and include teleport. The appropriate trial interval is from each `trial_start` pulse until track completion/teleport onset, giving variable-duration trials." The AI verified this by printing `trial_start`, `trial number` and `position` around each pulse and by counting: 12,216 trials over 152 sessions (mostly 80 per session, consistent with the paper's "80.5 ± 7.4 trials").

## 1-e. How are trials filtered based on quality controls?

i. Two filters only, both structural rather than behavioural: (1) a trailing `trial_start` that never reaches a `teleport` (a clipped final lap) is dropped; (2) a trial that yields fewer than 2 time bins (i.e. < 8 source frames, ≈ 0.5 s) is dropped. No filtering on running, licking, position coverage, or trial duration. In practice neither filter removes anything measurable: the output contains all 12,216 trials.

ii.
```python
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    if e <= s:
        continue
    # Ignore a clipped final trial if it never reaches teleport.
    if not len(tq):
        continue
    offsets = np.arange(s, e, block, dtype=int)
    if len(offsets) < 2:
        continue
```

iii. The AI's justification (step 15): "A small number of unusually long trials exist, but these likely reflect genuine pauses and should not be arbitrarily discarded unless source processing does so." It found no trial-exclusion rule in the repository or Methods (its `grep` for "trial exclusion|excluded" returned only analysis-level criteria such as "at least three omission trials within the set"), so it kept everything that the source data defines as a lap.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is taken from the NWB `processing/ophys/Deconvolved/planeN/data` arrays — Suite2p's own deconvolved traces (`spks`) that ship with the release. The raw `Fluorescence` and `Neuropil` arrays, which are also present in every NWB file, are not read at all. In multi-plane sessions the accepted cells of `plane0` and `plane1` are concatenated.

ii.
```python
dg = f['processing/ophys/Deconvolved']
plane_names = sorted(dg.keys(), key=lambda x: int(x.replace('plane','')))
...
for name in plane_names:
    pi = int(name.replace('plane',''))
    dset = dg[name]['data']
    mask = iscell[plane_idx == pi]
    if dset.shape[1] != len(mask):
        raise ValueError(f'ROI count mismatch for {name}: {dset.shape[1]} vs {len(mask)}')
    plane_series.append((dset, np.flatnonzero(mask)))
n_cells = sum(len(idx) for _,idx in plane_series)
...
raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
raw = np.concatenate(raw_parts, axis=1)
```

iii. The AI's stated reason, in the file docstring, is "Use the supplied OASIS-deconvolved activity" and in step 16 "The source NWBs already contain paper-processed OASIS-deconvolved activity". It had read the Methods passage describing the paper's dF/F + OASIS pipeline (it printed line 29 of `methods.txt` at step 15) and inferred that the stored `Deconvolved` array *is* the output of that pipeline; it never opened `src/reward_relative/preprocessing.py::dff` to check, and never compared `Deconvolved` against `Fluorescence`/`Neuropil`. For pooling planes it cited the Methods directly: "planes were pooled for all analyses except those in Extended Data Fig. 7".

## 2-b. How is the `neural` data processed?

i. Beyond selecting `iscell` ROIs and pooling planes, the only processing is a block mean: every 4 consecutive source frames are averaged into one bin, and the result is transposed to (n_neurons, n_bins) and cast to `float16`. There is no dF/F computation, no neuropil subtraction (`neu_coef = 0.7`), no per-trial maximin baseline over a 20 s window, no 2-sample Gaussian smoothing, and no OASIS deconvolution of a dF/F trace with `tau = 0.7` — i.e. none of the pipeline in `preprocessing.dff`, which is what the paper's analyses run on. The values stored are therefore Suite2p `spks` in raw-fluorescence units (max value in the converted file is 6620).

ii.
```python
def binned_mean(x, starts, block):
    # x is time x features (or a vector). Drop only the final incomplete block.
    return np.stack([np.asarray(x[s:min(s+block, len(x))]).mean(axis=0)
                     for s in starts], axis=0)
...
raw = np.concatenate(raw_parts, axis=1)
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
```

iii. The AI assumed the stored deconvolved traces were already the paper's processed signal, so it deliberately did no further processing. For the block mean it reasoned about tractability (step 15): "Using all 152 sessions, all accepted neurons, and native 15.5 Hz bins would produce about 2.11 billion neural values (~7.86 GiB as float32), too large for reliable pickle loading and decoder training", and chose `float16` because it is "a safe representation for deconvolved activity".

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter: ROIs are restricted to `PlaneSegmentation/iscell[:,0] == 1`, Suite2p's label after the authors' manual curation, applied per plane. The Methods' second, explicitly programmatic filter — "Additional putative interneurons were detected for exclusion from further analysis by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells" — is not applied. The result is 138,678 neurons across sessions (range 155–2341 per session).

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:,0] == 1
plane_idx = seg['planeIdx'][:].astype(int) if 'planeIdx' in seg else np.zeros(len(iscell), int)
plane_series = []
for name in plane_names:
    pi = int(name.replace('plane',''))
    dset = dg[name]['data']
    mask = iscell[plane_idx == pi]
```

iii. At step 4 the AI observed that "`iscell[:,0]` clearly distinguishes accepted neurons", and it tied this to the Methods passage it printed at step 15 about manual curation eliminating ROIs "exhibiting high and continuous fluorescence fluctuation typical of putative interneurons", concluding that curation alone yields "putative pyramidal neurons per session" in the quoted 155–2172 range. It cross-checked its own per-session counts against that range. It gives no reason for skipping the speed-correlation exclusion; the step is simply never mentioned — plausibly because it does not compute dF/F and so has no trace to correlate with speed. The multi-plane handling was added reactively after the converter crashed on `sub-m17_ses-01` with an ROI-count mismatch (step 19–20).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial slicing itself: the first neural bin of every trial begins at the `trial_start` frame. `metadata['temporal_alignment_event']` is set to "trial_start pulse at the 0 cm corridor crossing", `off_start = 0.0` and `off_end = None` (trials have variable length). No pre-trial baseline window is included.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
```
```python
'temporal_alignment_event': 'trial_start pulse at the 0 cm corridor crossing',
'off_start': 0.0, 'off_end': None,
```

iii. Step 14: "Trials should be aligned to `trial_start` pulses, which occur as position crosses 0 cm." Because the behavioural and ophys arrays in these NWBs are frame-aligned (identical lengths; the AI verified 19,818 frames for both in the first session it inspected), slicing both with the same frame indices aligns them by construction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the data are rebinned. Source frames arrive at 15.5078125 Hz (64.484 ms); the converter averages blocks of 4 frames into a single 257.934 ms bin, a 4× reduction in temporal resolution, and declares `time_bin_size = 257.93 ms`. All streams (neural, position, speed) use the block mean; lick uses a block OR. The final, incomplete block of a trial is kept and averaged over however many frames remain (the comment above `binned_mean` says the opposite). The resulting trials average 54.6 bins (median 49.8, range 24–840).

The AI initially derived the block size from the ophys `rate` attribute, then caught a real problem before submitting: 28 NWBs report `rate = 31.015625` Hz on the RoiResponseSeries, but *all* 152 files have shared frame timestamps at 15.5078 Hz. It audited this across the whole dataset (`124 (0.06448363, 15.507812)`, `28 (0.06448363, 31.015625)`), concluded the stored rate is the scanner rate for two-plane sessions, switched to deriving the interval from `position/timestamps`, and regenerated the whole dataset so that every session really is on a common bin.

ii.
```python
TARGET_DT = 4 / 15.5078125
...
# Arrays are aligned by frame index. In 28 NWBs the ophys rate attribute
# is the resonant scanner line/plane rate (31 Hz), while shared frame
# timestamps are 15.5 Hz. Use those timestamps as the authoritative rate.
bt = b['position/timestamps']
dt = float(np.median(np.diff(bt[:min(len(bt), 2000)])))
rate = 1.0 / dt
block = int(round(TARGET_DT / dt))
if not np.isclose(block * dt, TARGET_DT, rtol=0, atol=2e-6):
    raise ValueError(f'Unexpected aligned-frame interval {dt} in {path}')
```
```python
'time_bin_size': float(TARGET_DT*1000),
```

iii. Two justifications, both in the trajectory. For rebinning at all (step 15): native-resolution data would be "about 2.11 billion neural values (~7.86 GiB as float32), too large for reliable pickle loading and decoder training", so "a coarser common time bin is therefore necessary for downstream analysis"; 258 ms was picked as an integer multiple of the frame interval. For the rate fix (step 27/29): "the NWB deconvolved `starting_time.rate` can report ~31 Hz while the frame-aligned behavioral timestamps ... advance at ~15.5 Hz. Since neural and behavioral arrays have identical lengths and are aligned by index, behavioral timestamps are the authoritative effective sample interval ... The existing pickle consequently uses 516 ms bins for those 28 sessions while declaring 258 ms bins. This violates the common-bin requirement."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from a stored timestamp vector per sample. It is synthesized from the bin index and the nominal bin duration `TARGET_DT`, where `TARGET_DT = 4 / 15.5078125 s` and the 4-frame block size is validated against the median diff of `position/timestamps` for that session. So the raw variable behind it is `BehavioralTimeSeries/position/timestamps`, used once per session to fix the sampling interval rather than sample-by-sample.

ii.
```python
bt = b['position/timestamps']
dt = float(np.median(np.diff(bt[:min(len(bt), 2000)])))
rate = 1.0 / dt
block = int(round(TARGET_DT / dt))
...
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
    ...
])
```

iii. The AI established that all behavioural series in a file share one timestamp vector and that the interval is constant across the entire dataset (its audit printed a single interval, `0.06448363 s`, for all 152 sessions), so a nominal time axis and the stored timestamps are interchangeable. The `np.isclose(block*dt, TARGET_DT, atol=2e-6)` guard is what enforces that assumption per session.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond construction: the first bin of every trial is 0.0 s and each subsequent bin adds 257.934 ms. The value is float32. (Because the time axis is nominal, each bin is labelled by its left edge rather than its centre, and no clock drift within a session is reflected.)

ii.
```python
T = len(offsets)
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
    ...
])
```

iii. Implicit: the instructions ask for "time from start of trial in seconds", and trials are cut at the trial-start pulse, so elapsed time is simply bin index × bin duration.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction — the time vector is generated from the same `offsets` array that defines the neural bins, so `inp.shape[1] == nbin.shape[1]` always. More generally, the AI relies on the neural and behavioural arrays in each NWB being frame-aligned (same index = same moment), which it verified on sampled files. It does not compare the ophys array length with the behavioural array length; in 10 sessions the ophys arrays are actually one frame longer than the behavioural ones, which is harmless here only because the surplus is on the neural side and trial ends are set by behavioural indices.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
T = len(offsets)
inp = np.vstack([np.arange(T, dtype=np.float32) * np.float32(TARGET_DT), ...])
```

iii. Step 4: "in the sampled session, all behavioral vectors and deconvolved/fluorescence arrays have 19,818 frames, with 349 ROIs. This makes direct temporal alignment possible." Step 29 reinforces it: "neural and behavioral arrays have identical lengths and are aligned by index".

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `BehavioralTimeSeries/environment` data stream.

ii.
```python
env = b['environment/data'][:]
...
ev = int(round(float(np.median(env[s:e]))))
ev = 1 if ev > 0 else 0
```

iii. Step 5: the AI enumerated the unique values of every behavioural stream and found `environment` takes a small set of discrete values that partition sessions into two groups, matching the paper's ENV1/ENV2 distinction; its whole-dataset scan printed the per-session environment sets (`[0]`, `[1]`, `[0, 1]` — the last on cross-environment switch days).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the stream over the trial's frames is taken, rounded, and mapped to binary (`>0 → 1`, else 0), then broadcast as a constant across all bins of that trial. So environment is treated as a per-trial constant even on sessions where the environment switches mid-session (it switches between trials, so this loses nothing).

ii.
```python
# Environment is a per-trial input. Use the NWB stream, converting ENV1/2 to 0/1.
ev = int(round(float(np.median(env[s:e]))))
ev = 1 if ev > 0 else 0
...
np.full(T, ev, np.float32),
```

iii. The AI initially believed the stream was coded −1/+1 (step 5: "environment is encoded -1/+1") and wrote the `>0` mapping to be robust to either coding; the comment in the code says "converting ENV1/2 to 0/1". Taking the median over the trial rather than the per-sample value is the same robustness reflex — it makes the input immune to isolated out-of-trial codes such as the −1 that the stream carries outside laps.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The `BehavioralTimeSeries/trial number` data stream, summarized per trial: the median of that stream over the trial's frames, rounded to an integer. It is not the loop counter.

ii.
```python
trial_num = b['trial number/data'][:]
...
tr = int(round(float(np.median(trial_num[s:e]))))
...
np.full(T, tr, np.float32),
```

iii. The AI knew from step 5 that the stream runs −1 outside trials and 0…N−1 within them, and from step 14 that its transitions do not coincide with `trial_start`. Taking the median over the `[trial_start, teleport)` window is its way of reading the stream's within-lap value while ignoring the boundary disagreement. The same value is then reused for two other purposes: ordering trials chronologically, and deciding whether a trial is before or after the reward switch.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Median + round, broadcast as a constant across the trial's bins as float32. The per-trial values are additionally used as a sort key to put trials in chronological order before the previous-outcome pass. The resulting range is 0–99, i.e. one 0-indexed count per session that resets each session.

ii.
```python
tr = int(round(float(np.median(trial_num[s:e]))))
...
trial_records.append((tr, rewarded, nbin, inp, out))
# Previous outcome follows actual chronological trial order, not list accidents.
trial_records.sort(key=lambda x: x[0])
```

iii. "Previous outcome follows actual chronological trial order, not list accidents" — the AI wanted a session-intrinsic trial index rather than one that depends on how it happened to enumerate laps. (Empirically the two coincide: in every session I checked, the per-trial median of `trial number` is exactly the sequential lap index 0,1,2,…, so the sort is a no-op.)

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From `BehavioralTimeSeries/Reward`, specifically its `timestamps` (the event times of reward delivery), combined with the `position/timestamps` of the trial window. Each trial's own reward outcome is computed first, then shifted by one trial.

ii.
```python
# Reward is a TimeSeries: event values are in data and event times in timestamps.
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
...
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
```

iii. The AI inspected the `Reward` group repeatedly (steps 8, 13, 14, 15), printing its `data` and `timestamps` datasets and their attributes, and established that rewards are sparse events carrying their own timestamps in the same seconds clock as the behavioural series (it compared `behavior t range` with the reward times) — hence an interval test rather than an index lookup.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Trials are sorted by trial number, then swept in order carrying a `prev` flag: the first trial of a session gets 0, every later trial gets the preceding trial's reward outcome (1 if any reward event fell inside that trial's `[trial_start, teleport)` window, 0 otherwise). The value is broadcast as a constant over the trial's bins. The input row is allocated as zeros during the trial pass and overwritten in the second pass.

ii.
```python
inp = np.vstack([
    ...
    np.zeros(T, np.float32),  # filled after chronological outcomes known
])
...
# Previous outcome follows actual chronological trial order, not list accidents.
trial_records.sort(key=lambda x: x[0])
prev = 0
for tr, rewarded, nbin, inp, out in trial_records:
    inp[3,:] = prev
    prev = rewarded
```

iii. The file docstring: "Reward outcome is presence of an NWB Reward event between trial start and teleport; previous outcome is chronological within session (first trial=0)." The two-pass structure exists so that "previous" is defined over the true chronological sequence rather than the order in which trials were appended, and the first-trial convention follows the instruction that the variable is binary (omitted = 0).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the binned `position` stream and the per-trial reward-zone interval. The zone interval is *not* read from the `reward_zone` behavioural stream; it comes from the scene name embedded in the NWB `identifier` (e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`) mapped through the repository's coordinate table, with switch sessions changing zone at trial 30.

ii.
```python
ZONE_COORDS = {'A': (80.,130.), 'B': (200.,250.), 'C': (320.,370.)}

def scene_info(identifier):
    scene = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
    scene = scene.rstrip('/').split('/')[-1]
    # LocationA, LocationA_to_B, and cross-environment A_to_Env2_B forms.
    import re
    locs = re.findall(r'(?:Location)?([ABC])', scene)
    if not locs:
        raise ValueError(f'Cannot identify reward location from {scene}')
    z0, z1 = locs[0], (locs[-1] if len(locs) > 1 else locs[0])
    switched = z1 != z0
    return scene, z0, z1, switched
...
zone = z1 if (switched and tr >= 30) else z0
lo, hi = ZONE_COORDS[zone]
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
dist_cls = discretize_distance(pbin, lo, hi)
```

iii. The AI first tried to use the `reward_zone` stream and rejected it (step 6: "`reward_zone` taking values 0–6 within every session strongly suggests it is not the per-trial A/B/C identity"), though it confirmed that the sparse non-zero samples sit near 80/200/320 cm. It then found the repository's authoritative mapping, `behavior.py::get_reward_zones(sess, rz_dict=None, change_trial=30)` with `reward_zone_dict = {'X': [80,130], 'Y': [200,250], 'Z': [320,370]}` and the `X→A, Y→B, Z→C` scene logic, and discovered (step 13) that "NWB `identifier` embeds the original scene, e.g. `Env1_LocationB_to_A`, so the repository's zone logic can be applied directly". It enumerated all 24 distinct scene strings across the dataset to make sure the regex covers the `LocationA`, `LocationA_to_B` and cross-environment `A_to_Env2_B` forms, and it grepped for `change_reward_trial` overrides, finding only the default of 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first block-averaged to the 258 ms grid; then the signed distance to the nearest edge of the trial's reward zone is computed — negative before the zone (`pos - lo`), positive after it (`pos - hi`), exactly 0 anywhere inside `[lo, hi]` — and immediately discretized into the 7 instructed classes. The continuous distance is never stored.

ii.
```python
def discretize_distance(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
    y = np.empty(d.shape, np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d < -10)] = 1
    y[(d >= -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y
```

iii. The instructions define distance "to any location in the reward zone", which the AI read as distance to the nearest point of the zone — hence 0 throughout the zone and a signed distance to the near edge outside it. The zone edges come from the repository's `reward_zone_dict` as described in 7-a.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes with explicit boolean masks, matching the instructed edges: `< -50 → 0`, `[-50,-10) → 1`, `[-10,0) → 2`, `== 0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `> 50 → 6`. Class 3 is reserved for being inside the zone. The upper boundaries are closed (a distance of exactly +10 cm lands in class 4), whereas the lower ones are half-open; with continuous position this affects a measure-zero set. Stored as `int8`.

ii.
```python
y[d < -50] = 0
y[(d >= -50) & (d < -10)] = 1
y[(d >= -10) & (d < 0)] = 2
y[d == 0] = 3
y[(d > 0) & (d <= 10)] = 4
y[(d > 10) & (d <= 50)] = 5
y[d > 50] = 6
```
```python
'output_values': [
    ['< -50 cm','-50 to -10 cm','-10 to <0 cm','in reward zone','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
    ...
```

iii. Direct transcription of the "Decoder Outputs" bin table in the instructions; the `d == 0` case is labelled "in reward zone" in `output_values`, making the semantics of class 3 explicit.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices and the same `block` partition as the neural data, so bin *k* of the output and bin *k* of the neural matrix cover the same 4 source frames. No lag or shift is introduced.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
```

iii. Same as 2-d/3-c: the NWB streams are frame-aligned, so shared indices imply temporal alignment. (Note the one asymmetry the AI did not comment on: neural activity is an average over the bin while the discretized position is derived from the average position over the same bin, so both are bin-averaged quantities — consistent with each other.)

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `BehavioralTimeSeries/position` stream, in cm along the 450 cm virtual corridor.

ii.
```python
pos = b['position/data'][:]
...
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. The AI characterized this stream early (step 5): "Behavioral position includes teleport values (notably -50 and -500)" — which is one reason it cuts trials at teleport onset, so that only in-corridor positions reach the output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Block-average over the 4 frames of each bin, then `np.digitize` into 5 classes. No wrapping, clipping or re-referencing.

ii.
```python
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. Implicit — the stream is already the quantity the instructions ask for ("Absolute position in corridor"); the only decision is how to aggregate within a bin, and the AI used the mean for all continuous behavioural quantities ("Behavioral continuous quantities are averaged, lick is any lick in a bin").

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes with edges at 90/180/270/360 cm — the 450 cm track split into 5 equal 90 cm bins — with the first and last classes left open so the handful of samples slightly outside [0, 450] fall into the end classes rather than out of range. `right=False` means a value exactly on an edge goes to the upper class.

ii.
```python
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```
```python
['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],
```

iii. Direct transcription of the instructions' 5-bin specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same indices and same binning as the neural data; see 7-d.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
```

iii. Frame-level alignment of the NWB streams, as established at step 4.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `BehavioralTimeSeries/lick` stream, which the AI found to be a per-frame lick count rather than a flag.

ii.
```python
lick = b['lick/data'][:]
...
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. Step 5: "Lick is a per-frame count (0–7), so it should become binary."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized with an OR over the bin: a bin is 1 if *any* of its (up to 4) source frames has a lick count > 0. Because the bins are 4× wider than the source frames, this inflates the positive class relative to per-frame binarization — 41.5% of bins are licks in the converted file.

ii.
```python
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```
```python
['no lick','lick'],
```

iii. The docstring states the rule: "Behavioral continuous quantities are averaged, lick is any lick in a bin." Averaging would not have produced a categorical variable, and the instructions require "Lick, time-varying. 0 = no, 1 = yes", so within the AI's chosen bin size an OR is the only binary-preserving aggregator.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same bin partition as the neural data: `offsets` are the absolute frame indices of each bin start, and each bin's lick window is `[q, min(q+block, e))`, i.e. exactly the frames that went into the corresponding neural bin (trial-end clipping included).

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. Frame-level alignment as above; the explicit `min(q+block, e)` cap is what keeps the last, partial bin from reaching past teleport onset into the next lap.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the NWB root `identifier` string — the original scene name — parsed for the reward-location letters, plus the `trial number` stream to decide which side of the reward switch a trial is on. The `reward_zone` behavioural stream is deliberately not used.

ii.
```python
scene, z0, z1, switched = scene_info(f['identifier'][()])
...
locs = re.findall(r'(?:Location)?([ABC])', scene)
z0, z1 = locs[0], (locs[-1] if len(locs) > 1 else locs[0])
switched = z1 != z0
...
tr = int(round(float(np.median(trial_num[s:e]))))
zone = z1 if (switched and tr >= 30) else z0
...
np.full(T, ZONE_ID[zone], np.int8),
```

iii. As in 7-a: the AI traced the repository's `get_reward_zones` (scene → `rz_dict['X'/'Y'/'Z']` → labels A/B/C, `change_trial = 30`), found the scene preserved in the NWB `identifier`, enumerated all 24 scene strings in the release to make sure its regex handles every form, and grepped for per-session `change_reward_trial` overrides (none are defined in `sessions_dict.py`, so the default 30 stands). Its stated decision: "Reward-zone coordinates and labels follow reward_relative.behavior: A/X=80--130, B/Y=200--250, C/Z=320--370 cm. NWB identifiers preserve scene names. Switches occur at trial 30, as in the task/repository default."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter is mapped to an integer through `ZONE_ID = {'A':0,'B':1,'C':2}` and broadcast as a constant over all bins of the trial. On non-switch sessions `z0 == z1` and the trial index is irrelevant; on switch sessions (including the cross-environment `Env1_A_to_Env2_B` form) trials 0–29 get the first letter and trials ≥ 30 the second. Resulting class balance is 32.9% / 33.7% / 33.4%.

ii.
```python
ZONE_ID = {'A':0, 'B':1, 'C':2}
...
zone = z1 if (switched and tr >= 30) else z0
out = np.vstack([
    dist_cls, pos_cls, speed_cls, lbin,
    np.full(T, ZONE_ID[zone], np.int8),
    np.full(T, rewarded, np.int8),
])
```

iii. See 10-a; the instructions require the per-trial encoding 0 = A, 1 = B, 2 = C, and the AI chose to emit it as a constant time series because the format asks for time-varying outputs "if at all possible".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The event timestamps of `BehavioralTimeSeries/Reward`, tested against the trial's time window taken from `position/timestamps`.

ii.
```python
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
...
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
```

iii. Established by direct inspection of the `Reward` group (steps 8/13/14/15), which showed a sparse event series with its own `timestamps` in the same seconds clock as the behavioural streams; the AI also noted `autoreward` exists as a separate stream and did not fold it in.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial: 1 if at least one reward timestamp falls in `[t_start, t_lastframe + 1/rate)`, i.e. the half-open interval covering the trial's frames up to but not including the teleport frame; 0 otherwise. Broadcast as a constant over the trial's bins, `int8`. 84.3% of trials are rewarded. The same value also feeds the previous-trial-outcome input.

ii.
```python
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
...
np.full(T, rewarded, np.int8),
```

iii. Docstring: "Reward outcome is presence of an NWB Reward event between trial start and teleport". Extending the window by one frame past the last in-trial sample is what makes the interval cover the whole final frame rather than ending at its leading edge; rewards delivered during the teleport period are excluded along with the teleport samples themselves.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter is mostly fail-fast rather than repair-in-place:
- **Unparseable scene**: `raise ValueError(f'Cannot identify reward location from {scene}')`.
- **Plane rates disagreeing within a session**: raise.
- **Unexpected frame interval** (not an integer divisor of the target bin): raise.
- **ROI-count mismatch** between a plane's data matrix and its slice of the `iscell`/`planeIdx` table: raise. This one actually fired, on the first two-plane session, and was fixed by generalizing the ROI selection rather than by skipping data.
- **Clipped final lap** with no teleport: the trial is silently dropped.
- **Trials too short** to form 2 bins: silently dropped.
- **Missing `Reward/timestamps`**: falls back to an empty array, so the session's trials are all scored unrewarded.
- **Sessions with < 2 usable trials**: skipped with a printed `SKIP` message.
- **Ragged final bin**: kept, averaged over whatever frames remain rather than zero-padded or discarded.

Not handled: there is no check that the ophys arrays and the behavioural arrays have the same length, and no assertion that reward event times land within the sampled window. In 10 of the 152 sessions the ophys arrays are in fact one frame longer than the behavioural ones; this is benign here only because the extra frame is at the end of the neural stream and all trial boundaries are behavioural indices.

ii.
```python
if not np.allclose(rates, rates[0]):
    raise ValueError(f'Plane frame rates disagree in {path}: {rates}')
...
if not np.isclose(block * dt, TARGET_DT, rtol=0, atol=2e-6):
    raise ValueError(f'Unexpected aligned-frame interval {dt} in {path}')
...
if dset.shape[1] != len(mask):
    raise ValueError(f'ROI count mismatch for {name}: {dset.shape[1]} vs {len(mask)}')
...
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
...
if len(nt) < 2:
    print('SKIP (<2 trials)', p, flush=True)
    continue
```

iii. The guards were added as the AI's assumptions were formalized — the rate check after it discovered the 31 Hz/15.5 Hz discrepancy, the ROI check before it knew multi-plane sessions existed. Its reaction to the one failure (step 19) is the clearest statement of intent: rather than skipping the session, "the converter should select curated cells within each plane and concatenate their deconvolved activity", i.e. mistakes are treated as signs of an unhandled data layout, to be fixed, not tolerated.

## 13-a. What are the most time-consuming steps of the code?

i. In rough order:
1. **Per-trial HDF5 reads of the deconvolved matrices** — `dset[s:e, :][:, idx]` is issued once per trial per plane (≈ 12,200 reads), each pulling *all* ROIs for that frame range out of the compressed chunks before subsetting to accepted cells. This is the dominant I/O cost and the reason the whole-dataset pass is bounded by decompression rather than computation.
2. **`binned_mean` over the neural block** — a Python-level list comprehension that calls `np.mean` once per bin, ~670,000 small reductions over ~2.4 billion source values in total.
3. **Pickling and writing the 1.15 GiB output**, done in one `pickle.dump` at the end.
4. Reading the whole behavioural streams (`pos`, `speed`, `lick`, `env`, `trial_num`, `trial_start`, `teleport`, `timestamps`) per session — cheap by comparison.

The full run over 152 sessions completed in roughly a minute and a half of wall clock in the trajectory, so none of these is pathological.

ii. N/A (see the snippets in 2-a and 2-b).

iii. Not discussed explicitly in the trajectory. The per-trial read pattern is a deliberate memory trade: the AI avoided materializing full-session deconvolved matrices (it had computed that the dataset is ~7.9 GiB at native resolution) by reading only the frames a trial needs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- **`binned_mean`**: for the common case where the trial length is an exact multiple of `block`, the whole loop is `x[:n*block].reshape(n, block, -1).mean(axis=1)`, with the single ragged tail handled separately. As written it builds a Python list of per-bin means and `np.stack`s them.
- **The lick comprehension**: `np.any(lick[q:q+block] > 0)` per bin is the same reshape-and-reduce (`.any(axis=1)`).
- **The per-trial loop in `convert_session`**: `discretize_distance`, both `np.digitize` calls and the reward-interval test could be computed once over the whole session and then sliced, since the reward zone is the only per-trial-varying parameter and it is piecewise constant across the switch. The per-trial HDF5 reads could likewise be replaced by one read per plane per session, at a memory cost.

Additionally, `discretize_distance` itself evaluates 7 full boolean masks (plus two `np.where` passes) over the data; `np.digitize` with the same edges would be one pass.

ii. N/A.

iii. Not discussed in the trajectory. The per-trial structure follows naturally from variable-length trials and from the AI's decision to keep memory small by reading each trial's neural slice on demand.

## 13-c. What processing does the code repeat multiple times?

i. - `np.arange(0, e-s, block)` is recomputed three times per trial (for the neural, position and speed bins) and is just `offsets - s`, which is already in hand.
- `binned_mean` is called three times per trial, each re-entering the same Python loop over the same bin partition; the three streams could share one set of slice bounds.
- `import re` sits inside `scene_info`, so the import machinery is re-entered once per session (cheap, but it belongs at module scope).
- The trial's frame window `[s:e]` is re-sliced from each behavioural array separately (`pos[s:e]`, `speed[s:e]`, `env[s:e]`, `trial_num[s:e]`), each producing a copy.
- `np.flatnonzero(teleport[s:])` rescans the tail of the teleport array from every trial start, making trial-boundary detection O(n_trials × T) where one pass over rising edges would be O(T).

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
sbin = binned_mean(speed[s:e], np.arange(0, e-s, block), block)
```
```python
    tq = np.flatnonzero(teleport[s:] > 0)
```

iii. Not discussed. Unlike the reference solution, this converter makes only a single pass over the dataset — it needs no survey pass, because the reward-zone identity comes from file metadata rather than from a fit across all trials.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and all of it cheap:
- The ophys `starting_time.rate` attributes are read and cross-checked for every plane, but the value is then discarded: the authoritative interval is recomputed from `position/timestamps`. The `rates` list survives only as a consistency assertion, and `rate` itself is used only for the metadata record and for the one-frame extension of the reward window.
- `trial_records.sort(key=lambda x: x[0])` is a no-op in this dataset — the NWB trial numbers are already the sequential lap order in every session — as is the two-pass structure it exists to support.
- `scene_info` returns `scene` and `switched` for every session, but `switched` is redundant with `z0 != z1` and `scene` is only stored in metadata.
- Per-trial constants (environment, trial number, previous outcome, reward zone, reward outcome) are materialized as full-length time series; the format permits a scalar per trial, so five of the ten rows are replicated ~55× per trial. This matches how the reference solution stores them and is what the instructions ask for ("If at all possible, make it time-varying"), so it is arguably not waste.
- Each behavioural stream is read in full for every session even though only in-lap frames are ever emitted (teleport periods are read and thrown away) — a small fraction of total I/O.

ii.
```python
rates = [float(dg[n]['starting_time'].attrs['rate']) for n in plane_names]
if not np.allclose(rates, rates[0]):
    raise ValueError(f'Plane frame rates disagree in {path}: {rates}')
...
rate = 1.0 / dt
```
```python
trial_records.sort(key=lambda x: x[0])
```

iii. Not discussed in the trajectory. The retained rate check is a deliberate leftover from the 31 Hz/15.5 Hz investigation — the AI kept it as a guard even after it stopped trusting the attribute for the bin size.
