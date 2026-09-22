# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the frozen `bwm_release.csv`, chooses one representative EID per subject, constructs its cache path, and directly loads revisioned parquet/NumPy files. Thus its “full” conversion loads 134 retained sessions, not every eligible released session.

ii.
```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
first_idx = by_subject.groups[subject][0]
row = bwm.loc[first_idx]
return pd.read_parquet(p), p
```

iii. The notes say this reproduces `0_data_caching.py` (seed 42, first-listed EID per subject) and avoids treating repeated sessions as independent. They report 139 candidates and 134 retained sessions.

## 1-b. How are the data split into subjects?

i. Subjects come from the CSV `subject` column. One candidate session is chosen per unique subject; retained names are sorted and each session receives a `subject_idx`.

ii.
```python
subjects = np.unique(bwm.subject)
by_subject = bwm.groupby('subject', sort=True)
subjects = sorted(set(s['info']['subject'] for s in sessions))
subject_map = {v:i for i,v in enumerate(subjects)}
```

iii. The agent states that the frozen CSV provides subject identity and that one-session-per-subject follows its reading of the reference caching workflow.

## 1-c. How are the data split into sessions?

i. Each chosen EID is a session. Its directory is constructed from lab, subject, date, and session number, then it is processed independently.

ii.
```python
def session_dir(row):
    return (SOURCE / str(row.lab) / 'Subjects' / str(row.subject) /
            str(row.date) / f'{int(row.session_number):03d}')
z = process_session(item, br)
sessions.append(z)
```

iii. It justified the EID as the natural session unit and merged probes sharing the same trials and behavior.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. Retained row indices select trial event times; neural and behavioral windows are made per selected row.

ii.
```python
trial_idx = np.flatnonzero(mask)
selected = trials.iloc[trial_idx]
neural = bin_spikes(probes, selected.stimOn_times.to_numpy(float))
```

iii. The notes treat the trials table as authoritative and preserve original row indices so filtering leaves visible gaps.

## 1-e. How are trials filtered based on quality controls?

i. Trials require finite fields, behavior coverage, reaction time 0.08–2 s, feedback after stimulus, positive duration no longer than 10 s, choice ±1, and prior near 0.2/0.5/0.8. Sessions with fewer than two survivors are skipped.

ii.
```python
mask = (finite & behavior_coverage & (rt >= 0.08) & (rt <= 2.0) &
        (feedback >= stim) & (duration > 0) & (duration <= 10.0) &
        np.isin(choice, [-1.0, 1.0]) & valid_prior)
```

iii. The agent says this combines reference event/duration checks, paper reaction-time QC, categorical validity, and no extrapolation for behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `spikes.times.npy` and `spikes.clusters.npy`; cluster metrics/channels and channel atlas IDs determine unit QC and Beryl labels.

ii.
```python
metrics = pd.read_parquet(metrics_p)
ch = np.load(parent / 'clusters.channels.npy', mmap_mode='r')
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
channel_ids = np.load(parent / 'channels.brainLocationIds_ccf_2017.npy', mmap_mode='r')
```

iii. The notes identify spike times/assignments as the signal and cluster/anatomy files as curation metadata.

## 2-b. How is the `neural` data processed?

i. Good units from all listed probes are concatenated. Per trial/probe, spikes in −0.5 to +1.5 s are assigned to 20 ms bins and accumulated. Stored values are raw spike counts (`float32`), not rates or smoothed activity.

ii.
```python
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
flat = local[ok].astype(np.int64) * N_BINS + tb[ok]
counts = np.bincount(flat, minlength=p['n_good'] * N_BINS)
mat[off:off+p['n_good']] += counts.reshape(p['n_good'], N_BINS).astype(np.float32)
```

iii. The agent explicitly interpreted the supplied method code as using spike counts and documented “counts, not smoothed rates.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units require cluster `label >= 1`, a valid channel, and a Beryl label other than root, void, nan, none, or empty. Empty probes/sessions are dropped.

ii.
```python
bad_names = np.isin(np.char.lower(regions), ['root', 'void', 'nan', 'none', ''])
good = (labels >= 1.0) & anatomical_ok & (~bad_names)
```

iii. The notes equate `label >= 1` with all RIGOR criteria and describe the anatomy cut as retaining good grey-matter neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike-bin edges are each trial’s `stimOn_times` plus a fixed −0.5 to +1.5 s grid.

ii.
```python
abs_edges = stim + EDGES_REL
lo = int(np.searchsorted(ts, abs_edges[0], side='left'))
hi = int(np.searchsorted(ts, abs_edges[-1], side='left'))
```

iii. The agent chose the exact stimulus-onset alignment/window it found in the reference caching parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There are 100 non-overlapping 20 ms bins over two seconds. Spikes are histogrammed once; no smoothing or later rebinning occurs.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
```

iii. The agent says these are the exact reference caching parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the fixed relative bin grid; `stimOn_times` supplies absolute alignment but the same relative vector is stored for every trial.

ii.
```python
TIME_REL = EDGES_REL[1:].astype(np.float32)
inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)])
```

iii. The notes say continuous time was requested and identify these as behavior sampling times/right bin edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The 100 right edges of the bin grid are used, producing −0.48 through 1.50 s at 20 ms spacing, cast to `float32` and repeated per trial.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, N_BINS + 1)
TIME_REL = EDGES_REL[1:].astype(np.float32)
```

iii. It claims right-edge sampling matches the reference behavior interpolation semantics.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The row uses each neural bin’s right edge, while neural counts cover left-closed/right-open intervals. Behavioral targets use those same right-edge times.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
tb = np.floor((t - abs_edges[0]) / BIN + 1e-10).astype(np.int32)
```

iii. The agent considered this a shared stimulus-aligned grid and reported no shift in its plots.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Despite the requested variable, it uses the original zero-based row index within the entire session; blocks are not derived from `probabilityLeft`.

ii.
```python
for raw_i, (_, tr) in zip(trial_idx, selected.iterrows()):
    inp = np.vstack([TIME_REL, np.full(N_BINS, raw_i, dtype=np.float32)])
```

iii. The notes say original indices preserve QC gaps and “true trial position/block progression”; the saved name is `trial_number_in_session`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. No block reset occurs. The raw session index is cast to `float32` and broadcast over 100 timepoints.

ii.
```python
np.full(N_BINS, raw_i, dtype=np.float32)
'input_names':['time_since_stimulus_onset_s','trial_number_in_session']
```

iii. The agent preferred uncompressed session order to preserve gaps after QC.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`; values other than −1/+1 are removed.

ii.
```python
choice = trials.choice.to_numpy(float)
np.isin(choice, [-1.0, 1.0])
```

iii. The notes identify the native IBL choice field and exclude no-go trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw −1 is mapped to left/0 and +1 to right/1, then repeated through all time bins.

ii.
```python
choice = 0 if float(tr.choice) == -1 else 1
out[0] = choice
```

iii. The agent says IBL −1 denotes a left wheel turn and +1 right; repetition gives a uniform 4×T output.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft`.

ii.
```python
prior = trials.probabilityLeft.to_numpy(float)
valid_prior = np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(1)
```

iii. The agent describes this as the blockwise left-stimulus prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The closest valid level 0.2/0.5/0.8 is encoded 0/1/2 and repeated across time.

ii.
```python
prior_levels = np.array([0.2, 0.5, 0.8])
prior = int(np.argmin(np.abs(prior_levels - float(tr.probabilityLeft))))
out[1] = prior
```

iii. The mapping is required; repetition provides a common temporal output shape.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses wheel timestamps and position.

ii.
```python
wt = np.load(choose_file(alf, '_ibl_wheel.timestamps.npy'), mmap_mode='r')
wp = np.load(choose_file(alf, '_ibl_wheel.position.npy'), mmap_mode='r')
```

iii. The notes follow IBL/reference wheel primitives rather than treating position differences as ready speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Nonfinite/duplicate samples are removed, position is interpolated to 1 kHz, filtered velocity is computed, absolute value is taken, and speed is interpolated at trial right-edge times.

ii.
```python
_, unique_idx = np.unique(wt0, return_index=True)
pos_i, time_i = wheel_utils.interpolate_position(wt0, wp0, freq=1000, kind='linear')
vel_i, _ = wheel_utils.velocity_filtered(pos_i, fs=1000)
speed_i = np.abs(vel_i)
wheel_vals[i] = np.interp(targets[i], time_i, speed_i).astype(np.float32)
```

iii. The agent says these are exact IBL primitives; deduplication repaired a real malformed timestamp.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Dataset-pooled empirical tertiles over all retained aligned samples define low/medium/high.

ii.
```python
wheel_all = np.concatenate([s['wheel'].ravel() for s in sessions])
wthr = np.quantile(wheel_all, [1/3, 2/3]).astype(float)
out[2] = np.searchsorted(wthr, w, side='right').astype(np.int8)
```

iii. The agent chose pooled thresholds for deterministic, balanced classes, with each aligned bin contributing once.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated at `stimOn_times + TIME_REL`, the right edge associated with every neural bin; only covered trials remain.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
coverage = (targets[:, 0] >= time_i[0]) & (targets[:, -1] <= time_i[-1])
wheel_vals[i] = np.interp(targets[i], time_i, speed_i)
```

iii. The notes cite common stimulus alignment and no extrapolation.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera times/ROI motion energy when valid, otherwise the right camera.

ii.
```python
for side in ('left', 'right'):
    ct = np.load(choose_file(alf, f'_ibl_{side}Camera.times.npy'), mmap_mode='r')
    cv = np.load(choose_file(alf, f'{side}Camera.ROIMotionEnergy.npy'), mmap_mode='r')
```

iii. Left-first/right-fallback is documented as matching reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Released values are filtered for finite, unique, increasing timestamps, then linearly interpolated at trial right-edge times. No normalization/smoothing is applied.

ii.
```python
cf = np.isfinite(ct) & np.isfinite(cv)
_, ci = np.unique(ct0, return_index=True)
whisk_vals[i] = np.interp(targets[i], ct, cv).astype(np.float32)
```

iii. The trace is otherwise used directly; deduplication is robustness for malformed timestamps.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Dataset-pooled 1/3 and 2/3 quantiles define low/medium/high using right-sided `searchsorted`.

ii.
```python
whisk_all = np.concatenate([s['whisk'].ravel() for s in sessions])
qthr = np.quantile(whisk_all, [1/3, 2/3]).astype(float)
out[3] = np.searchsorted(qthr, q, side='right').astype(np.int8)
```

iii. The agent used the wheel balancing rationale and stores thresholds in metadata.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated at `stimOn_times + TIME_REL`; incomplete windows are excluded.

ii.
```python
targets = stim[:, None] + TIME_REL[None, :]
coverage = coverage & (targets[:, 0] >= ct[0]) & (targets[:, -1] <= ct[-1])
whisk_vals[i] = np.interp(targets[i], ct, cv)
```

iii. The notes describe a shared stimulus-onset grid and prohibit extrapolation/imputation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The newest coherent revision is selected; nonfinite/duplicate behavior samples are removed; camera fallback is attempted. Missing streams, insufficient trials, or no good neurons skip a session; uncovered trials are dropped and exceptions logged, not imputed.

ii.
```python
try:
    z = process_session(item, br)
except Exception as e:
    skipped.append({'eid': item['eid'], 'reason': f'{type(e).__name__}: {e}'})
if len(trial_idx) < 2:
    raise ValueError(f'only {len(trial_idx)} valid trials')
```

iii. The stated policy is never to synthesize/extrapolate. Duplicate repair is justified as preferable to dropping an otherwise usable session.

## 10-a. What are the most time-consuming steps of the code?

i. Reading/searching huge spike arrays and per-trial spike binning dominate core work; serializing the multi-gigabyte pickle may dominate total wall time.

ii.
```python
st = np.load(parent / 'spikes.times.npy', mmap_mode='r')
sc = np.load(parent / 'spikes.clusters.npy', mmap_mode='r')
for k, stim in enumerate(stim_times):
```

iii. The notes say probes can exceed 50 million spikes, motivating memory maps and windowed `searchsorted`; pickle I/O may dominate.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Behavior interpolation over trials, nested spike trial/probe loops, input construction, and output construction could be batched further. Expensive inner work is already vectorized.

ii.
```python
for i in np.flatnonzero(coverage & np.isfinite(stim)):
for k, stim in enumerate(stim_times):
    for p, off in zip(probes, offsets):
for (choice,prior), w, q in zip(s['scalar'],s['wheel'],s['whisk']):
```

iii. The agent emphasizes avoiding loops over individual spikes and vectorizing lookup, time-bin assignment, and accumulation.

## 10-c. What processing does the code repeat multiple times?

i. The BWM CSV is reread per session; recursive file searches recur per required file; similar finite/deduplication logic is separately applied to wheel and camera.

ii.
```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
candidates = [p for p in root.rglob(basename) ...]
```

iii. The notes do not explicitly justify these repetitions; they emphasize self-contained, memory-bounded session processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads/stores neuron UUIDs and extensive provenance unused by the decoder, repeatedly invokes garbage collection, and defines unused `SESSIONS_PQT`, `DATASETS_PQT`, `target_n`, and loop variable `k`. Optional plots are diagnostic.

ii.
```python
uuids_p = parent / 'clusters.uuids.csv'
'neuron_uuids': uuids.tolist()
target_n = 2 if sample else len(candidates)
gc.collect()
```

iii. UUIDs/metadata support provenance and duplicate checks; plots are optional. Unused variables are not justified.

