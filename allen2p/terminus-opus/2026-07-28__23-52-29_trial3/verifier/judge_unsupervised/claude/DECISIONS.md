# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files using the Allen SDK's `BehaviorOphysExperiment.from_nwb()`. It first reads an experiment metadata CSV (`ophys_experiment_table.csv`) to identify available experiments, then cross-references with NWB files on disk. It filters to active (non-passive) experiments, giving 202 experiments. For each experiment, it loads neural events, ophys timestamps, trials, stimulus presentations, running speed, and eye tracking data.

ii.
```python
def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0])
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
    ...

def load_experiment(eid):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
        return {
            'ophys_timestamps': ds.ophys_timestamps.copy(),
            'events': np.vstack(ds.events.events.values).astype(np.float32),
            'trials': ds.trials.copy(),
            'stimulus_presentations': ds.stimulus_presentations.copy(),
            'running_speed': ds.running_speed.copy(),
            'eye_tracking': ds.eye_tracking.copy(),
            'metadata': dict(ds.metadata),
        }
```

iii. The AI justified using the Allen SDK for data loading as it's the standard API for this dataset. The decision to filter to `passive==False` comes from the instruction to use the "Visual Behavior" task (active behavior, not passive replay). The AI documented that it uses all 202 active experiments from the available NWB files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from the experiment metadata table. A sorted list of unique mouse_id strings is created. Each session is mapped to a subject index via `subject_idx`.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
# ...per experiment:
asi.append(s2i[str(row['mouse_id'])])
```

iii. The AI noted 38 unique mice across the 202 active experiments. This matches the metadata table. The CONVERSION_NOTES confirms "Subjects: 38".

## 1-c. How are the data split into sessions?

i. Each ophys experiment (imaging plane) is treated as a separate session. For single-plane (CAM2P) experiments, this is one session per ophys_session. For multi-plane (MESO) experiments, each imaging plane within the same ophys_session becomes its own session. This results in 202 sessions.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
    asi.append(s2i[str(row['mouse_id'])])
    abri.append(np.full(nn, r2i[row['targeted_structure']], dtype=np.int64))
```

iii. The AI's CONVERSION_NOTES state "202 active experiments" as sessions. The trajectory shows the agent recognized that MESO experiments have multiple imaging planes per session but chose to treat each experiment (plane) as a separate session. This means MESO sessions with shared behavioral data appear as multiple sessions with the same behavioral outputs but different neural data.

## 1-d. How are the data split into trials?

i. Trials are extracted from the `trials` table of each experiment. Only trials that are Go or Catch (and not aborted or auto-rewarded) are included. For each valid trial, neural and output data are extracted based on the trial's `start_time` and `stop_time`. Trials with fewer than 2 time bins are skipped.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
# ...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
```

iii. The AI justified this based on the instructions which explicitly say "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 1-e. How are trials filtered based on quality controls?

i. No additional quality filtering is applied beyond the Go/Catch selection and minimum 2-bin requirement. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if len(vt) == 0: return [], [], nn
# ...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
    skipped += 1; continue
```

iii. The AI's CONVERSION_NOTES state that the minimum trial requirement ensures the decoder has at least two trials per session for train/test splitting.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `events` attribute of `BehaviorOphysExperiment`, which provides deconvolved calcium events (not dF/F traces).

ii.
```python
'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. The AI's CONVERSION_NOTES state: "Paper uses 'events' (deconvolved calcium events) for neural analysis." The trajectory shows the agent identified this from the paper's methods which mention "detected calcium events."

## 2-b. How is the `neural` data processed?

i. The neural events are resampled from their native ophys timestamps to a uniform 93ms time grid. The resampling uses bin averaging: for each target time bin, the mean of all ophys frames within a 93ms window centered on the bin center is computed. If no frames fall within a bin, the previous bin's value is carried forward.

ii.
```python
TARGET_BIN_SIZE = 0.093

def resample_session(events, ophys_ts, bc):
    h = TARGET_BIN_SIZE / 2
    nn, nb = events.shape[0], len(bc)
    out = np.zeros((nn, nb), dtype=np.float32)
    li = np.searchsorted(ophys_ts, bc - h, side='left')
    ri = np.searchsorted(ophys_ts, bc + h, side='left')
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
    return out
```

iii. The AI chose 93ms as the bin size to match the MESO frame rate (~11Hz), applying this uniformly to both MESO (~93ms native) and CAM2P (~32ms native) experiments. The CONVERSION_NOTES state: "Resample all to ~93ms bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural quality filtering is applied. The AI relies on the NWB files already containing only valid ROIs (cells that passed the Allen Institute's multi-label classifier filtering with `valid_roi=True`).

ii. No explicit filtering code exists; the events loaded from the NWB are used directly.

iii. The AI's CONVERSION_NOTES state: "ROI filtering already applied (valid_roi=True)" and "The filtering is already applied to the data (invalid ROIs are excluded)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. A continuous time grid of bin centers is created starting from the earliest trial start time (with half-bin offset). Neural data is resampled to this grid for the entire session, then sliced per trial using `start_time` and `stop_time` boundaries.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
# ...
nr = resample_session(ev, ots, bc)
# ...per trial:
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
ntl.append(nr[:, ti])
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI creates a uniform time grid across the session and resamples ophys data to it, then extracts per-trial segments. The metadata records `temporal_alignment_event: 'Trial start time'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 93ms (0.093 seconds). Rebinning is applied: CAM2P data (native ~32ms, ~31Hz) is downsampled by averaging, while MESO data (native ~93ms, ~11Hz) is approximately preserved at its native resolution.

ii.
```python
TARGET_BIN_SIZE = 0.093
```

iii. The AI chose 93ms to match the MESO frame rate. The CONVERSION_NOTES state "93ms bins" and the metadata reports `time_bin_size: 93.0` (in ms).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`, filtered to the change detection stimulus block and excluding omitted stimuli.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. The AI filters to the `change_detection` stimulus block to ensure only task-relevant images are used. 16 unique image names were identified across all experiments.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each image name is mapped to a categorical integer index (0-15). For each time bin, the most recently presented image is determined using `searchsorted` on stimulus start times. Time bins before the first stimulus get index 0.

ii.
```python
n2i = {n: i for i, n in enumerate(imn)}
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. The AI uses a forward-fill approach: each time bin gets the identity of the most recently presented (non-omitted) image. This is documented as "Categorical, time-varying" in the CONVERSION_NOTES mapping.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same time bin grid as the neural data (93ms bins), using stimulus presentation start times matched to bin centers via searchsorted. Per-trial extraction uses the same time indices.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
# ...per trial:
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The shared time grid ensures alignment between neural and output data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of `stimulus_presentations`, filtered to the change detection block.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The AI uses the Allen SDK's `is_change` flag which marks stimulus presentations where the image identity changed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is created. For each change event, a 750ms window after the change start time is marked as 1. All other time bins are 0.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The 750ms window corresponds to the image presentation interval (250ms stimulus + 500ms inter-stimulus). The AI's CONVERSION_NOTES describe this as "Binary, time-varying."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. The output values are ['no_change', 'change'].

ii.
```python
ov = [imn, ['no_change','change'], ...]
```

iii. The instructions specify "binary variable" for image change, which is directly implemented.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same session-wide time grid as the neural data, then extracted per trial using the same time indices.

ii.
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
# per trial:
otl.append(np.stack([ib[ti], cb[ti], ...]))
```

iii. Same alignment mechanism as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the `running_speed` table's `speed` column, with corresponding `timestamps`.

ii.
```python
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. The AI loads running speed from `ds.running_speed` which provides the animal's running speed on the running wheel.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native sampling rate (~60Hz) to the 93ms time bin centers using linear interpolation. NaN values are excluded before interpolation. The interpolated values are then discretized into 5 percentile bins computed globally across all experiments.

ii.
```python
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
rb = dig(ri, rbe)
```

iii. The AI used global percentile bins (computed via `fast_collect_stats` using h5py with 10x subsampling) to ensure consistent binning across all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-4). Bin edges are computed from percentiles [0, 20, 40, 60, 80, 100] of all non-NaN running speed values across all experiments. NaN values are assigned to the middle bin (bin 2).

ii.
```python
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. The AI chose global percentile bins so that bin definitions are consistent across sessions. NaN values are assigned to the middle bin as a default.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 93ms time bin centers used for neural data, ensuring temporal alignment. Per-trial extraction uses the same time indices.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
```

iii. The shared time grid ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area`, which is the pupil area (not diameter) from the eye tracking data.

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. The AI loads `pupil_area` from the eye tracking data. The CONVERSION_NOTES mapping states "eye_tracking.pupil_area -> output[3]: pupil_diameter". The output is labeled "pupil_diameter" but the underlying data is pupil area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is interpolated from its native sampling rate (~30Hz) to 93ms bin centers using linear interpolation (excluding NaN values). The interpolated values are then discretized into 5 percentile bins computed globally.

ii.
```python
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc)
pb = dig(pi, pbe)
```

iii. Same approach as running speed: linear interpolation followed by global percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins (0-4) computed globally across all experiments. NaN values assigned to middle bin (bin 2).

ii.
```python
pbe = pct_bins(pa_all, 5)
pb = dig(pi, pbe)
```

iii. The AI uses the same `pct_bins` and `dig` functions for both running speed and pupil data.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: interpolation to the shared 93ms time grid, per-trial extraction with same indices.

ii.
```python
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', ...)(bc)
```

iii. Same alignment approach as all other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm` columns of the trials table. The fourth category (correct_reject) is inferred as the else case.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. The AI maps trial outcomes to integers: hit=0, miss=1, false_alarm=2, correct_reject=3. These correspond to the standard trial types in the Visual Behavior task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is a static per-trial variable. The outcome integer is replicated across all time bins within the trial.

ii.
```python
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The instructions specify trial outcome as "Static per-trial," so the same value fills all time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used:
- **NaN in running/pupil data**: Excluded from interpolation; NaN interpolated values assigned to middle percentile bin (bin 2).
- **Missing eye tracking**: If fewer than 2 valid pupil data points, entire session gets NaN (then middle bin).
- **Empty neural bins**: If no ophys frames fall in a resampled bin, the previous bin's value is forward-filled.
- **Short trials**: Trials with fewer than 2 time bins are skipped.
- **Empty sessions**: Sessions with fewer than 2 valid trials are skipped entirely.
- **Unknown image names**: Mapped to index 0 via `n2i.get(n, 0)`.

ii.
```python
# Forward fill for neural
elif b > 0: out[:, b] = out[:, b-1]

# NaN -> middle bin
b[np.isnan(v)] = (len(e) - 1) // 2

# Skip short trials
if len(ti) < 2: continue

# Skip empty sessions
if len(ntl) < 2: skipped += 1; continue
```

iii. The CONVERSION_NOTES mention "2,621 trials with all-zero neural data (expected for sparse calcium events)" which the AI accepted as normal behavior for sparse event data.

## 9-a. What are the most time-consuming steps of the code?

i. Based on the conversion output timing, the most time-consuming step is loading experiments via the Allen SDK (`load_experiment`), averaging ~5.5 seconds per experiment. For 202 experiments, this totals ~18 minutes. The h5py stats collection takes ~20 seconds total. Processing each experiment takes ~0.5 seconds.

ii.
```python
# Timing from conversion_full_out.txt:
# [1/202] Exp 775614751: 89n 39t load=5.3s proc=0.3s
# [24/202] Exp 822024770: 436n 260t load=9.9s proc=0.8s
```

iii. The CONVERSION_NOTES include a timing table estimating ~20 minutes total, dominated by SDK loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_session` function contains a Python loop over all time bins that could be vectorized:

ii.
```python
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
    elif b > 0: out[:, b] = out[:, b-1]
```

Additionally, the image change window marking loop iterates over change times:
```python
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

And the trial extraction loop iterates per trial:
```python
for _, t in vt.iterrows():
    ...
```

iii. The AI's CONVERSION_NOTES mention "Session-wide resampling then trial extraction (vs per-trial resampling)" as an optimization, but the resampling itself still uses a Python loop.

## 9-c. What processing does the code repeat multiple times?

i. The code performs two passes over all NWB files:
1. `fast_collect_stats` (h5py pass): Collects image names and running/pupil statistics for global percentile bin computation.
2. Main processing loop: Loads each experiment again via SDK for full processing.

The running speed and pupil data are loaded in both passes.

ii.
```python
# Pass 1: fast_collect_stats
imn, rs_all, pa_all = fast_collect_stats(el)

# Pass 2: main loop
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
    ...
```

iii. The AI justified the two-pass approach: the first pass uses h5py for speed (~0.1s/experiment) to compute global statistics, while the second uses the full SDK for proper data loading.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the full experiment metadata dict (`ds.metadata`) via the SDK but only uses `mouse_id` and `targeted_structure` from the experiment table (not from metadata). The metadata is loaded but discarded after processing. The eye tracking data loads many columns (pupil area, width, height, etc.) but only uses `pupil_area` and `timestamps`. Similarly, the full stimulus_presentations table is loaded but only a subset of columns are used.

ii.
```python
'metadata': dict(ds.metadata),  # loaded but not used from experiment data
'eye_tracking': ds.eye_tracking.copy(),  # loads all eye tracking columns
```

iii. The AI did not specifically document this as a concern, but noted memory-efficient processing by deleting experiment data after use (`del ed; gc.collect()`).
