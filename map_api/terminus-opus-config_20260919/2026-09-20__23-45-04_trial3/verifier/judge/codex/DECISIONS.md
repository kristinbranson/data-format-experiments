# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files with a sorted glob over `/app/data/sub-*/*.nwb`, optionally slices to 2 files for `--sample`, and processes each file once with `pynwb.NWBHDF5IO`. Inside each file it reads the trials table, behavioral event series, units table, and electrodes table.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
```

```python
trials = nwb.intervals['trials']
df = trials.to_dataframe()
...
be = nwb.acquisition['BehavioralEvents'].time_series
...
units = nwb.units
...
electrodes = nwb.electrodes.to_dataframe()
```

iii. In `CONVERSION_NOTES.md` the AI says the NWB release is organized as `sub-<subject_id>/sub-<id>_ses-...nwb`, that there are 174 session files, and that all data must be loaded through `pynwb`. It also notes in Step 6 that each session is opened once and processed once.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.subject_id` as the subject identifier for each session, then constructs `subjects` as the sorted unique subject IDs and `subject_idx` as the per-session index into that list.

ii. 
```python
subject_id = str(nwb.subject.subject_id)
```

```python
subjects = sorted(set(r['subject_id'] for r in sessions))
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[r['subject_id']] for r in sessions], dtype=np.int64),
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI records that each NWB file contains both the numeric `subject_id` and mouse name in `nwb.subject`, and in Step 5 it explicitly maps `subject.subject_id` to `subjects` / `subject_idx`.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Sessions are ordered by the sorted file list, and each session is identified by `nwb.identifier`.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
```

```python
session_id = nwb.identifier
```

```python
'session_info': [{'session_id': r['session_id'], 'subject_id': r['subject_id'],
                  'mouse_name': r['mouse_name'], 'n_units': r['n_units_used'],
                  'n_trials': r['n_trials_used'], ...} for r in sessions],
```

iii. In Step 2, the AI says `nwb.identifier` matches the paper’s session naming and that the dataset is one file per session, so no further grouping is needed.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI converts it to a DataFrame, uses one row per trial, checks that the number of `go_start_times` matches the number of trials, and then keeps trial rows by boolean mask / `trial_idx`.

ii. 
```python
trials = nwb.intervals['trials']
df = trials.to_dataframe()
n_trials_all = len(df)
...
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
...
if len(go_times) != n_trials_all:
    raise RuntimeError('%s: %d go cues for %d trials' % (session_id, len(go_times), n_trials_all))
```

```python
trial_idx = np.where(keep)[0]
n_trials = len(trial_idx)
...
go = go_times[trial_idx]
tstart = start_time[trial_idx]
tstop = stop_time[trial_idx]
```

iii. In Step 2, the AI notes that `go_start_times` has exactly one event per trial in all 174 sessions, while `sample` and `delay` can repeat within early-lick trials. That is its stated reason for using the trials table as the primary trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if they are not `auto_water`, not `free_water`, have the go cue inside the trial, and are observed by every retained unit according to `units.obs_intervals`. After neural binning it also drops trials with zero spikes across all retained units. Sessions with fewer than 2 remaining trials are dropped.

ii. 
```python
keep = (auto_water == 0) & (free_water == 0)
keep &= (go_times >= start_time) & (go_times <= stop_time)
```

```python
obs_trial = np.ones(n_trials_all, dtype=bool)
obs_index = units['obs_intervals']
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
    m = np.zeros(n_trials_all, dtype=bool)
    if oi.size:
        rows = np.searchsorted(start_time, oi[:, 0] + 1e-6) - 1
        rows = rows[(rows >= 0) & (rows < n_trials_all)]
        m[rows] = True
    obs_trial &= m
...
keep &= obs_trial
```

```python
spikes_in_trial = np.zeros(n_trials, dtype=np.int64)
...
spikes_in_trial += tb[:, 1] - tb[:, 0]
...
has_spikes = spikes_in_trial > 0
...
rates = rates[:, has_spikes, :]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says it deliberately keeps early-lick, ignore, and photostim trials because they are decoder variables, but excludes `auto_water` and `free_water`. In Steps 6 and 10 it says it added the `obs_intervals` and all-zero-spike filters after finding sessions where ephys stopped before behavior ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']`, with trial go-cue times from `BehavioralEvents.go_start_times` used to place the bins. Trial start/stop times are also used to clip the window to observed trial time.

ii. 
```python
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
```

```python
spike_index = units['spike_times']
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
```

```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
```

iii. In Step 5, the AI writes that `units.spike_times` should be converted to firing rates by subtracting / aligning to the go cue and counting spikes in 50 ms bins.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the AI bins spikes into 80 non-overlapping 50 ms bins around the go cue using `np.searchsorted`, differences adjacent cumulative counts to get per-bin spike counts, and divides by 0.05 s to convert to firing rates in spikes/s. It clips bin edges to the trial interval, so bins outside the observed part of the trial become zeros.

ii. 
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]           # (n_trials, NBINS+1)
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
...
flat_edges = edges_clipped.ravel()
rates = np.zeros((n_units, n_trials, NBINS), dtype=np.float32)
...
pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
rates[k] = np.diff(pos, axis=1).astype(np.float32)
rates /= BIN_SIZE
```

iii. In Step 1 and Step 5, the AI says it is following the reference estimator of binned spike counts divided by bin width, but changing the bin width to the task-required 50 ms bins. In Step 5 it also says partially observed bins are zero-filled.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered to `classification == 'good'`. The AI then further requires each good unit to map to one of its coarse hemisphere-resolved brain regions using `anno_name` and `electrodes.x`; sessions with no such units are dropped.

ii. 
```python
classification = np.asarray([str(c) for c in units['classification'][:]])
anno = np.asarray([str(a) for a in units['anno_name'][:]])
good = classification == 'good'
if good.sum() == 0:
    return None
```

```python
for i in np.where(good)[0]:
    reg = coarse_region(anno[i])
    if reg is None or not np.isfinite(ml[i]):
        continue
    side = 'left' if ml[i] >= ML_MIDLINE else 'right'
    region_idx[i] = BRAIN_REGION_INDEX['%s %s' % (side, reg)]
use_unit = good & (region_idx >= 0)
unit_ids = np.where(use_unit)[0]
```

iii. In Steps 2 and 5, the AI justifies `classification == 'good'` as the NWB equivalent of the reference QC classifier output. It also says it wants the same 14 coarse reference region groups split by hemisphere.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. It creates a common grid of relative bin edges from -2.5 s to +1.5 s and adds those offsets to each trial’s `go_start_times` timestamp.

ii. 
```python
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
```

```python
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
...
go = go_times[trial_idx]
edges = go[:, None] + BIN_EDGES_REL[None, :]
```

iii. In Steps 3 and 5, the AI states that everything in NWB shares one session clock, so alignment is done by subtracting / indexing around the go cue rather than by interpolating between clocks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4.0 s window, for 80 time bins total. No secondary temporal rebinning is applied after the direct spike-time binning.

ii. 
```python
BIN_SIZE = 0.05
T_START = -2.5
T_STOP = 1.5
NBINS = int(round((T_STOP - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. In Step 1, the AI explicitly notes that the reference code uses 40 ms / 3.4 ms sliding bins, but the decoder task requires non-overlapping 50 ms bins instead.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents.sample_start_times` and `BehavioralEvents.go_start_times`.

ii. 
```python
sample_times = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
...
go = go_times[trial_idx]
```

iii. In Step 2, the AI writes that early-lick trials can replay the sample epoch, so `sample_start_times` may contain more events than trials and the relevant tone is the last sample onset before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each kept trial, the AI takes the last sample onset before the go cue, falls back to `go - 1.85` s if the tone is missing or predates the kept trial start, and then adds each go-aligned bin center to the trial’s tone-to-go offset.

ii. 
```python
pos = np.searchsorted(sample_times, go, side='left') - 1
tone_onset = np.where(pos >= 0, sample_times[np.clip(pos, 0, len(sample_times) - 1)], np.nan)
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
...
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

iii. In Step 5, the AI says it uses the last sample onset because early-lick trials replay the sample/delay epochs. In Step 9, it notes the `1.85 s` fallback is a defensive default for a standard trial structure; its Step 9 summary says the fallback was not needed on the final full conversion.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the same 80 go-aligned bin centers used for neural data, so each timepoint in the input array matches the corresponding neural bin.

ii. 
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

```python
rates /= BIN_SIZE
...
inputs = np.stack([time_from_tone.astype(np.float32), photostim], axis=1)
```

iii. In Step 5, the AI explicitly defines all variables on the common `-2.5 .. 1.5 s` go-cue grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents.photostim_start_times` and `BehavioralEvents.photostim_stop_times` when those event series are present.

ii. 
```python
if 'photostim_start_times' in be:
    stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
    stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
else:
    stim_on = np.zeros(0)
    stim_off = np.zeros(0)
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the AI says it verified the NWB event series against the trial-table onset/duration fields and chose the event timestamps as the direct source of laser timing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI assigns each stimulation event to a kept trial based on trial start times, then marks a bin as `1` when the bin interval overlaps the stimulation interval and `0` otherwise. Trials without laser events remain all zeros.

ii. 
```python
photostim_b = np.zeros((n_trials, NBINS), dtype=bool)
if len(stim_on):
    which = np.searchsorted(tstart, stim_on, side='right') - 1
    for s_on, s_off, w in zip(stim_on, stim_off, which):
        if w < 0 or w >= n_trials:
            continue
        if s_on < tstart[w] or s_on > tstop[w]:
            continue
        ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
        photostim_b[w] |= ov
photostim = photostim_b.astype(np.float32)
```

iii. In Step 5, the AI says the decoder needs a time-varying photostim input rather than a trial flag, and its notes emphasize that the laser occupies the last part of the delay period and should be represented on the common bin grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation timestamps are compared against the same absolute bin edges (`go + relative_edges`) used to bin spikes, so the binary photostim signal is on the same time axis as neural data.

ii. 
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
...
ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
```

iii. In Step 5, the AI describes photostim as an interval relative to the go cue, then says it is turned into a per-bin on/off series on the same decoding window as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the raw `left_lick_times` and `right_lick_times` behavioral event streams, using the first lick after the go cue within the response window.

ii. 
```python
left_licks = np.asarray(be['left_lick_times'].timestamps[:], dtype=float)
right_licks = np.asarray(be['right_lick_times'].timestamps[:], dtype=float)
```

```python
answer_end = np.minimum(tstop, go + 1.5)
choice = np.full(n_trials, 2, dtype=np.int64)
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
```

iii. In Step 10, the AI says it initially cross-checked against `outcome x instruction`, found a small number of discrepancies, and chose the first-lick definition because it is the more direct measure of behavior and matches the task’s “lick direction choice” more directly.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It codes `0=left`, `1=right`, `2=no lick`. For each trial it finds the earliest post-go left/right lick within the answer period, assigns the side of the earliest lick, or leaves `2` if there is no lick. The per-trial label is repeated across all 80 time bins.

ii. 
```python
choice = np.full(n_trials, 2, dtype=np.int64)
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
    if len(l) == 0 and len(r) == 0:
        continue
    if len(r) == 0:
        choice[i] = 0
    elif len(l) == 0:
        choice[i] = 1
    else:
        choice[i] = 0 if l[0] < r[0] else 1
```

```python
outputs = np.empty((n_trials, 4, NBINS), dtype=np.int64)
outputs[:, 0, :] = choice[:, None]
```

iii. In Step 10, the AI justifies the 1.5 s answer-period restriction from the methods text and says it changed the code after finding that licks just outside the behaviorally counted response period could disagree with the trial outcome labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials table `outcome` column.

ii. 
```python
outcome_str = np.asarray(df['outcome'].values, dtype=object)
...
outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]], dtype=np.int64)
```

iii. In Steps 2 and 5, the AI records that the trials table already stores `'ignore'`, `'miss'`, and `'hit'`, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to `0=ignore`, `1=miss`, `2=hit`, then repeats that per-trial code across all 80 bins.

ii. 
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]], dtype=np.int64)
...
outputs[:, 1, :] = outcome[:, None]
```

iii. In Step 5, the AI says it follows the decoder specification’s three outcome classes and stores per-trial outputs as time-constant rows in the output tensor.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trials table `early_lick` column.

ii. 
```python
early_str = np.asarray(df['early_lick'].values, dtype=object)
...
early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]], dtype=np.int64)
```

iii. In Steps 2 and 5, the AI notes that the trials table already stores this field as `'early'` / `'no early'`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'early'` to `1` and everything else to `0`, then repeats that per-trial value across all bins.

ii. 
```python
early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]], dtype=np.int64)
...
outputs[:, 2, :] = early[:, None]
```

iii. In Step 5, the AI says early lick is a required decoder output and therefore should be preserved as a per-trial categorical variable instead of being used only for filtering.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI uses `BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically its timestamps, y-coordinate column, and likelihood column.

ii. 
```python
ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
vts = np.asarray(ts_obj.timestamps[:], dtype=float)
vdata = np.asarray(ts_obj.data[:], dtype=float)
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. In Step 2, the AI says this series is `(x, y, likelihood)` at 300 Hz and is present for all sessions, making it the natural source of tongue position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI thresholds visibility at likelihood `> 0.9`, computes cumulative sums of visible `tongue_y` and visible-frame counts, uses `searchsorted` at each trial/bin edge to get a visible-frame mean per 50 ms bin, collects all finite binned means in the kept session to compute the 40th and 60th percentiles, and leaves bins with no visible frames as missing.

ii. 
```python
LIKELIHOOD_THRESH = 0.9
...
visible = vdata[:, 2] > LIKELIHOOD_THRESH
ysum = np.concatenate([[0.0], np.cumsum(np.where(visible, tongue_y, 0.0))])
vcnt = np.concatenate([[0], np.cumsum(visible.astype(np.int64))])
vidx = np.searchsorted(vts, edges_clipped)
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
n_y = vcnt[vidx[:, 1:]] - vcnt[vidx[:, :-1]]
with np.errstate(invalid='ignore', divide='ignore'):
    ybin = np.where(n_y > 0, s_y / np.maximum(n_y, 1), np.nan)
vis_vals = ybin[np.isfinite(ybin)]
if vis_vals.size >= 10:
    p40, p60 = np.percentile(vis_vals, [40, 60])
```

iii. In Step 5, the AI justifies the `0.9` threshold by saying the likelihood distribution is strongly bimodal and therefore insensitive to the exact cutoff. It also says it averages visible frames within each 50 ms bin because the decoder requires 50 ms bins rather than the paper’s 300 Hz frame grid.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses `p40` and `p60` percentiles computed from visible binned tongue-y values. It assigns class `0` below `p40`, class `1` from `p40` through `p60`, class `2` above `p60`, and class `3` when no visible frame contributes to the bin.

ii. 
```python
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)
if np.isfinite(p40):
    fin = np.isfinite(ybin)
    tongue_cls[fin & (ybin < p40)] = 0
    tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
    tongue_cls[fin & (ybin > p60)] = 2
```

iii. In Step 5 and Step 7, the AI says it is following the task’s required 40/20/40 split among visible bins and using `3` as the explicit “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue position on the same go-centered 50 ms bins as the neural data by applying `searchsorted` to the camera timestamps at the same per-trial bin edges used for spikes.

ii. 
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
...
vidx = np.searchsorted(vts, edges_clipped)
```

iii. In Step 5, the AI says all time series share the NWB session clock and therefore can be aligned with the same go-relative grid; in Step 10 it also notes that bins beyond trial end are clipped and become class `3`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: sessions with no good units are skipped; trials lacking ephys coverage are dropped by `obs_intervals`; trials with zero spikes across all units are dropped; missing photostim event series produce all-zero photostim inputs; missing or invalid tone onset falls back to `go - 1.85 s`; low-confidence or absent tongue detections become NaN and then tongue class `3`.

ii. 
```python
if good.sum() == 0:
    return None
```

```python
if 'photostim_start_times' in be:
    ...
else:
    stim_on = np.zeros(0)
    stim_off = np.zeros(0)
```

```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
...
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
```

```python
visible = vdata[:, 2] > LIKELIHOOD_THRESH
...
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)
```

iii. In Steps 6, 9, and 10, the AI explains these choices as protections against known data issues in the NWB release: ephys ending before behavior, rare trailing no-spike trials, sessions without good units, and bins where the tongue is not visible.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are per-session NWB I/O, reading the large `spike_times` and video arrays, the per-unit spike binning loop, and final pickle writing. The script parallelizes across sessions with a multiprocessing pool.

ii. 
```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
```

```python
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32)
```

```python
from multiprocessing import Pool
with Pool(min(args.nproc, len(files))) as pool:
    for i, r in enumerate(pool.imap(_worker, jobs)):
```

iii. In Step 6 and Step 7, the AI explicitly says session loading, spike-time searchsorted, and the tongue array are the main costs, and that it parallelized over sessions to stay within runtime limits.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python loops that could be vectorized further: the loop over units to intersect `obs_intervals`, the per-unit spike binning loop, the per-stimulation-event loop that paints photostim bins, and the per-trial loop that computes first-lick choice. Tongue binning was already partly vectorized with cumulative sums.

ii. 
```python
for ui in unit_ids:
    oi = np.asarray(obs_index[int(ui)], dtype=float)
    ...
    obs_trial &= m
```

```python
for k, ui in enumerate(unit_ids):
    st = np.asarray(spike_index[int(ui)], dtype=float)
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
```

```python
for s_on, s_off, w in zip(stim_on, stim_off, which):
    ...
    photostim_b[w] |= ov
```

```python
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
```

iii. The AI’s Step 6 notes say the unit loop is “unavoidable” because spike storage is ragged, but the code still leaves a few other loops in Python for clarity.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some indexing work: each unit’s spike times are searched twice, once against all bin edges and again against trial bounds for the all-zero-trial check; the lick event arrays are rescanned separately for every trial when computing choice; and summary code later restacks / concatenates per-session outputs and inputs for reporting statistics.

ii. 
```python
pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
rates[k] = np.diff(pos, axis=1).astype(np.float32)
tb = np.searchsorted(st, trial_bounds).reshape(n_trials, 2)
spikes_in_trial += tb[:, 1] - tb[:, 0]
```

```python
for i in range(n_trials):
    l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
    r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
```

```python
outs = np.concatenate([np.stack(r['output'])[:, :, 0] for r in sessions])
...
tongue = np.concatenate([np.stack(r['output'])[:, 3, :].ravel() for r in sessions])
...
stim = np.concatenate([np.stack(r['input'])[:, 1, :] for r in sessions])
```

iii. The AI’s notes emphasize a single-pass conversion, but the final script still repeats some searches and reductions for validation and summary output.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra QA / provenance work that is not needed by downstream decoder training: optional plotting, timing breakdowns, per-session diagnostics such as `frac_fully_observed`, `n_bad_tone`, `n_trials_unobserved`, `n_trials_nospike`, and summary-statistic recomputation for console output. It also computes `observed`, `spikes_in_trial`, and several metadata-only fields that do not affect the final neural/input/output tensors.

ii. 
```python
timing = {}
...
timing['setup'] = time.time() - t0
...
timing['neural'] = time.time() - t1
...
timing['input'] = time.time() - t1
...
timing['output'] = time.time() - t1
```

```python
if make_plots:
    plot_session(result, dict(...))
```

```python
'frac_fully_observed': float(np.mean(observed.all(axis=1))),
'frac_bins_observed': float(np.mean(observed)),
'n_bad_tone': n_bad_tone,
'n_trials_unobserved': n_trials_unobserved,
'n_trials_nospike': n_trials_nospike,
'timing': timing,
'total_time': time.time() - t0,
```

iii. These extra computations are justified in `CONVERSION_NOTES.md` as validation and debugging aids, but they are not part of the downstream decoder’s required data representation.
