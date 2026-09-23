# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `/app/data/sub-*/*.nwb`, then processes each file as one session with `h5py`. Within each file it reads the subject id, trials table, behavioral events, units table, electrodes table, and behavioral time series.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
with h5py.File(filepath, 'r') as f:
    subject = _decode(f['general/subject/subject_id'][()])
    identifier = _decode(f['identifier'][()])
    tr = f['intervals/trials']
    be = f['acquisition/BehavioralEvents']
    u = f['units']
```

iii. In `CONVERSION_NOTES.md`, the AI says the released data are in a DANDI-style NWB layout with 174 files and that each NWB file already contains the behavioral, spike, and tracking streams needed for conversion, so one pass over the sorted file list is sufficient.

## 1-b. How are the data split into subjects?

i. The AI uses `general/subject/subject_id` from each NWB file as the mouse identifier, then builds `subjects` as the sorted unique ids and `subject_idx` as a per-session index into that list.

ii.
```python
subject = _decode(f['general/subject/subject_id'][()])
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[s['subject']] for s in sessions]),
```

iii. The notes say the NWB layout is `sub-<animalid>/...nwb`, and that `subject_id` is the canonical per-animal field, so no additional grouping heuristic is needed.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order follows the sorted file list, and the session id comes from the file’s `identifier` field.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
identifier = _decode(f['identifier'][()])
...
'session_id': identifier,
```

iii. The notes explicitly say the DANDI release is one file per behavioral session, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table (`intervals/trials`) as the master trial list. Go-cue timestamps come from `BehavioralEvents/go_start_times`; if their count already matches the trial count, they are used directly, otherwise go events are assigned back to trials by `searchsorted`, and trials without a finite go cue are later dropped.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
...
ntrials_all = len(start_time)
```

```python
go_all = be['go_start_times']['timestamps'][:]
if len(go_all) != ntrials_all:
    gi = np.searchsorted(start_time, go_all, side='right') - 1
    go = np.full(ntrials_all, np.nan)
    go[gi] = go_all
else:
    go = go_all
```

iii. The notes say there is exactly one `go_start_times` event per trial in the released files, but the code still contains a defensive fallback to map go cues to trials if counts ever differ.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are covered by ephys recording, have a finite go cue, and are not `auto_water` or `free_water`. It infers ephys-covered trials from `units/obs_intervals` by intersecting observation masks across all QC-passing units. After spike binning it also drops trials whose entire neural matrix sums to zero, treating them as recording-truncation artifacts. Sessions with fewer than 2 remaining trials are dropped.

ii.
```python
good_pre = (classification_pre == 'good') & (anno_pre != '')
...
observed = np.ones(ntrials_all, dtype=bool)
for ui in np.where(good_pre)[0]:
    iv = u_pre['obs_intervals'][oi_starts[ui]:oii[ui]]
    ...
    observed &= m
```

```python
keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed
trial_idx = np.where(keep)[0]
if len(trial_idx) < 2:
    return None
```

```python
nonempty = fr.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    fr = fr[:, nonempty, :]
    trial_idx = trial_idx[nonempty]
    go_keep = go_keep[nonempty]
if len(trial_idx) < 2:
    return None
```

iii. The notes justify this as combining the reference `get_regular_trial_mask` with NWB-specific cleanup: auto/free-water trials are removed as non-regular task trials, trials outside `obs_intervals` are dropped because the probes were not recording, and all-zero trials are treated as mid-recording-stop artifacts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` and `units/spike_times_index`, using go-cue times from `BehavioralEvents/go_start_times` to define trial-aligned bin edges. Only QC-passing units are used.

ii.
```python
go_keep = go[trial_idx]
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)
```

iii. The notes describe `units/spike_times` as the raw neural representation in the NWB release and say the goal is to preserve the reference pipeline’s firing-rate convention while changing only the binning spec required by the decoder task.

## 2-b. How is the `neural` data processed?

i. The AI converts spikes to firing rates by binning spike counts into 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue, then dividing counts by 0.05 s to get spikes/s. It bins all trials at once per unit by flattening the trial-by-edge array and using `np.searchsorted`, followed by `np.diff`.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
...
for k, u in enumerate(unit_idx):
    st = spike_times[starts[u]:spike_index[u]]
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    out[k] = np.diff(pos, axis=1)
out /= BIN_SIZE
```

iii. The notes say this matches the reference convention `rate = spike_count / bin_width`, but swaps in non-overlapping 50 ms bins because that is what the decoder instructions require.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units/classification == 'good'` and a non-empty `units/anno_name`. If a session has no such units, the whole session is dropped.

ii.
```python
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
if len(unit_idx) == 0:
    return None
```

iii. In the notes, the AI argues that `classification == 'good'` is the released output of the Chen/Liu QC classifier and that the reference code also requires histology / CCF annotation, represented here by non-empty `anno_name`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to go-cue onset. For each trial the AI adds the fixed relative bin edges `[-2.5, 1.5]` to that trial’s go time and bins spikes against those absolute edges.

ii.
```python
go_keep = go[trial_idx]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The notes repeatedly state that the reference preprocessing is go-cue aligned and that all NWB timestamps share the session clock, so alignment only requires using `go_start_times` as t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, with 80 bins per trial over the 4 s window from -2.5 to +1.5 s. There is no secondary temporal rebinning after the initial spike binning.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```

iii. The AI notes that the reference papers used 40 ms sliding windows with 3.4 ms stride, but the conversion deliberately changes that to the task-specified 50 ms non-overlapping bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `BehavioralEvents/sample_start_times` and the per-trial go cues. For each trial the AI chooses the last sample-epoch start before the go cue as the relevant tone onset.

ii.
```python
sample_all = be['sample_start_times']['timestamps'][:]
...
j = np.searchsorted(sample_all, go, side='right') - 1
tone = np.where(j >= 0, sample_all[np.clip(j, 0, len(sample_all) - 1)], np.nan)
```

iii. The notes justify using the last sample onset because early licks can replay the sample epoch, so the last pre-go tone is the one behaviorally tied to that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes this as the bin center time relative to go cue plus the per-trial offset between go cue and tone onset. It also falls back to `go - 1.85 s` when the inferred tone time is missing or lies outside the trial.

ii.
```python
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)
```

```python
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
```

iii. The notes say the nominal tone-to-go interval is 1.85 s (0.65 s sample + 1.2 s delay), and present the fallback as defensive handling for malformed or missing sample events.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the exact same 80-bin go-cue-relative grid used for the neural data, using the same `BIN_CENTERS`.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
...
time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]
```

iii. The notes say this input is intentionally constructed on the neural bin centers so every trial has a time-varying input array aligned one-to-one with the firing-rate matrix.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trials-table fields `photostim_onset`, `photostim_duration`, and `photostim_power`, together with `start_time` and the per-trial go cue. `photostim_power` is used to decide whether a trial is stimulated at all.

ii.
```python
ps_onset = np.array([_tofloat(x) for x in tr['photostim_onset'][:]])
ps_dur = np.array([_tofloat(x) for x in tr['photostim_duration'][:]])
ps_power = np.array([_tofloat(x, 0.0) for x in tr['photostim_power'][:]])
```

```python
has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)
...
s0 = start_time[i] + ps_onset[i] - go[i]
s1 = s0 + ps_dur[i]
```

iii. The notes describe this as reconstructing trial-relative photostim timing from the trials table and then expressing it relative to the go cue, which is the common alignment axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI builds a binary time series with one value per 50 ms bin. A bin is marked 1 if the laser interval overlaps that bin and 0 otherwise.

ii.
```python
photostim = np.zeros((len(trial_idx), N_BINS), dtype=np.float32)
...
overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
photostim[k, overlap] = 1.0
```

iii. The notes say the decoder needs a time-varying photostim input rather than a per-trial flag, and describe the intended representation as “1 while ALM photoinhibition laser is on during the bin.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are converted to times relative to each trial’s go cue, then compared against the same fixed go-cue-relative bins as the neural data.

ii.
```python
s0 = start_time[i] + ps_onset[i] - go[i]
s1 = s0 + ps_dur[i]
overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)
```

iii. The notes justify this by stating that all streams are expressed on the go-cue axis in the converted dataset, so photostim must also be converted onto that axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from one NWB field. The AI derives it from `intervals/trials/outcome` and `intervals/trials/trial_instruction`.

ii.
```python
oc = outcome[trial_idx]
ins = instruction[trial_idx]
...
licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')
choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
```

iii. The notes say this mirrors the task logic: hit means the instructed lick occurred, miss means the opposite side, and ignore means no lick choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0 = left`, `1 = right`, `2 = no lick`, then repeats that per-trial value across all 80 bins in the output tensor.

ii.
```python
OUTPUT_VALUES = [['left', 'right', 'no lick'],
                 ['ignore', 'miss', 'hit'],
                 ['no', 'yes'],
                 ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]
```

```python
choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1)).astype(np.int64)
...
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome_code[i]),
                     np.full(N_BINS, early_code[i]), tongue[i]]).astype(np.int64)
           for i in range(ntr)]
```

iii. The notes say the decoder prefers time-varying outputs when possible, so trial-level categorical outputs are broadcast across bins to share a common `(n_output, n_timepoints)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii.
```python
outcome = np.array([_decode(x) for x in tr['outcome'][:]])
...
oc = outcome[trial_idx]
```

iii. The notes say the NWB trials table already uses the needed three outcome labels: `hit`, `miss`, and `ignore`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome to integer codes `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats that code across all 80 bins.

ii.
```python
outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2)).astype(np.int64)
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome_code[i]),
                     np.full(N_BINS, early_code[i]), tongue[i]]).astype(np.int64)
           for i in range(ntr)]
```

iii. The notes justify this as a direct categorical recoding of the raw trial labels into the decoder’s output format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trials-table `early_lick` column.

ii.
```python
early = np.array([_decode(x) for x in tr['early_lick'][:]])
...
el_tr = early[trial_idx]
```

iii. The notes say this matches the reference behavior field for whether the mouse licked during sample or delay.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to 0 and every other decoded early-lick string to 1, then repeats that trial-level code across all 80 bins.

ii.
```python
early_code = np.array([0 if e == 'no early' else 1 for e in el_tr], dtype=np.int64)
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome_code[i]),
                     np.full(N_BINS, early_code[i]), tongue[i]]).astype(np.int64)
           for i in range(ntr)]
```

iii. The notes describe this as preserving early-lick trials because early lick is itself a required decoder output, so the field is recoded rather than used as an exclusion criterion.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps plus the tracking `data` array, where column 1 is y-position and column 2 is DeepLabCut likelihood.

ii.
```python
tt_ds = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vdata = tt_ds['data'][:]
vts = tt_ds['timestamps'][:]
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
yv = vdata[:, 1]
```

iii. The notes identify side-view tongue tracking as the reference behavioral stream and explicitly describe the three columns as x, y, and likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first thresholds frame visibility at likelihood `> 0.9`. It then computes the 40th and 60th percentile thresholds from all visible-frame y-values in the session. For each trial and each bin, it uses the last visible frame in that bin; bins with no visible frame keep class 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
```

```python
vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
yv = vdata[:, 1]
if np.any(vis):
    thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
else:
    thr_lo, thr_hi = np.nan, np.nan
```

```python
ybin = np.full(N_BINS, np.nan)
ybin[idx] = yy
good = ~np.isnan(ybin)
c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
cls[i, good] = c
```

iii. The notes justify this by appealing to the reference marker-alignment helper, which keeps the last frame inside each bin, and by arguing that the visibility distribution is strongly bimodal so a high likelihood threshold is acceptable.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses per-session 40th and 60th percentiles of visible-frame y-values as thresholds. For each bin, visible bins are assigned class 0 if below the 40th percentile, class 1 if between the thresholds, class 2 if above the 60th percentile, and bins with no visible frame are assigned class 3.

ii.
```python
if np.any(vis):
    thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])
else:
    thr_lo, thr_hi = np.nan, np.nan
```

```python
c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))
cls[i, good] = c
```

iii. The notes say the 40/60 split comes from the decoder instructions and that the extra class 3 is needed because the tongue is often not visible before the response epoch.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue tracking to the same go-cue-relative trial window as neural activity. It finds each trial’s camera-frame slice using `go + OFF_START` and `go + OFF_END`, then bins frames by their offset within that trial window.

ii.
```python
t0 = go_times + OFF_START
t1 = go_times + OFF_END
lo = np.searchsorted(ts, t0, side='left')
hi = np.searchsorted(ts, t1, side='left')
for i in range(ntrials):
    ...
    idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)
```

iii. The notes say the camera timestamps share the NWB session clock with spikes and events, so explicit resynchronization is unnecessary; the same go-aligned bin grid can be used directly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: non-string unit annotations/classifications decode to empty strings and therefore fail QC; sessions with no QC-passing units are dropped; trials missing go cues are dropped; trials outside the ephys observation interval are dropped; all-zero neural trials after binning are dropped; malformed or missing tone times fall back to `go - 1.85 s`; and bins without visible tongue frames remain in a dedicated `not visible` class.

ii.
```python
def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, str):
        return x
    return ''
```

```python
bad_tone = ~np.isfinite(tone) | (tone < start_time) | (tone > go)
tone = np.where(bad_tone, go - 1.85, tone)
```

```python
keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed
...
nonempty = fr.sum(axis=(0, 2)) > 0
```

iii. `CONVERSION_NOTES.md` frames these as practical safeguards against NWB-release quirks: partially recorded sessions, truncated final trials, NaN text fields, and occasional missing event metadata.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies loading large per-session arrays and spike binning as the dominant costs, with tongue discretization and final pickling as secondary costs. The notes specifically call out reading `spike_times`, `bin_spikes`, tongue processing, and writing the large pickle.

ii.
```python
t0 = time.time()
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
timings['read_spikes'] = time.time() - t0
```

```python
t0 = time.time()
fr = bin_spikes(spike_times, spike_index, unit_idx, go_keep)
timings['bin_spikes'] = time.time() - t0
```

```python
t0 = time.time()
...
tongue = tongue_class_per_bin(vts, yv, vis, go_keep, thr_lo, thr_hi)
timings['tongue'] = time.time() - t0
```

iii. The notes contain explicit runtime tables showing `read_spikes`, `bin_spikes`, tongue discretization, multiprocessing, and final pickling as the main runtime contributors.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loops are the per-unit loop in `bin_spikes`, the per-trial loop in `tongue_class_per_bin`, and the per-trial loop used to rasterize photostim into bins.

ii.
```python
for k, u in enumerate(unit_idx):
    st = spike_times[starts[u]:spike_index[u]]
    ...
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
```

```python
for i in range(ntrials):
    ...
    ybin[idx] = yy
```

```python
for k, i in enumerate(trial_idx):
    if not has_stim[i]:
        continue
    ...
    photostim[k, overlap] = 1.0
```

iii. The notes say the AI already vectorized across trials for spike binning, but left ragged per-unit spike searches and per-trial tongue processing in loop form for simplicity.

## 10-c. What processing does the code repeat multiple times?

i. The AI code repeats some session preprocessing. It decodes `classification` and `anno_name` twice, once in a prepass to compute the observation-interval mask and again in the main unit-selection block. It also scans `obs_intervals` across good units before later rereading the full spike buffers for those same units.

ii.
```python
classification_pre = np.array([_decode(x) for x in u_pre['classification'][:]])
anno_pre = np.array([_decode(x) for x in u_pre['anno_name'][:]])
good_pre = (classification_pre == 'good') & (anno_pre != '')
```

```python
classification = np.array([_decode(x) for x in u['classification'][:]])
anno = np.array([_decode(x) for x in u['anno_name'][:]])
good = (classification == 'good') & (anno != '')
unit_idx = np.where(good)[0]
```

iii. The notes explain this as an NWB-specific prepass to remove trials outside ephys coverage, but the implementation does re-read and re-decode the same unit-level metadata rather than caching it once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs some extra work that is not needed for the core decoder arrays: it reads `stop_time` without using it, reads electrode `y` coordinates without using them, computes and stores per-unit hemisphere labels only in metadata, and maintains per-session timing diagnostics used for logging rather than the converted dataset’s main neural/input/output tensors.

ii.
```python
stop_time = tr['stop_time'][:]
```

```python
ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
ux, uy, uz = ex[eidx], ey[eidx], ez[eidx]
hemi = np.where(ux >= CCF_ML_MIDLINE, 'left', 'right')
```

```python
'hemisphere': hemi[unit_idx],
'timings': timings,
```

iii. The notes present these extras as diagnostics and metadata for sanity checks and review. They are not part of the essential conversion needed to train the downstream decoder.
