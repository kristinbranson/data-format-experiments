# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent treats the dataset as one NWB file per session under `/app/data/*/*.nwb`. It loads the full file list with `glob`, then processes sessions in parallel with `multiprocessing.Pool`. Inside each file it reads the trial table, behavioral events, units table, subject id, and tongue-tracking data directly with `h5py`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

```python
def process_session(args):
    fname, name2anc = args
    ...
    with h5py.File(fname, 'r') as f:
        out = _process_session(f, fname, name2anc, info)
```

```python
tr = f['intervals/trials']
be = f['acquisition/BehavioralEvents']
```

iii. In the trajectory, the agent explicitly concluded that there are 174 NWB session files organized by subject, that one NWB file corresponds to one session, and that full-file processing with parallel workers was feasible on the available hardware.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `general/subject/subject_id` in each NWB file. After session processing, the agent constructs `subjects` as the sorted unique ids and `subject_idx` as the per-session index into that list.

ii. 
```python
subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
info['subject'] = subject
```

```python
subjects = sorted({r['subject'] for r in results})
...
'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
```

iii. In the trajectory, the agent treated the NWB `subject_id` field as the canonical animal identifier and relied on the NWB/session layout rather than inventing another grouping key.

## 1-c. How are the data split into sessions?

i. The agent uses one NWB file as one session. Session boundaries are therefore the file boundaries; after curation, each surviving processed file becomes one output session.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
```

```python
results.append(res)
...
'neural': [r['neural'] for r in results],
'input': [r['input'] for r in results],
'output': [r['output'] for r in results],
```

iii. In the trajectory, the agent repeatedly stated that the release is organized as one NWB per recording session and used that as the basis for session handling.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. Go-cue timestamps are mapped into those trial intervals with `map_events_to_trials`, and the resulting per-trial `go` vector is used as the trial alignment anchor.

ii. 
```python
trial_start = tr['start_time'][:]
trial_stop = tr['stop_time'][:]
```

```python
go_all = be['go_start_times/timestamps'][:]
gidx = map_events_to_trials(go_all, trial_start, trial_stop)
go = np.full(ntrials_all, np.nan)
go[gidx[gidx >= 0]] = go_all[gidx >= 0]
```

iii. In the trajectory, the agent noted that each trial has exactly one go cue and that some other event streams can repeat within a trial, so it chose the trial table plus go-cue mapping rather than re-deriving trial structure from event sequences alone.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a much broader trial filter than the reference. It first keeps only non-`auto_water`, non-`free_water` trials with finite go cues and adequate video coverage. It then intersects that with trials observed by all selected units via `units/obs_intervals`. After building trial arrays, it drops trials with missing tone onset or all-zero neural data.

ii. 
```python
keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)
```

```python
observed = np.ones(ntrials_all, dtype=bool)
for u in unit_idx:
    ...
    observed &= m
...
keep = keep & observed
```

```python
valid = [i for i, x in enumerate(input_trials)
         if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

iii. In the trajectory, the agent justified these filters by saying free/auto-water trials make choice/outcome ambiguous, low-coverage video undermines the tongue output, `obs_intervals` identify when ephys was actually running, and zero-spike trials reflect acquisition absence rather than silence.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, together with each trial’s go-cue time from `BehavioralEvents/go_start_times/timestamps`. The selected unit set comes from `units/classification` and later `units/is_good_trials`.

ii. 
```python
classification = _str(f['units/classification'][:])
good_unit = classification == 'good'
...
sp_index = f['units/spike_times_index'][:]
sp_start = np.concatenate([[0], sp_index[:-1]])
spike_ds = f['units/spike_times']
```

```python
go_all = be['go_start_times/timestamps'][:]
...
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. In the trajectory, the agent identified `spike_times` as the only raw neural representation available in the NWB files and used go-cue timestamps as the required alignment reference.

## 2-b. How is the `neural` data processed?

i. The agent converts spike times to firing rates in non-overlapping 50 ms bins. For each kept unit it uses `np.searchsorted` against flattened trial-specific bin edges, differences adjacent cumulative counts, and divides by `BIN_SIZE` to get spikes/s. It does not smooth, baseline-subtract, or normalize firing rates.

ii. 
```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
flat_edges = edges.ravel()
order = np.argsort(flat_edges, kind='stable')
sorted_edges = flat_edges[order]
```

```python
for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
    ...
    counts_sorted = np.searchsorted(st, sorted_edges)
    counts = np.empty_like(counts_sorted)
    counts[order] = counts_sorted
    counts = counts.reshape(len(trials), NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE
```

iii. In the trajectory, the agent said the only intentional change from the paper preprocessing was the task-mandated 50 ms bin width, and that spike binning was verified against brute-force histograms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only units with `units/classification == 'good'`. It then further drops units whose `units/is_good_trials` flag is false on any kept trial, and drops entire sessions if no units survive.

ii. 
```python
classification = _str(f['units/classification'][:])
good_unit = classification == 'good'
if good_unit.sum() == 0:
    info['excluded'] = 'no good units'
    return None
```

```python
ig_all = f['units/is_good_trials'][:]
...
unit_ok[i] = bool(flag[keep].all())
...
unit_idx = unit_idx[unit_ok]
if len(unit_idx) == 0:
    info['excluded'] = 'no good units after per-trial QC'
    return None
```

iii. In the trajectory, the agent justified the extra `is_good_trials` screen as removing units that drift or become unstable on analyzed trials, so that every retained neuron is “well isolated on every kept trial.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. The agent constructs trial-specific bin edges by adding the fixed `[-2.5, 1.5]` second window to each trial’s go time, and bins spikes directly on that session-wide clock.

ii. 
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
```

```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. In the trajectory, the agent stated that spikes, trials, events, and video are already on the same NWB session clock, so no additional synchronization offset is needed beyond centering on the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping bins of width 50 ms spanning -2.5 s to +1.5 s relative to the go cue. No additional temporal smoothing or rebinning is applied beyond direct 50 ms binning.

ii. 
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE
```

iii. In the trajectory, the agent explicitly contrasted this with the paper’s 40 ms / 3.4 ms preprocessing and said it changed only the binning to satisfy the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times/timestamps` and each trial’s go-cue time. The agent maps sample events into trials and stores, for each trial, the last sample/tone onset that occurs before the go cue.

ii. 
```python
sam_all = be['sample_start_times/timestamps'][:]
sidx = map_events_to_trials(sam_all, trial_start, trial_stop)
tone_onset = np.full(ntrials_all, np.nan)
for t, i in zip(sam_all, sidx):
    ...
```

iii. In the trajectory, the agent noted that early-lick trials replay the sample epoch, so a trial can contain multiple sample starts; it therefore chose the last one before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the per-trial tone onset, the agent computes each bin’s absolute center time relative to the go cue and subtracts the tone time. If no tone onset is found, the trial gets `NaN` for this input and is later dropped.

ii. 
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
```

```python
if np.isfinite(tone_onset[t]):
    time_from_tone = (g + bin_centers) - tone_onset[t]
else:
    time_from_tone = np.full(NBINS, np.nan)
```

iii. In the trajectory, the agent described this as the go-centered bin grid shifted into “time since tone onset,” with no other transformation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is placed on exactly the same 80-bin go-centered grid as the neural data, using the same `bin_centers` used to interpret neural bins.

ii. 
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
...
time_from_tone = (g + bin_centers) - tone_onset[t]
```

```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. In the trajectory, the agent’s rationale was that all streams share the same session clock, so using the same go-centered bin grid guarantees alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation decoder input is derived from `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, mapped into trials. Separately, the trials-table `photostim_onset` field is used only for session-level “control trial” selection.

ii. 
```python
photostim_onset = _str(tr['photostim_onset'][:])
...
control = (photostim_onset == 'N/A') & (early == 'no early')
```

```python
ps_on = np.full(ntrials_all, np.nan)
ps_off = np.full(ntrials_all, np.nan)
if 'photostim_start_times' in be:
    pst = be['photostim_start_times/timestamps'][:]
    psp = be['photostim_stop_times/timestamps'][:]
    pidx = map_events_to_trials(pst, trial_start, trial_stop)
```

iii. In the trajectory, the agent said the absolute photostim start/stop events were easier to align to the go-centered grid than parsing the string-valued trial-table onset fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each kept trial, the agent expresses photostim onset/offset relative to the trial’s go cue and marks a bin as 1 when the bin interval overlaps the stimulation interval; otherwise it is 0. Non-stimulated trials remain all zeros.

ii. 
```python
photostim = np.zeros(NBINS, dtype=np.float32)
if np.isfinite(ps_on[t]):
    a, b = ps_on[t] - g, ps_off[t] - g
    photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. In the trajectory, the agent justified this as a time-varying on/off signal aligned to the neural bins rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The agent subtracts the trial’s go time from the absolute photostim start/stop times, then compares those relative times to the same bin boundaries used for neural alignment.

ii. 
```python
a, b = ps_on[t] - g, ps_off[t] - g
photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. In the trajectory, the agent’s stated logic was that all timestamps live on the same session clock, so putting photostim on the go-relative axis is sufficient for alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated field. It is derived from the trial-table fields `trial_instruction` and `outcome`.

ii. 
```python
instruction = _str(tr['trial_instruction'][:])
outcome = _str(tr['outcome'][:])
```

```python
if outcome[t] == 'hit':
    ch = instruction[t]
elif outcome[t] == 'miss':
    ch = other[instruction[t]]
else:
    ch = 'no lick'
```

iii. In the trajectory, the agent said this was cross-checked against recorded lick times and gave >99% agreement, so it treated instruction plus outcome as a reliable derived choice label.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps hit trials to the instructed side, miss trials to the opposite side, and ignore trials to a third `no lick` class. It encodes these as `0/1/2` and repeats the per-trial label across all 80 bins.

ii. 
```python
other = {'left': 'right', 'right': 'left'}
choice_map = {'left': 0, 'right': 1, 'no lick': 2}
```

```python
out = np.stack([
    np.full(NBINS, choice_map[ch], dtype=np.int64),
    ...
])
```

iii. In the trajectory, the agent justified the repeated-per-bin representation as a way to keep all outputs in a uniform `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table field `outcome`.

ii. 
```python
outcome = _str(tr['outcome'][:])
```

iii. In the trajectory, the agent treated the trial-table outcome field as already providing exactly the decoder’s required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped to `0=ignore`, `1=miss`, `2=hit`, then repeated across all 80 bins for each trial.

ii. 
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64)
```

iii. In the trajectory, the agent described outcome as a per-trial categorical output that was simply broadcast across time.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trial-table field `early_lick`.

ii. 
```python
early = _str(tr['early_lick'][:])
```

iii. In the trajectory, the agent noted that early-lick trials had to be kept because the decoder task explicitly asks for early lick as an output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `early` to 1 and `no early` to 0, then repeats that per-trial value across all 80 bins.

ii. 
```python
np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64)
```

iii. In the trajectory, the agent treated this exactly like the other per-trial categorical outputs and repeated it along time.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The agent uses the timestamp vector, `data[:, 1]` as `tongue_y`, and `data[:, 2]` as the visibility/likelihood signal.

ii. 
```python
key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
vts = f[key + '/timestamps'][:]
vdata = f[key + '/data'][:]
tongue_y = vdata[:, 1].astype(np.float64)
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. In the trajectory, the agent identified the side-camera DeepLabCut series as the relevant tongue source and verified that its three columns are x, y, and likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent applies several processing steps that differ from the reference: it thresholds visibility at likelihood `> 0.9`, cleans visible frames with a five-sigma velocity outlier rule plus interpolation, computes session-wide 40th/60th percentiles from all visible-frame `tongue_y` values, rejects sessions with too little usable video, rejects trials with <95% video coverage, then averages visible `tongue_y` values within each 50 ms bin.

ii. 
```python
LIKELIHOOD_THRESH = 0.9
VELOCITY_SIGMA = 5.0
MIN_TRIAL_VIDEO_COVERAGE = 0.95
MIN_SESSION_GOOD_VIDEO = 0.5
```

```python
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
...
p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
```

```python
nframes_win = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
coverage = nframes_win * np.median(np.diff(vts)) / (OFF_END - OFF_START)
good_video = coverage >= MIN_TRIAL_VIDEO_COVERAGE
```

iii. In the trajectory, the agent justified this by saying the tongue output required reliable video coverage, that the likelihood distribution was effectively bimodal, and that Wang/Kurgyis marker-cleaning logic should be reused.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent uses session-specific thresholds `p40` and `p60` computed from visible-frame `tongue_y` values across the session. For each trial/bin it computes the mean visible `tongue_y`; bins below `p40` are class 0, bins up to `p60` are class 1, bins above `p60` are class 2, and bins with no visible frames remain class 3.

ii. 
```python
p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
```

```python
tclass = np.full(NBINS, 3, dtype=np.int64)
...
tclass[seen] = np.where(ymean[seen] < p40, 0,
                        np.where(ymean[seen] <= p60, 1, 2))
```

iii. In the trajectory, the agent said it wanted per-session discretization with an explicit “not visible” class and preferred thresholds taken from frames deemed reliably visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The agent aligns tongue output to the same go-centered window as neural activity. For each trial it finds the camera frames between `go + OFF_START` and `go + OFF_END`, converts frame times to 50 ms bin indices relative to `go + OFF_START`, then fills the corresponding trial bins.

ii. 
```python
lo = np.searchsorted(vts, g + OFF_START)
hi = np.searchsorted(vts, g + OFF_END)
...
b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
np.clip(b_idx, 0, NBINS - 1, out=b_idx)
```

iii. In the trajectory, the agent stated that camera timestamps share the same NWB session clock as spikes and events, so direct search and binning on the go-centered grid was sufficient.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent generally handles missing or suspect data by exclusion rather than by representing it in the output. It drops sessions with poor behavior, missing/non-monotonic/insufficient tongue-tracking video, sessions with no surviving good units, trials outside `obs_intervals`, trials with missing tone onset, and trials with all-zero neural activity. For tongue tracking it also marks bins with no visible frames as class 3, and it has a fallback that skips `is_good_trials` filtering if that matrix shape is unexpected.

ii. 
```python
if not (perf > PERF_THRESH and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    info['excluded'] = 'behavior'
    return None
```

```python
if np.any(np.diff(vts) <= 0):
    info['excluded'] = 'non-monotonic video timestamps'
    return None
...
if good_video.mean() < MIN_SESSION_GOOD_VIDEO:
    info['excluded'] = 'video does not cover the analysis window'
    return None
```

```python
valid = [i for i, x in enumerate(input_trials)
         if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

iii. In the trajectory, the agent repeatedly framed these exclusions as preventing “absence of data” from being encoded as meaningful zeros or categories.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are opening each NWB file, loading large spike-time and tongue-tracking arrays, and the per-unit spike-binning loop. The code is structured to parallelize over sessions, indicating that per-session I/O and per-unit processing are the expensive parts.

ii. 
```python
with h5py.File(fname, 'r') as f:
    out = _process_session(f, fname, name2anc, info)
```

```python
vdata = f[key + '/data'][:]
...
spike_ds = f['units/spike_times']
for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
```

```python
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

iii. In the trajectory, the agent explicitly treated full-session NWB reads, tongue/video handling, and spike binning as the main runtime costs and used multiprocessing to reduce wall-clock time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain: looping over units to intersect `obs_intervals`, looping again over units for `is_good_trials`, looping over units to bin spikes, looping over sample/photostim events to assign them to trials, and looping over trials to build per-trial inputs/outputs and tongue classes.

ii. 
```python
for u in unit_idx:
    iv = obs[obs_start[u]:obs_idx[u], 0]
    ...
    observed &= m
```

```python
for i, u in enumerate(unit_idx):
    ...
    unit_ok[i] = bool(flag[keep].all())
```

```python
for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
    ...
```

```python
for k, t in enumerate(trials):
    ...
```

iii. In the trajectory, the agent accepted these loops as a pragmatic tradeoff, especially for ragged per-unit spike and QC data, but they are the obvious places where more vectorization could have been attempted.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several related computations: it scans `obs_intervals` once to decide observed trials and again indirectly when mapping `is_good_trials`; it computes video-window search operations for session-level coverage and then again per trial for tongue binning; and it reconstructs repeated per-trial constant output arrays for choice/outcome/early lick inside the trial loop.

ii. 
```python
nframes_win = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
...
for k, t in enumerate(trials):
    lo = np.searchsorted(vts, g + OFF_START)
    hi = np.searchsorted(vts, g + OFF_END)
```

```python
for u in unit_idx:
    ...
for i, u in enumerate(unit_idx):
    ...
```

```python
out = np.stack([
    np.full(NBINS, choice_map[ch], dtype=np.int64),
    np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
    np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
    tclass,
])
```

iii. In the trajectory, the agent focused on matching its chosen curation rules rather than minimizing recomputation, so it accepted some repeated trial/unit passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Some work is not used by the downstream decoder. `frame_dt` is computed but never used. The script also builds detailed per-session bookkeeping fields such as behavioral performance and coverage statistics that are stored in metadata but not consumed by decoding. More broadly, the extra session/video/unit curation logic serves the agent’s chosen pipeline, not a format requirement.

ii. 
```python
frame_dt = np.median(np.diff(vts))
```

```python
info['performance'] = perf
info['n_correct_left'] = n_correct_left
info['n_correct_right'] = n_correct_right
...
info['frac_trials_good_video'] = float(good_video.mean())
...
info['n_unobserved_trials'] = int((~observed[trials]).sum())
```

```python
'metadata': {
    ...
    'session_info': [r['info'] for r in results],
    ...
}
```

iii. In the trajectory, the agent treated these statistics as documentation and sanity checks for its own conversion decisions rather than something the decoder itself needed.
