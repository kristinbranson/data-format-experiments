# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing every `.nwb` file under `/app/data/sub-*/`, then processes each file with `h5py` in `process_session`. It does not use `pynwb`; it reads the NWB HDF5 groups directly and runs sessions in parallel with `multiprocessing.Pool`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    infos = []
    for info in pool.imap_unordered(process_session, files):
        infos.append(info)
```

```python
with h5py.File(path, 'r') as f:
    trials = f['intervals/trials']
    ...
    events = f['acquisition/BehavioralEvents']
```

iii. In the trajectory, the agent first established that the dataset contains 174 NWB files organized by subject/session and then decided to process them directly with `h5py` after inspecting the NWB tree. It also explicitly chose parallel session processing for the full conversion after testing smaller runs.

## 1-b. How are the data split into subjects (mice)?

i. The AI splits subjects from the filename, not from `nwb.subject.subject_id`. `session_name()` parses `sub-<subject>` from each file name, and the final `subjects` list is built in first-seen session order while `subject_idx` stores each session's index into that list.

ii.
```python
def session_name(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return sub, ses
```

```python
subject, ses = session_name(path)
info = {'file': os.path.basename(path), 'subject': subject, 'session': ses}
...
if s not in subjects:
    subjects.append(s)
subject_idx.append(subjects.index(s))
```

iii. The trajectory shows that the agent treated the published file layout as authoritative and used the filename-derived subject id consistently after verifying the `sub-*` organization. It did not justify any need to read subject ids from the NWB subject object once it saw the filename convention.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. It parses the `ses-...` token from the filename for the session label and keeps sessions in sorted file order when assembling the final output.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
subject, ses = session_name(path)
info = {'file': os.path.basename(path), 'subject': subject, 'session': ses}
```

```python
kept = sorted([i for i in infos if 'out_path' in i], key=lambda i: i['file'])
```

iii. In the trajectory, the agent concluded early that the dataset is "NWB files per subject/session" and kept that file-level session boundary throughout. No extra within-file grouping was attempted.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table under `intervals/trials`. The AI reads the per-trial columns there, sets `ntrials_all = len(start)`, and checks that the number of go-cue timestamps equals the number of behavioral trials.

ii.
```python
trials = f['intervals/trials']
start = trials['start_time'][:]
stop = trials['stop_time'][:]
outcome = to_str(trials['outcome'][:])
early = to_str(trials['early_lick'][:])
instruction = to_str(trials['trial_instruction'][:])
...
ntrials_all = len(start)
...
go = events['go_start_times/timestamps'][:]
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
    return info
```

iii. The trajectory shows that the agent inspected the NWB structure, confirmed the trials table and go-cue event stream, and used the one-go-cue-per-trial relationship as its trial-level consistency check.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial/session filters: it keeps only trials covered by `units/obs_intervals` for all QC-passing annotated units; computes session-level performance criteria on those recorded trials; removes `auto_water` and `free_water` trials; removes trials with less than 50% expected video-frame coverage in the analysis window; and drops sessions with fewer than 50 remaining trials.

ii.
```python
good0 = np.where((cls0 == 'good') & (anno0 != ''))[0]
...
recorded = np.ones(ntrials_all, dtype=bool)
for uid in good0:
    ...
    recorded &= m
...
control = recorded & (~photostim_trial) & (early == 'no early') & (~auto_water) & (~free_water)
responded = control & (outcome != 'ignore')
performance = float(np.mean(outcome[responded] == 'hit')) if responded.sum() else 0.0
...
keep = recorded & (~auto_water) & (~free_water)
...
nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
expected_frames = (OFF_END - OFF_START) * 300.0
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
...
if len(trial_idx) < MIN_TRIALS_PER_SESSION:
    info['skip'] = 'only %d usable trials (mostly missing video)' % len(trial_idx)
    return info
```

iii. The trajectory gives explicit reasons for each filter: `obs_intervals` was added after the agent found all-zero trials in partially recorded sessions (steps 67-69); `auto_water`/`free_water` were excluded because reward is not contingent on choice (steps 44, 46, 62); and the video-coverage filter plus 50-trial session minimum were added after finding a session with only 7 video-usable trials (steps 82-84).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from `units/spike_times` and `units/spike_times_index`, using `go_start_times/timestamps` to define each trial window. Trial inclusion is additionally constrained by `units/obs_intervals` earlier in the session-processing pipeline.

ii.
```python
go = events['go_start_times/timestamps'][:]
...
spike_times = units['spike_times']
sidx = units['spike_times_index'][:]
edges_lo = win_lo[trial_idx]
```

iii. The trajectory states that the agent verified the spike times are on the session clock and decided to derive firing rates directly from those spike timestamps in 50 ms bins around each go cue.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 80 bins from `go-2.5 s` to `go+1.5 s` for each kept trial and each retained unit. It uses `np.searchsorted` to bound each trial window, `np.bincount` to count spikes per 50 ms bin, and then divides by bin width to convert counts to firing rates in spikes/s.

ii.
```python
neural = np.zeros((ntrials, nneurons, NBINS), dtype=np.float32)
...
for n, uid in enumerate(unit_ids):
    ...
    lo = np.searchsorted(sp, edges_lo)
    hi = np.searchsorted(sp, win_hi[trial_idx])
    for t in range(ntrials):
        if hi[t] <= lo[t]:
            continue
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
neural /= BIN_SIZE  # spikes/s
```

iii. In the trajectory, the agent explicitly chose "spike counts/50 ms -> rates" for the neural representation and justified leaving bins outside the recorded interval as zero because the NWB stores spikes only inside each trial's observation window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units where `classification == 'good'` and `anno_name` is non-empty, then further removes units whose CCF annotation cannot be mapped into its ontology grouping or whose electrode `x` coordinate is not finite.

ii.
```python
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
good = (classification == 'good') & (anno != '')
...
for i in unit_ids:
    reg = coarse_region(anno[i], utarget[i])
    x = ux[i]
    if reg is None or not np.isfinite(x):
        keep_unit.append(False)
        regions.append(None)
        continue
```

iii. The trajectory repeatedly states that `classification == 'good'` plus CCF annotation reproduced the published unit counts, and that annotation plus electrode coordinates were needed for its hemisphere/coarse-region labeling scheme (steps 53, 54, 58, 61, 62).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to go-cue onset by constructing absolute trial windows with `win_lo = go + OFF_START` and `win_hi = go + OFF_END`, where `OFF_START = -2.5` and `OFF_END = 1.5`.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
...
go = events['go_start_times/timestamps'][:]
...
win_lo = go + OFF_START
win_hi = go + OFF_END
```

iii. The trajectory says the agent verified that spikes, behavior events, and video all share the same session-time clock, so alignment only required offsetting each trial by its go-cue timestamp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and 80 total bins spanning 4 s. There is no secondary temporal rebinning or smoothing step.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

```python
b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
...
neural /= BIN_SIZE
```

iii. The trajectory explicitly notes that the reference code used a different neural binning scheme, but the agent switched to 50 ms bins because the decoder instructions required it.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps` and `go_start_times/timestamps`. The AI selects the last sample-epoch start before each go cue.

ii.
```python
sample_on = events['sample_start_times/timestamps'][:]
...
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go
```

iii. The trajectory says the agent verified that "tone onset (last sample start before go) [is] available for every trial" and specifically chose the last sample onset because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes bin centers on the go-cue-aligned grid, computes the tone time relative to go, and then fills each kept trial's first input row with `bin_centers - tone_rel_go`, i.e. seconds since tone onset at each bin.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
...
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go
...
for k, t in enumerate(trial_idx):
    inputs[k, 0] = bin_centers - tone_rel_go[t]
```

iii. In the trajectory, the agent justified this as a signed, time-varying decoder input on the same 50 ms grid as the neural data rather than as a one-time event marker.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by being evaluated on exactly the same go-cue-relative bin centers used for the neural trial windows. The same `trial_idx` filtering is also applied.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
...
inputs = np.zeros((ntrials, 2, NBINS), dtype=np.float32)
for k, t in enumerate(trial_idx):
    inputs[k, 0] = bin_centers - tone_rel_go[t]
```

iii. The trajectory treats all decoder inputs and outputs as living on the same 80-bin go-cue-centered time base so they can be paired directly with the neural arrays.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived primarily from the event streams `photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, with `trials/start_time` used to map each stimulation interval back to a trial. The trials-table `photostim_onset` string is read too, but only for session-level control-trial filtering and metadata.

ii.
```python
photostim_trial = to_str(trials['photostim_onset'][:]) != 'N/A'
...
stim_on = events['photostim_start_times/timestamps'][:]
stim_off = events['photostim_stop_times/timestamps'][:]
...
stim_trial = np.searchsorted(start, stim_on) - 1
```

iii. The trajectory states that the agent checked event/table consistency across sessions and found that "photostim events match the table perfectly," after which it chose to use the event timestamps directly for the time-varying decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary time series per kept trial. For each stimulation interval mapped to a kept trial, it converts the onset and offset into bin indices relative to the analysis window and marks all overlapping bins as `1.0`.

ii.
```python
inputs = np.zeros((ntrials, 2, NBINS), dtype=np.float32)
...
for on, off, tr in zip(stim_on, stim_off, stim_trial):
    if tr < 0 or tr >= ntrials_all or not keep[tr]:
        continue
    k = int(np.searchsorted(trial_idx, tr))
    b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
    b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
    ...
    if b1 > b0:
        inputs[k, 1, b0:b1] = 1.0
```

iii. In the trajectory and final validation, the agent justified this as a binary time-varying input and noted that the typical occupied interval is roughly `-1.25` to `-0.7` s relative to the go cue, matching the late-delay photostimulation protocol.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by expressing its bin indices relative to each trial's `win_lo = go - 2.5 s`, so the resulting binary series sits on the same 80-bin grid as the neural data.

ii.
```python
win_lo = go + OFF_START
...
b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
...
inputs[k, 1, b0:b1] = 1.0
```

iii. The trajectory says all streams share the same session clock, so direct conversion into bin indices on the go-cue-relative window was sufficient for alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from a stored lick-choice field. The AI derives it from `trial_instruction` and `outcome`: `hit` means the instructed side, `miss` means the opposite side, and other outcomes are treated as no lick.

ii.
```python
instruction = to_str(trials['trial_instruction'][:])
outcome = to_str(trials['outcome'][:])
...
choice_map = {'left': 0, 'right': 1}
for k, t in enumerate(trial_idx):
    if outcome[t] == 'hit':
        ch = choice_map[instruction[t]]
    elif outcome[t] == 'miss':
        ch = choice_map['right' if instruction[t] == 'left' else 'left']
    else:
        ch = 2
```

iii. The trajectory shows that the agent explicitly compared lick-based and outcome/instruction-based choice logic and chose the latter derived formulation as the stable categorical output.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes choice as `0=left`, `1=right`, `2=no lick` and repeats that per-trial class across all 80 bins.

ii.
```python
choice_map = {'left': 0, 'right': 1}
...
outputs = np.zeros((ntrials, 4, NBINS), dtype=np.int64)
...
outputs[k, 0, :] = ch
```

iii. In the trajectory, the agent justified keeping `ignore`/no-response trials because the decoder task explicitly requires them as an output class rather than dropping them as in some reference analyses.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii.
```python
outcome = to_str(trials['outcome'][:])
```

iii. The trajectory repeatedly treats `outcome` as a native per-trial categorical variable already stored in the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to integers `ignore=0`, `miss=1`, `hit=2` and repeats that value across all 80 bins for each kept trial.

ii.
```python
outputs[k, 1, :] = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome[t]]
```

iii. The trajectory justification is that the decoder requires categorical outputs; outcome already has exactly the three target categories, so only integer coding and broadcasting across time were needed.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii.
```python
early = to_str(trials['early_lick'][:])
```

iii. The trajectory says the agent kept early-lick trials, despite the reference analyses often excluding them, because the decoder task explicitly asks for early lick as an output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early` to `1` and any other value (`no early`) to `0`, then repeats the class across all 80 bins.

ii.
```python
outputs[k, 2, :] = 1 if early[t] == 'early' else 0
```

iii. The trajectory justification is the same as for outcome and choice: the decoder expects categorical outputs on the shared neural time grid, so the per-trial label is broadcast across bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically `data[:, 1]` for tongue y-position and `data[:, 2]` for DeepLabCut likelihood, with `timestamps` used for temporal alignment.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. The trajectory shows that the agent inspected the tracking channels and their likelihood distribution, concluded that tongue visibility is sparse and bimodal, and used that side-camera tongue track as the only tongue output source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks frames as visible when likelihood is above `0.9`, computes session-wide 40th/60th percentiles from visible raw `tongue_y` frames, drops trials whose `[-2.5, 1.5]` window contains less than half of the expected video frames, then for each remaining trial and bin takes the median visible `tongue_y` among frames in that bin.

ii.
```python
LIKELIHOOD_THRESH = 0.9
VIDEO_COVERAGE_MIN = 0.5
...
visible = vdata[:, 2] > LIKELIHOOD_THRESH
if visible.sum() > 0:
    p40, p60 = np.percentile(tongue_y[visible], [40, 60])
...
nframes = np.searchsorted(vts, win_hi) - np.searchsorted(vts, win_lo)
expected_frames = (OFF_END - OFF_START) * 300.0
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
...
for bi in range(NBINS):
    m = (b == bi) & vv
    if not np.any(m):
        continue
    y = float(np.median(yy[m]))
    cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. The trajectory justification is explicit: the agent inspected the likelihood distribution, chose a high visibility threshold because visibility looked effectively bimodal, and later added the video-coverage filter after finding a session where only 1.3% of trials had enough frames for the decoder output.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` if the bin's median visible tongue y is below the session 40th percentile, `1` if it is between the 40th and 60th percentiles, `2` if above the 60th percentile, and `3` if no visible tongue frame falls in the bin.

ii.
```python
cls = np.full(NBINS, 3, dtype=np.int64)
...
if not np.any(m):
    continue
y = float(np.median(yy[m]))
cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. In the trajectory and final validation, the agent described this explicitly as "per-session percentiles over visible frames" plus a `"not visible"` class for bins with no confident tongue detection.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by taking camera timestamps within each trial's go-cue-centered window and binning them relative to `win_lo = go - 2.5 s`, the same left edge used for the neural analysis window.

ii.
```python
win_lo = go + OFF_START
win_hi = go + OFF_END
...
lo = np.searchsorted(vts, win_lo[t])
hi = np.searchsorted(vts, win_hi[t])
...
b = ((tt - win_lo[t]) / BIN_SIZE).astype(np.int64)
np.clip(b, 0, NBINS - 1, out=b)
```

iii. The trajectory says the agent verified that video timestamps live on the same session clock as spikes and events, so this direct binning was its alignment method.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several failure modes defensively: `to_str()` converts bytes and missing/NaN text entries into usable strings; sessions with no QC-passing annotated units are skipped; sessions with mismatched go/trial counts are skipped; sessions with no visible tongue frames are skipped; low-video-coverage trials are dropped; and tongue bins with no visible frames are assigned class `3`.

ii.
```python
def to_str(arr):
    return np.array([x.decode() if isinstance(x, bytes) else ('' if isinstance(x, float) else str(x))
                     for x in arr])
```

```python
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
    return info
...
if len(good0) == 0:
    info['skip'] = 'no QC-passing units with CCF annotation'
    return info
...
if not np.isfinite(p40):
    info['skip'] = 'no visible tongue frames in session'
    return info
```

```python
cls = np.full(NBINS, 3, dtype=np.int64)
```

iii. The trajectory explains these choices as avoiding fabricated data: `obs_intervals` was added after discovering all-zero spike trials, the one un-QC'ed session was excluded because classification/annotation were missing, and no-visibility tongue bins were represented explicitly instead of imputing values.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI code is per-session NWB I/O plus the nested unit-by-trial spike binning loop. Secondary heavy work is reading the full tongue-tracking arrays and the per-trial/per-bin tongue classification loops. The code also writes every processed session to a temporary pickle and re-reads those files during assembly.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        ...
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
```

```python
for k, t in enumerate(trial_idx):
    ...
    for bi in range(NBINS):
        ...
        y = float(np.median(yy[m]))
```

```python
with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=4)
...
with open(info['out_path'], 'rb') as fh:
    sess = pickle.load(fh)
```

iii. The trajectory does not give a formal profiler output, but it does show the agent deliberately using `Pool` and temporary per-session pickle files to make the full conversion feasible at dataset scale.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the intersection of `obs_intervals` coverage across units, the nested neural loop over units and then trials, the per-trial construction of the time-from-tone input, the per-trial output-label loop, and especially the per-trial/per-bin tongue discretization loop.

ii.
```python
for uid in good0:
    ...
    recorded &= m
```

```python
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        ...
```

```python
for k, t in enumerate(trial_idx):
    inputs[k, 0] = bin_centers - tone_rel_go[t]
```

```python
for k, t in enumerate(trial_idx):
    ...
    for bi in range(NBINS):
        ...
```

iii. The trajectory does not explicitly defend these loops as optimal; it only shows that the agent prioritized getting a working full conversion and used process-level parallelism rather than reworking the inner loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some processing. It reads `classification` and `anno_name` twice (`units0` for coverage/session filtering, then `units` for final unit selection), writes each processed session to a temporary pickle and then loads it again during assembly, and does repeated linear `list.index()` lookups while building `brain_region_idx` and `subject_idx`.

ii.
```python
units0 = f['units']
cls0 = to_str(units0['classification'][:])
anno0 = to_str(units0['anno_name'][:])
...
units = f['units']
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
```

```python
with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=4)
...
with open(info['out_path'], 'rb') as fh:
    sess = pickle.load(fh)
```

```python
for r in sess['regions']:
    if r not in brain_regions:
        brain_regions.append(r)
data['brain_region_idx'].append(
    np.array([brain_regions.index(r) for r in sess['regions']], dtype=np.int64))
...
if s not in subjects:
    subjects.append(s)
subject_idx.append(subjects.index(s))
```

iii. The trajectory does not mention these as deliberate optimizations; they appear to be implementation choices made while scaling the conversion to the full dataset.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary/discarded work is the temporary-session serialization pipeline: each session is written to `/app/tmp_sessions/*.pkl` and then read back only to build the final pickle. The script also computes extensive session-level curation statistics and a coarse ontology/hemisphere mapping that are only used for metadata and `brain_region_idx`, not for the decoder inputs/outputs themselves.

ii.
```python
TMP_DIR = '/app/tmp_sessions'
...
result = {
    'info': info,
    'neural': [neural[k] for k in range(ntrials)],
    'input': [inputs[k] for k in range(ntrials)],
    'output': [outputs[k] for k in range(ntrials)],
    ...
}
out_path = os.path.join(TMP_DIR, '%s_%s.pkl' % (subject, ses))
with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=4)
```

```python
info.update(performance=performance, n_correct_left=n_correct_left,
            n_correct_right=n_correct_right, n_trials_total=int(ntrials_all),
            n_trials_recorded=int(recorded.sum()))
```

iii. The trajectory shows these choices were made pragmatically: temporary per-session files were part of the agent's scaling strategy, and the extra curation statistics were used to justify/monitor filtering decisions rather than to create decoder features.
