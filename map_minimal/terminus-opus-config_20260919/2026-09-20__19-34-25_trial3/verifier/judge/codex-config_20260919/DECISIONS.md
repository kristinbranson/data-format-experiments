# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `sub-*/*.nwb` file, sorts the paths, and processes the files in parallel with `h5py`. Each worker reads the trials, behavioral events, units, electrodes, and tongue-tracking groups, writes a temporary per-session pickle, and the main process combines retained sessions.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with Pool(args.nproc) as pool:
    for info in pool.imap_unordered(process_session, files):
        infos.append(info)
```

iii. The trajectory says the agent identified the NWB files as one file per session, used multiprocessing for speed, and tested a small subset before converting all 174 files.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from each filename's `sub-<id>` prefix. During assembly, subjects are accumulated in first-seen order and every retained session gets an integer index.

ii.
```python
sub = base.split('_')[0].replace('sub-', '')
...
if s not in subjects:
    subjects.append(s)
subject_idx.append(subjects.index(s))
```

iii. The agent treated the filename subject identifier as the canonical mouse identifier; sorted session filenames make the resulting ordering deterministic.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The acquisition timestamp is parsed from the filename, and only sessions surviving unit, behavioral-performance, correct-trial-count, video, and minimum-usable-trial checks are assembled. The final dataset has 142 of 174 sessions.

ii.
```python
subject, ses = session_name(path)
...
kept = sorted([i for i in infos if 'out_path' in i], key=lambda i: i['file'])
```

iii. The trajectory says the session boundary came directly from the NWB layout. The additional session filters were chosen from the paper's stated `>65%` performance and 50 correct trials per direction criteria; the 50-usable-trial rule was added after finding a session with only seven video-covered trials.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The code reads trial start/stop and trial-level labels, and requires exactly one go-cue timestamp per row. Retained row indices are used consistently for neural, input, and output arrays.

ii.
```python
trials = f['intervals/trials']
start = trials['start_time'][:]
stop = trials['stop_time'][:]
...
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
```

iii. The agent found the trial table to be the explicit trial definition and used the go-count assertion to guard against an ambiguous row/event mapping.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be covered by `obs_intervals` for every initially eligible unit, must not be auto-water or free-water trials, and must contain at least half the expected side-camera frames in the four-second window. Sessions are also filtered by control-trial performance, at least 50 correct trials in each direction, and at least 50 final usable trials. Photostimulation, early-lick, miss, and ignore trials are retained.

ii.
```python
recorded &= m
...
control = recorded & (~photostim_trial) & (early == 'no early') & (~auto_water) & (~free_water)
...
keep = recorded & (~auto_water) & (~free_water)
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
if len(trial_idx) < MIN_TRIALS_PER_SESSION:
    return info
```

iii. The agent discovered all-zero trials during verification and traced them to partial ephys coverage, motivating `obs_intervals`. It excluded noncontingent-water trials, retained task-required classes, and imposed video/minimum-trial checks to avoid largely undefined tongue outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `units/spike_times` array and its `spike_times_index`; go-cue timestamps determine each trial window. Unit classification, annotation, electrodes, CCF x-coordinate, and ontology mapping determine which units and regions are retained.

ii.
```python
spike_times = units['spike_times']
sidx = units['spike_times_index'][:]
edges_lo = win_lo[trial_idx]
```

iii. The agent identified spike times as the raw neural measurement and used the common absolute clock shared by spikes and behavioral events.

## 2-b. How is the `neural` data processed?

i. For every retained unit and trial, spikes in the analysis window are assigned to 50-ms bins with integer indices and `bincount`. Counts are divided by 0.05 seconds to obtain firing rates in Hz. There is no smoothing or normalization.

ii.
```python
b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
...
neural /= BIN_SIZE
```

iii. The agent explicitly chose spike count divided by bin width to implement the requested 50-ms firing rates and match the paper's processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'`, a nonempty `anno_name`, an ontology-resolvable coarse region, and finite electrode x-coordinate. Sessions with no eligible units are removed.

ii.
```python
good = (classification == 'good') & (anno != '')
...
if reg is None or not np.isfinite(x):
    keep_unit.append(False)
```

iii. The agent cited the region-specific QC classifier and paper/white-paper curation, and required CCF localization so every neuron could receive a hemisphere and brain-region label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is `[go - 2.5, go + 1.5)` on the session-absolute clock. Spike offsets from the lower edge determine the same 80 go-aligned bins for every trial.

ii.
```python
win_lo = go + OFF_START
win_hi = go + OFF_END
b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
```

iii. The agent reasoned that spikes and events share a clock, so adding fixed offsets to each go cue performs the required alignment without interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins spanning four seconds. Raw spike events are histogrammed directly into these bins; no further resampling is applied.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. This is the resolution and window explicitly required by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and `go_start_times`. For each trial, the last sample start before the go cue is selected.

ii.
```python
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go
```

iii. The agent noted that early licking can replay the sample epoch, making the last tone before go the relevant onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Go-relative bin centers are computed, then the selected tone's go-relative time is subtracted, yielding seconds since tone onset at every bin center.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
inputs[k, 0] = bin_centers - tone_rel_go[t]
```

iii. The agent considered this a direct clock translation with no additional filtering or scaling.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the centers of the exact same 80 go-cue-relative bins used for spike rates.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
```

iii. The common go-relative grid guarantees elementwise temporal correspondence.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`; trial starts map each stimulation event to a trial.

ii.
```python
stim_on = events['photostim_start_times/timestamps'][:]
stim_off = events['photostim_stop_times/timestamps'][:]
stim_trial = np.searchsorted(start, stim_on) - 1
```

iii. The agent preferred the recorded event timestamps because they directly define the actual illumination interval on the same clock as neural activity.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The input starts at zero. Each retained stimulation interval is converted to the range of 50-ms bins it overlaps using floor for onset and ceil for offset, and all those bins are set to one.

ii.
```python
b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
inputs[k, 1, b0:b1] = 1.0
```

iii. The agent intended a binary time-varying indicator and described its observed timing as consistent with the methods.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation timestamps are offset from the same `go - 2.5` lower edge used for spike bins.

ii.
```python
b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
```

iii. Shared absolute timestamps and the shared window origin provide alignment without a separate clock correction.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii.
```python
if outcome[t] == 'hit':
    ch = choice_map[instruction[t]]
elif outcome[t] == 'miss':
    ch = choice_map['right' if instruction[t] == 'left' else 'left']
else:
    ch = 2
```

iii. The trajectory accepted this derivation because no direct choice column exists and instruction crossed with outcome uniquely determines the response class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, or 2 no lick and repeated across all 80 time bins.

ii.
```python
outputs[k, 0, :] = ch
```

iii. It is a per-trial categorical output; repetition lets it share the standard time-indexed output matrix.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` field.

ii.
```python
outcome = to_str(trials['outcome'][:])
```

iii. The stored categories already match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, and 2 hit, then repeated through time.

ii.
```python
outputs[k, 1, :] = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome[t]]
```

iii. The encoding follows the declared `output_values`; outcome is trial-level rather than time-varying.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` string field.

ii.
```python
early = to_str(trials['early_lick'][:])
```

iii. The NWB trial table explicitly provides the requested label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` is encoded as 1 and every other expected value (`'no early'`) as 0, repeated across all bins.

ii.
```python
outputs[k, 2, :] = 1 if early[t] == 'early' else 0
```

iii. This makes the per-trial binary flag compatible with the common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and data from `Camera0_side_TongueTracking`: column 1 is y-position and column 2 is tracking likelihood.

ii.
```python
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. The agent identified the side-camera DeepLabCut stream as the available tongue measurement and likelihood as its visibility indicator.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only when likelihood exceeds 0.9. Within each trial/bin, the median y of visible frames is used; a bin with no visible frame remains class 3. Trials with less than 50% expected video-frame coverage are discarded.

ii.
```python
visible = vdata[:, 2] > LIKELIHOOD_THRESH
...
y = float(np.median(yy[m]))
...
cls = np.full(NBINS, 3, dtype=np.int64)
```

iii. Diagnostics showed likelihood was nearly binary, leading the agent to choose a stringent 0.9 threshold. Median aggregation was chosen as a robust per-bin summary, and missing detections were represented explicitly rather than imputed.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session from all visible raw frame y-values. A per-bin median below p40 is 0, from p40 through p60 is 1, above p60 is 2, and no visible frame is 3.

ii.
```python
p40, p60 = np.percentile(tongue_y[visible], [40, 60])
...
cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. The per-session percentile scope and category boundaries follow the task, while the agent chose raw visible frames as the percentile population.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go - 2.5, go + 1.5)` are selected via timestamps and assigned to 50-ms bins by offset from the same lower edge as the neural window.

ii.
```python
lo = np.searchsorted(vts, win_lo[t])
hi = np.searchsorted(vts, win_hi[t])
b = ((tt - win_lo[t]) / BIN_SIZE).astype(np.int64)
```

iii. The agent relied on camera, event, and spike timestamps sharing the NWB session clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-string/NaN text becomes an empty string; sessions without QC/annotation are dropped. Trials without common ephys coverage, noncontingent-water trials, and trials with poor video coverage are dropped. No visible tongue in a bin becomes category 3. Go-count mismatch, no visible tongue in a session, or too few retained trials causes the session to be skipped.

ii.
```python
return np.array([x.decode() if isinstance(x, bytes) else ('' if isinstance(x, float) else str(x))
                 for x in arr])
...
cls = np.full(NBINS, 3, dtype=np.int64)
```

iii. The agent's guiding distinction was to remove observations for which core streams were absent, while representing legitimate time-local tongue invisibility with an explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is reading large NWB datasets, nested unit-by-trial spike histogramming, per-trial/per-bin tongue processing, writing temporary session pickles, then reading them and writing the roughly 9-GB combined pickle. Multiprocessing reduces wall-clock conversion time.

ii.
```python
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        neural[t, n] = np.bincount(...)
```

iii. The trajectory reports fast parallel conversion but a 9.2-GB output; it focused optimization on session-level multiprocessing and later spent substantial time training/validation rather than conversion itself.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit/per-trial neural loop could vectorize trial edge searches/count differences, as the reference does. The per-trial/per-bin tongue loop could use grouped sums/counts or bincounts. Subject/region assembly also uses repeated list membership/index scans.

ii.
```python
for n, uid in enumerate(unit_ids):
    for t in range(ntrials):
        ...
for bi in range(NBINS):
    m = (b == bi) & vv
```

iii. The agent did not explicitly justify these residual nested loops; it prioritized correctness and parallelized across sessions.

## 10-c. What processing does the code repeat multiple times?

i. Unit classification and annotation are loaded twice in each session (`units0` then `units`). Ephys coverage is recomputed for every good annotated unit even though the trajectory found units within a session share coverage. The 80-bin center/tone mapping is also rebuilt per session, and temporary results are serialized then deserialized.

ii.
```python
cls0 = to_str(units0['classification'][:])
anno0 = to_str(units0['anno_name'][:])
...
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
```

iii. No justification was recorded for duplicated reads/checks; temporary files were a deliberate way to collect parallel session results without holding worker outputs in the parent during processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes/parses unused `stop` times, downloads and traverses the full Allen ontology to create coarse/hemisphere labels not needed for the requested decoder variables, computes session diagnostics, writes extensive temporary metadata, and writes then rereads temporary pickles. Some derived fields are retained only in metadata, not decoder training.

ii.
```python
stop = trials['stop_time'][:]
...
ONTOLOGY = load_ontology()
...
with open(out_path, 'wb') as fh:
    pickle.dump(result, fh, protocol=4)
```

iii. Region processing was justified as satisfying required `brain_region_idx`; diagnostics and temporary serialization supported validation and parallel assembly, though they do not affect downstream decoder values.
