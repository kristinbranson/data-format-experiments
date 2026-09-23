# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every subject-directory NWB file, scans all files with `h5py` in a multiprocessing pool, selects sessions, and converts selected files in a second multiprocessing pass. It reads trials, events, units, spikes, video, subject, and session metadata from NWB/HDF5 paths. Per-session pickle caching avoids reconversion.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
...
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
        sessions.append(result)
```

iii. The trajectory says NWB is the published source, reports checking all 174 files, and uses parallelism/cache because the full output is roughly 10 GB. The agent inspected HDF5 paths and tested several sessions before the full conversion.

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `general/subject/subject_id`; unique IDs are sorted and each session receives the corresponding integer index.

ii.
```python
subject = str(f['general/subject/subject_id'][()].decode())
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
```

iii. The agent treated the NWB subject field as the canonical identifier and reported 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Unlike merely retaining every usable file, the agent first requires performance above 65%, at least 50 correct left trials, at least 50 correct right trials, and at least one good unit. It later drops sessions with no mapped neurons or fewer than two retained trials.

ii.
```python
if s['performance'] <= MIN_PERFORMANCE: reasons.append(...)
if min(s['n_correct_left'], s['n_correct_right']) < MIN_CORRECT_PER_DIRECTION: reasons.append(...)
if s['n_good_units'] == 0: reasons.append('no good units')
...
sessions = [s for s in sessions
            if s['info']['n_neurons'] > 0 and s['info']['n_trials'] >= 2]
```

iii. The trajectory cites the data-paper session-selection text and says its hits/(hits+misses) definition on control, non-early trials reproduced the paper's performance range. This yielded 143/174 sessions, versus the human conversion's 173 usable sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`; go cues are taken in row order from `BehavioralEvents/go_start_times`. A boolean mask selects rows, and every selected trial becomes one matrix in each session list.

ii.
```python
tr = f['intervals/trials']
out['go_time'] = events['go_start_times']['timestamps'][:]
trials = np.where(trial_mask(tt))[0]
neural = [np.ascontiguousarray(rates[i].T) for i in range(n_trials)]
```

iii. The agent inspected event counts/timing and relied on the dataset's one go cue per trial ordering; it also verified sample and go events against trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. It retains trials that are neither auto-water nor free-water, are observed by every good unit according to `obs_intervals`, and have a valid preceding tone. It deliberately keeps photostimulation, early-lick, miss, and ignore trials.

ii.
```python
return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
        & tt['observed'] & tt['tone_valid'])
...
observed &= unit_observed
```

iii. It argued water-delivery trials are not choice-contingent, required early/ignore/stimulation trials for requested labels, and added `obs_intervals` filtering after finding otherwise all-zero neural trials. The human solution excludes free-water and unobserved trials but not auto-water or tone-invalid trials, and uses one representative good unit's observation intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index`, with `units/classification`, CCF annotation/electrode metadata, and go-cue timestamps used for selection, regions, and alignment.

ii.
```python
spike_times = units['spike_times'][:]
stop = units['spike_times_index'][:]
classification = f['units/classification'][:].astype(str)
go_time = tt['go_time'][trials]
```

iii. The agent identified spike times as the raw neural representation and independently spot-checked converted bins against `np.histogram` on raw spikes.

## 2-b. How is the `neural` data processed?

i. For each retained unit, absolute bin edges are formed around every go cue; `searchsorted` differences produce spike counts, which are divided by 0.05 s to obtain Hz. There is no smoothing, normalization, or baseline subtraction. Arrays are transposed to neuron-by-time.

ii.
```python
flat_edges = (go_time[:, None] + BIN_EDGES[None, :]).ravel()
counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
neural = [np.ascontiguousarray(rates[i].T) for i in range(n_trials)]
```

iii. It says this follows the reference's rate computation and validated exact agreement with an independent histogram.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and a CCF annotation that maps into the chosen coarse ontology; unmapped good units are discarded. A session without resulting neurons is dropped.

ii.
```python
good = classification == 'good'
region_names, has_region = unit_regions(f, good, region_map)
good_idx = np.where(good)[0][has_region]
```

iii. The agent tied `classification` to the spike-sorting white-paper classifier and claimed the method paper keeps units with both ephys and histology. It additionally grouped units into hemisphere-qualified coarse regions. The human solution retains all classifier-good units and derives a simpler label, so this extra anatomical exclusion changes neural content.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All timestamps are treated as session-absolute; offsets from -2.5 to +1.5 seconds are added to each go-cue onset, and spikes are binned at those absolute edges.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
flat_edges = edges.ravel()
```

iii. The agent found that NWB streams share a clock and identified go-cue onset as the requested and paper-used alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It uses 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` seconds. Raw spikes are histogrammed into those bins; no further temporal rebinning occurs.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
```

iii. These values come directly from the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and each trial's `go_start_times`; the last sample onset before the go cue is selected.

ii.
```python
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
```

iii. The agent reasoned that early licks can replay the sample/delay epoch, making the last pre-go sample the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone time is expressed relative to go time, then subtracted from each go-relative bin center. Trials without a sample onset within their trial are filtered.

ii.
```python
tone_rel = tt['tone_time'][trials] - go_time
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

iii. It described this as seconds since the last instruction-tone onset, with no interpolation required.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same 80 go-cue-relative bin centers as the neural bins.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. The shared bin grid guarantees one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trials-table `photostim_onset`, `photostim_duration`, and `start_time`, together with go-cue time. `'N/A'` means no stimulation.

ii.
```python
onset = tt['photostim_onset'][trials]
duration = tt['photostim_duration'][trials]
start_time = tt['start_time'][trials]
```

iii. The agent inspected the trials fields and event stream and determined onset is stored relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset/offset are converted to go-relative seconds. A bin is one if its interval overlaps stimulation at all, otherwise zero; nonstimulated trials remain all zero.

ii.
```python
on = start_time[i] + float(onset[i]) - go_time[i]
off = on + float(duration[i])
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The agent explicitly summarized this as a binary overlap with the stimulation window and inspected resulting active-bin timings. The human code classifies the bin center, so boundary bins can differ.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-start-relative onset is converted to the same go-relative coordinates as the neural `BIN_EDGES`, and the resulting 80 flags correspond one-for-one to neural bins.

ii.
```python
on = start_time[i] + float(onset[i]) - go_time[i]
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. The agent used the shared absolute clock/go reference and verified representative onset/offset patterns.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `outcome` and `trial_instruction`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii.
```python
licked_left = (((outcome_str == 'hit') & (instruction == 'left'))
               | ((outcome_str == 'miss') & (instruction == 'right')))
licked_right = (((outcome_str == 'hit') & (instruction == 'right'))
                | ((outcome_str == 'miss') & (instruction == 'left')))
```

iii. The agent says the file has no direct choice column and checked this derivation against the first post-go lick, reporting greater than 99.8% agreement (and 100% in a later spot-check).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are encoded `0=left`, `1=right`, `2=no lick`, then repeated across all 80 bins.

ii.
```python
choice = np.full(n_trials, 2, dtype=np.int64)
choice[licked_left] = 0
choice[licked_right] = 1
np.full(N_BINS, choice[i])
```

iii. Repetition allows trial-level and time-varying outputs to share one `(4,80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` strings.

ii.
```python
outcome_str = tt['outcome'][trials]
```

iii. The stored categories exactly match the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to `ignore=0`, `miss=1`, `hit=2` and are repeated over time.

ii.
```python
outcome[outcome_str == 'ignore'] = 0
outcome[outcome_str == 'miss'] = 1
outcome[outcome_str == 'hit'] = 2
np.full(N_BINS, outcome[i])
```

iii. The coding follows `output_values`; temporal repetition standardizes output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived directly from trials-table `early_lick` (`'early'` versus `'no early'`).

ii.
```python
early = tt['early_lick'][trials]
```

iii. The agent found an explicit trial flag, so no lick-time reconstruction was needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Equality to `'early'` is cast to `0/1` and repeated across 80 bins.

ii.
```python
early_lick = (early == 'early').astype(np.int64)
np.full(N_BINS, early_lick[i])
```

iii. This matches the requested no/yes classes and common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps and data columns 1 (y) and 2 (DeepLabCut likelihood), plus go-cue times.

ii.
```python
tracking = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = tracking['timestamps'][:]
data = tracking['data'][:]
y = data[:, 1]
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
```

iii. The agent inspected the tracking stream and likelihood distribution, finding it sharply bimodal and choosing 0.5 as a robust visibility threshold.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible-frame y values are averaged within each neural bin using cumulative sums and timestamp `searchsorted`. Percentiles are then computed from finite bin means only within retained trials/windows. If fewer than ten finite means exist, all bins remain not visible.

ii.
```python
idx = np.searchsorted(ts, go_time[:, None] + BIN_EDGES[None, :])
n_visible = np.diff(csum_n[idx], axis=1)
sum_y = np.diff(csum_y[idx], axis=1)
mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
...
finite = tongue_y[np.isfinite(tongue_y)]
p40, p60 = np.percentile(finite, [40, 60])
```

iii. It argued averaging should precede discretization and that likelihood removes retracted/unreliable estimates. However, it described this as session percentiles even though its population is only selected trial windows; the human solution bins the entire session before computing thresholds.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th/60th percentiles of the agent's finite retained-window bin means define categories: `<p40` is 0, `p40..p60` is 1, `>p60` is 2, and no visible frame is 3.

ii.
```python
tongue = np.full(tongue_y.shape, 3, dtype=np.int64)
tongue[visible & (tongue_y <= p60)] = 1
tongue[visible & (tongue_y < p40)] = 0
tongue[visible & (tongue_y > p60)] = 2
```

iii. The thresholds and four labels follow the task, but the percentile sample differs from the human session-wide computation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are searched at the same absolute `go_time + BIN_EDGES` boundaries used for spikes, yielding one mean/class per neural bin.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(ts, edges)
```

iii. The agent found camera and neural timestamps share the NWB session clock, so direct common-edge binning is sufficient.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good/mapped units are removed; unobserved, water, and invalid-tone trials are removed; missing/unreliable tongue bins become category 3; sessions with extreme tongue visibility are retained but flagged in metadata. No imputation is applied. Spike bins beyond recorded trial intervals remain zero and are documented.

ii.
```python
if len(good_idx) == 0:
    return np.zeros(len(trial_start), dtype=bool)
...
tongue = np.full(tongue_y.shape, 3, dtype=np.int64)
...
'known_limitation': 'spike times ... only stored inside the trial intervals ...'
```

iii. The agent investigated all-zero trials and partial ephys coverage, then filtered them. It retained error trials with truncated spike coverage because dropping them would largely erase the miss class, and flagged two anomalous tracking sessions rather than losing otherwise valid neural data.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is scanning/opening all NWBs, reading large spike/video arrays, per-unit spike binning, per-session conversion, and serializing the ~9.9-GB output. The code parallelizes scan and conversion across 16 processes and caches sessions.

ii.
```python
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
...
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
```

iii. The trajectory estimated neural payload size in advance, used timeouts/full-run timing, and moved cache files under `/tmp` to make reruns practical.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining notable loops are over good units in `bin_spikes`, good units/coverage patterns in `observed_trials`, trials for photostimulation, and trial arrays during assembly. Photostimulation could be vectorized directly; spike units are ragged and harder to combine. Tongue binning is already vectorized across all trials and bins.

ii.
```python
for i, u in enumerate(unit_ids): ...
for u in good_idx: ...
for i in range(n_trials):
    if onset[i] == 'N/A': continue
```

iii. The agent deliberately vectorized spikes across all trial edges per unit and used cumulative sums for video. It did not explicitly justify the small photostimulation loop, likely because it is negligible relative to I/O/spike binning.

## 10-c. What processing does the code repeat multiple times?

i. Every selected file is opened during `scan_session` and again during `convert_session`; `_trial_table`, `observed_trials`, performance, classification, and subject metadata are consequently read/computed twice. `session_performance` calls `trial_mask` again. Cache use avoids repeating the conversion on later program runs.

ii.
```python
with h5py.File(path, 'r') as f:  # scan_session
    tt = _trial_table(f)
...
with h5py.File(path, 'r') as f:  # convert_session
    tt = _trial_table(f)
```

iii. The two-pass design was chosen so selection metadata could be computed before parallel full conversion; the trajectory emphasizes cache-based reruns but does not discuss this duplicated scan work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It downloads/builds a full Allen ontology and calculates coarse hemisphere-qualified regions, extensive scan/rejection metadata, performance statistics, anomalous-tracking notes, and caches per-session objects. Some are retained only as metadata rather than decoder features; rejected-session scans and temporary caches do not enter downstream decoding. The full tracking stream is read although only selected windows become outputs.

ii.
```python
region_map = build_region_map()
...
'rejected_sessions': [...],
'session_info': [s['info'] for s in sessions],
```

iii. The agent considered anatomy and audit metadata important for matching the papers and documenting limitations, although the decoder itself does not consume them. The human conversion avoids the ontology/download and extra session-performance pass.
