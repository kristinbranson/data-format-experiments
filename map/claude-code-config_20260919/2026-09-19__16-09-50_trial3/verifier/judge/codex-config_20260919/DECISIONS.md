# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all `/app/data/sub-*/*.nwb` files, sorts them, performs a cheap first pass for session selection, then processes selected files (normally in a multiprocessing pool) with `h5py`. It reads trial, event, unit, electrode, and tongue datasets directly from each NWB/HDF5 file.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
for fp in files:
    with h5py.File(fp, 'r') as f:
        tr = read_trial_table(f)
...
with ctx.Pool(args.njobs, initializer=_worker_init) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
        results.append(res)
```

iii. The notes say the dataset is one NWB file per recording session (174 files, 28 subjects); direct HDF5 reads and session parallelism were chosen for speed.

## 1-b. How are the data split into subjects?

i. The AI obtains the DANDI subject id but uses the mouse name parsed from `nwb.identifier` (for example `SC015`) as the output subject. It builds sorted unique subjects and indexes each retained session into them, while asserting mouse-name/subject-id correspondence is one-to-one.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
mouse = session_id.split('_')[0]
...
subjects = sorted({r['subject'] for r in results})
'subject_idx': np.array([sub_to_idx[r['subject']] for r in results], dtype=np.int64),
```

iii. The AI regarded mouse names as more interpretable and verified their one-to-one mapping to DANDI ids. The reference instead uses `nwb.subject.subject_id` directly.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are identified by `identifier`, filtered using paper performance/count/QC criteria, converted independently, then sorted by session id. The final dataset contains 150 sessions.

ii.
```python
session_id = f['identifier'][()].decode()
ok, perf, ncl, ncr, reason = session_passes(f, tr)
...
results.sort(key=lambda r: r['session_id'])
```

iii. The AI justified session filtering by the data paper’s criteria: control performance above 65% and at least 50 correct trials per direction. The human conversion does not apply this paper-level session selection and retains 173 sessions with QC units.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials and are paired one-to-one with `go_start_times`; the AI errors if counts differ. Retained trial rows and their corresponding go cues are selected by a Boolean mask.

ii.
```python
ntrials = len(tr['start_time'])
if len(go) != ntrials:
    raise ValueError(...)
tr['go_time'] = go
...
go = tr['go_time'][keep]
```

iii. The notes state that the NWB trial table is authoritative and that every trial has one go cue, whereas sample events can repeat after early licks.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials within unit `obs_intervals`, removes both auto-water and free-water trials, and later drops completely silent four-second windows. It retains early-lick, ignore, and photostimulation trials. It also requires at least two usable trials per session.

ii.
```python
return ((tr['auto_water'] == 0) & (tr['free_water'] == 0) &
        observed_trial_mask(f, tr))
...
silent = fr.sum(axis=(1, 2)) == 0
keep[kept_positions[silent]] = False
```

iii. Trials lacking recorded spikes would create artificial all-zero matrices; water-delivery trials were considered not to reflect a decision. Required decoder classes/inputs motivated retaining early-lick, ignore, and photostim trials. The reference removes free-water and unobserved trials but not auto-water trials, and does not add the silent-window test.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times`, restricted by `units/classification == 'good'`; `go_start_times` supplies trial alignment.

ii.
```python
good = np.flatnonzero(classification == 'good')
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
spikes = [spike_times[starts[i]:ends[i]] for i in good]
```

iii. The AI reports that classifier-good units reproduce paper area counts and correspond to the white-paper QC verdict.

## 2-b. How is the `neural` data processed?

i. For each unit, the code uses `searchsorted` at all trial bin edges, differences cumulative positions into spike counts, and divides by 0.05 seconds to yield unsmoothed firing rates in Hz.

ii.
```python
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    fr[:, i, :] = np.diff(pos, axis=1)
fr /= BIN_SIZE
```

iii. This matches the reference pipeline’s rate convention; the AI explicitly retained Hz even though unstandardized decoder training scored slightly better with counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `good` are retained. No individual metric thresholds or per-trial `is_good_trials` mask is applied; sessions with no good units fail selection.

ii.
```python
classification = _dec(u['classification'][:])
good = np.flatnonzero(classification == 'good')
...
elif ngood == 0:
    reason = 'no units passed quality control'
```

iii. The notes say the classifier already incorporates the QC metrics and reproduces published counts; per-trial unit filtering cannot fit a rectangular neuron-by-time representation and affects very few pairs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue time is added to fixed edges from -2.5 to +1.5 seconds, and absolute spike times are counted between those edges.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
```

iii. All NWB timestamps share a session clock, so the AI says no offset correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI creates 80 non-overlapping 50-ms bins over four seconds. Raw spike times are newly histogrammed into this grid; there is no smoothing or further rebinning.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

iii. The 50-ms resolution and window are mandated by the decoder task, superseding the paper’s 40-ms sliding windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, `go_start_times`, and the fixed bin centers; the tone is the last sample start before each go cue.

ii.
```python
idx = np.searchsorted(sample_start, go, side='right') - 1
tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
```

iii. Early licks can replay the sample epoch, so the last preceding tone is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI subtracts tone time from each absolute bin-center time, algebraically using go-relative centers. If no preceding sample event exists, it imputes the nominal tone time `go - 1.85` seconds.

ii.
```python
tone = np.where(np.isnan(tone), go - 1.85, tone)
inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)
```

iii. The nominal fallback equals the 0.65-s sample plus 1.2-s delay. The reference assumes a preceding event and has no fallback, though the fallback appears defensive rather than active on this dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at exactly the same 80 go-relative bin centers as the neural firing rates.

ii.
```python
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)
inp[:, 0, :] = BIN_CENTERS[None, :] - (tone - go)[:, None]
```

iii. Using the shared go-cue grid guarantees one input value per neural bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It derives stimulation from trial-table `photostim_onset`, `photostim_duration`, and `start_time`, together with each go cue.

ii.
```python
onset_str = tr['photostim_onset'][keep]
dur_str = tr['photostim_duration'][keep]
start = tr['start_time'][keep]
```

iii. The trial columns directly specify relative onset and duration; `N/A` marks unstimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String values are parsed to floats, onset is converted from trial-relative to go-relative time, and a bin is 1 when its center lies in the half-open stimulation interval, otherwise 0.

ii.
```python
on_rel = start + onset - go
off_rel = on_rel + dur
inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
          (BIN_CENTERS[None, :] < off_rel[:, None]))
```

iii. The AI cross-checked derived intervals against raw photostimulation event counts and notes stimulation ends by the go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. On/off bounds are expressed relative to the go cue and compared with the same bin centers used by neural data.

ii.
```python
on_rel = start + onset - go
inside = (BIN_CENTERS[None, :] >= on_rel[:, None]) & ...
```

iii. Shared session timestamps and the common go-relative bin grid provide direct alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial `outcome` and `trial_instruction`, rather than directly from lick timestamps.

ii.
```python
outcome = tr['outcome'][keep]
instr = tr['instruction'][keep]
```

iii. A hit means the instructed side, a miss the opposite side, and ignore no lick; the AI reports checking this against post-go lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are 0 left, 1 right, 2 no lick, repeated across all 80 bins.

ii.
```python
choice = np.where(outcome == 'hit', instr_code,
                  np.where(outcome == 'miss', opposite, CHOICE_NOLICK))
out[:, 0, :] = choice_codes(tr, keep)[:, None]
```

iii. This implements the requested three categories; repetition gives a common time-varying output array for a per-trial label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial table’s `outcome` strings.

ii.
```python
'outcome': _dec(t['outcome'][:])
```

iii. The stored categories exactly match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` map to 0, 1, and 2 and are repeated over time.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. The coding follows the requested order and maintains a uniform output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial table’s `early_lick` field.

ii.
```python
'early_lick': _dec(t['early_lick'][:])
```

iii. The NWB table explicitly records whether an early lick occurred.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other expected value (`no early`) to 0, repeated across bins.

ii.
```python
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```

iii. This matches the no/yes category order and treats it as a per-trial label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps, y coordinate (data column 1), and DLC likelihood (column 2), plus go cues.

ii.
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = tt['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. The series description identifies these columns and it is the dataset’s side-view tongue measurement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood above 0.9 are visible. Visible y values are averaged within each retained trial’s 50-ms bins; empty bins remain NaN. Percentiles are then computed over visible bin means from retained trials only.

ii.
```python
vis = lik > TONGUE_LIKELIHOOD_THRESHOLD
counts = np.bincount(b, minlength=N_BINS)
sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
y_bin[i, nz] = sums[nz] / counts[nz]
...
p40, p60 = np.percentile(y_bin[visible], [40.0, 60.0])
```

iii. The AI says likelihood is strongly bimodal, making 0.9 insensitive, and bin-level percentiles match the quantity classified. Unlike the reference, it does not calculate percentile edges from visible bins across the entire session, only retained trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, values below p40 are class 0, inclusive p40–p60 class 1, above p60 class 2, and bins without a visible frame class 3.

ii.
```python
out = np.full(y_bin.shape, 3, dtype=np.int64)
code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
out[visible] = code[visible]
```

iii. This follows the requested four categories and handles exact boundaries in the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are sliced from `go-2.5` to `go+1.5` and assigned to the same 50-ms go-relative bins.

ii.
```python
lo = np.searchsorted(ts, go_times + OFF_START)
hi = np.searchsorted(ts, go_times + OFF_END)
b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. Camera, event, and spike timestamps share the NWB session clock, so identical boundaries align the modalities.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI validates event/trial counts and observation intervals, drops sessions with no good units, filters unobserved and fully silent trials, imputes a nominal tone only if no prior sample exists, and represents absent visible tongue frames with class 3. Worker exceptions become excluded results.

ii.
```python
if len(go) != ntrials: raise ValueError(...)
tone = np.where(np.isnan(tone), go - 1.85, tone)
silent = fr.sum(axis=(1, 2)) == 0
out = np.full(y_bin.shape, 3, dtype=np.int64)
```

iii. The notes describe partial ephys coverage and one no-QC session as real data issues; missing measurements are excluded or explicitly categorized rather than silently treated as valid zeros.

## 10-a. What are the most time-consuming steps of the code?

i. Full NWB array I/O, spike binning, tongue loading/binning, and writing the roughly 10-GB pickle dominate. Sessions are parallelized; reported conversion wall time was about 22 seconds with many workers, apart from saving/validation.

ii.
```python
with ctx.Pool(args.njobs, initializer=_worker_init) as pool:
    ...
spike_times = u['spike_times'][:]
with open(args.outfile, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes’ timing table identifies reading large spike/video buffers and binning as the session costs, and multiprocessing as the main acceleration.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. A loop remains over units for ragged spike arrays and over trials for tongue frames; diagnostic plotting also loops over trials/bins. Trials are already vectorized within each unit via flattened edges and frames within each tongue trial via `bincount`.

ii.
```python
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges)
...
for i in range(ntrials):
    ...
    counts = np.bincount(b, minlength=N_BINS)
```

iii. The AI says ragged per-unit spike arrays prevent a simple single search, while the tongue loop is small and clearer; the reference makes the same broad loop choices.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass design rereads each selected file’s trial table and recomputes session eligibility/trial masks: once during selection and again inside `process_session`. `trial_mask` is also invoked within `session_passes` and later again for conversion. Summary construction concatenates all outputs/inputs again to calculate fractions and ranges.

ii.
```python
# pass 1
tr = read_trial_table(f)
ok, ... = session_passes(f, tr)
...
# pass 2
tr = read_trial_table(f)
ok, ... = session_passes(f, tr)
keep = trial_mask(f, tr)
```

iii. The AI calls the first pass cheap and uses it to avoid expensive full processing of excluded sessions, accepting repeated metadata reads and checks.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. During conversion it loads annotation strings and CCF coordinates that are only used to derive region indices, stores several diagnostics in intermediate result dictionaries, and performs whole-dataset summary concatenations not saved as analysis variables. Optional plotting does extensive extra work only when requested. The extra paper-style session screening also processes metadata for sessions ultimately discarded.

ii.
```python
spikes, region_idx, anno, ccf_xyz = load_good_units(f, ontology)
...
'timing': {...}, 'mean_rate_hz': float(fr.mean()),
...
vals = np.concatenate([np.concatenate([o[k] for o in r['output']]) for r in results])
```

iii. These values support validation, logging, region assignment, and diagnostics, but `anno`, `ccf_xyz`, timing, rate summaries, and temporary concatenations are not part of the final decoder arrays.
