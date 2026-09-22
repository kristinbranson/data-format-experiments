# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `/app/data/sub-*/*.nwb`, sorted the paths, and processed each file with `pynwb.NWBHDF5IO`; full runs used a 16-process pool. Each file supplied its trials, events, units, electrodes, and tracking streams.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. It justified this from the DANDI layout (one NWB file per session), the explicit `pynwb` requirement, and parallelized sessions because they are independent.

## 1-b. How are the data split into subjects?

i. Each session uses `nwb.subject.subject_id`; unique sorted IDs form `subjects`, and each session gets the corresponding integer `subject_idx`.

ii.
```python
subject_id = str(nwb.subject.subject_id)
subjects = sorted(set(r['subject_id'] for r in sessions))
'subject_idx': np.array([subject_index[r['subject_id']] for r in sessions])
```

iii. The notes identify this as the canonical NWB animal ID and report 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session; output order follows the sorted path list (and ordered `Pool.imap`). A session is omitted if it has no usable units or fewer than two trials.

ii.
```python
jobs = [(f, i < plot_n) for i, f in enumerate(files)]
for i, r in enumerate(pool.imap(_worker, jobs)):
    results.append(r)
sessions = [r for r in results if 'neural' in r]
```

iii. The file/session identity follows the dataset layout. The single session without QC-good units was dropped, yielding the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Rows of `nwb.intervals['trials']` define trials and are paired positionally with one `go_start_times` event per row. A retained index array is then applied consistently to trial variables.

ii.
```python
df = nwb.intervals['trials'].to_dataframe()
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
if len(go_times) != n_trials_all: raise RuntimeError(...)
trial_idx = np.where(keep)[0]
```

iii. The agent verified one go cue per trials-table row and used the table rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. It removes auto-water, free-water, invalid-go-boundary, incomplete-ephys-coverage, and all-unit-zero-spike trials; sessions with fewer than two survivors are removed. It keeps early-lick, ignore, and photostimulation trials.

ii.
```python
keep = (auto_water == 0) & (free_water == 0)
keep &= (go_times >= start_time) & (go_times <= stop_time)
for ui in unit_ids: ...; obs_trial &= m
keep &= obs_trial
has_spikes = spikes_in_trial > 0
```

iii. It invoked the paper's regular-trial mask for water trials, retained explicitly requested decoder classes, and treated absent ephys as missing rather than zero firing. This produced 89,544 trials versus the reference's 90,860 because the reference excludes free-water but not auto-water and uses one representative unit's observation intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from ragged `units.spike_times`; `units.classification`, `units.anno_name`, unit-electrode links, electrode `x`, observation intervals, trials, and go times determine inclusion, regions, and alignment.

ii.
```python
spike_index = units['spike_times']
classification = np.asarray([str(c) for c in units['classification'][:]])
go_times = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
```

iii. The notes state spike times are the available neural representation and connect `classification == 'good'` to the published QC classifier.

## 2-b. How is the `neural` data processed?

i. For each retained unit, `searchsorted` counts spikes between all go-aligned edges, and counts are divided by 0.05 s to obtain float32 Hz. Edges are clipped to each trial's start/stop; no smoothing or normalization is applied.

ii.
```python
flat_edges = edges_clipped.ravel()
pos = np.searchsorted(st, flat_edges).reshape(n_trials, NBINS + 1)
rates[k] = np.diff(pos, axis=1).astype(np.float32)
rates /= BIN_SIZE
```

iii. It followed the reference's count/bin-width firing-rate convention while replacing the paper's sliding bins with the mandated non-overlapping 50 ms bins. Clipping was justified as respecting trial-gated recording.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units require `classification == 'good'`, a recognized custom coarse CCF region, and finite electrode ML coordinate. Sessions with zero retained units are dropped. No firing-rate threshold is used.

ii.
```python
good = classification == 'good'
reg = coarse_region(anno[i])
if reg is None or not np.isfinite(ml[i]): continue
use_unit = good & (region_idx >= 0)
```

iii. The classifier was deemed the intended QC. The agent rejected the method paper's 2 Hz criterion as analysis-specific and added region validity to support reference-style hemisphere-resolved coarse regions; it reports no good units were lost to annotation mapping in practice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It adds the common relative edge grid `[-2.5, 1.5]` to each absolute go-cue timestamp and bins absolute spike timestamps against it (after trial-bound clipping).

ii.
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
edges_clipped = np.clip(edges, tstart[:, None], tstop[:, None])
```

iii. Spikes, trial events, and go cues share the NWB session clock, so no clock conversion or interpolation was considered necessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50 ms bins from -2.5 to +1.5 s. Raw spikes are newly binned at this resolution; there is no further temporal rebinning.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((T_STOP - T_START) / BIN_SIZE))
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. These values directly implement the decoder specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents.sample_start_times` and each retained trial's go cue/start time; the last sample onset before the go cue is selected.

ii.
```python
pos = np.searchsorted(sample_times, go, side='left') - 1
tone_onset = np.where(pos >= 0, sample_times[np.clip(pos, 0, len(sample_times)-1)], np.nan)
```

iii. Early licking can replay the sample epoch, so the last pre-go sample is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The go-minus-tone offset is added to every go-relative bin center. Missing or pre-trial tones are replaced by the standard `go - 1.85` timing.

ii.
```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < tstart - 1e-9)
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

iii. This expresses continuous seconds since tone at bin centers. The fallback was defensive; the full run reported zero uses.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 bin centers relative to the same per-trial go cue as neural bins.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = (go - tone_onset)[:, None] + BIN_CENTERS_REL[None, :]
```

iii. The shared go-relative grid guarantees index-wise alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `BehavioralEvents.photostim_start_times` and `photostim_stop_times`, assigning events to retained trials via trial start times.

ii.
```python
stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
which = np.searchsorted(tstart, stim_on, side='right') - 1
```

iii. The agent cross-checked these events with trials-table onset/duration fields and chose the direct timestamp stream.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary bin is one whenever its interval overlaps a stimulation interval; multiple events are ORed. Invalid/out-of-trial events are ignored.

ii.
```python
ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
photostim_b[w] |= ov
photostim = photostim_b.astype(np.float32)
```

iii. This represents whether light was on at any part of a time bin; it was chosen as a time-varying rather than per-trial indicator.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are compared to the same absolute go-aligned bin edges used for spikes.

ii.
```python
edges = go[:, None] + BIN_EDGES_REL[None, :]
ov = (edges[w, :-1] < s_off) & (edges[w, 1:] > s_on)
```

iii. All timestamps share the NWB clock, so direct interval comparison aligns the streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. It derives choice from `left_lick_times`, `right_lick_times`, go cue, and the earlier of trial stop or go+1.5 s.

ii.
```python
l = left_licks[(left_licks >= go[i]) & (left_licks <= answer_end[i])]
r = right_licks[(right_licks >= go[i]) & (right_licks <= answer_end[i])]
```

iii. The agent treated choice as the first lick in the defined answer period and reports cross-checking it against instruction × outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Default is 2 (no lick); only-left maps to 0, only-right to 1, and when both occur the earlier timestamp wins. The per-trial code is repeated over 80 bins.

ii.
```python
choice = np.full(n_trials, 2, dtype=np.int64)
choice[i] = 0 if l[0] < r[0] else 1
outputs[:, 0, :] = choice[:, None]
```

iii. This directly operationalizes left/right/no-lick; repeating it allows all outputs to share a `(4,80)` shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the retained rows of the trials-table `outcome` column.

ii.
```python
outcome_str = np.asarray(df['outcome'].values, dtype=object)
outcome = np.array([omap[str(o)] for o in outcome_str[trial_idx]])
```

iii. NWB already provides precisely the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map as ignore=0, miss=1, hit=2 and are repeated across time.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
outputs[:, 1, :] = outcome[:, None]
```

iii. The mapping matches the requested categorical order; repetition is a storage convenience for a per-trial output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It uses the trials-table `early_lick` strings for retained trial indices.

ii.
```python
early_str = np.asarray(df['early_lick'].values, dtype=object)
```

iii. The source explicitly labels this requested variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1 and every other expected value (`'no early'`) to 0; the value is repeated across bins.

ii.
```python
early = np.array([1 if str(e) == 'early' else 0 for e in early_str[trial_idx]])
outputs[:, 2, :] = early[:, None]
```

iii. This realizes the requested no/yes coding for a per-trial variable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (DeepLabCut likelihood) of `Camera0_side_TongueTracking`.

ii.
```python
ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
vts = np.asarray(ts_obj.timestamps[:], dtype=float)
tongue_y = np.asarray(ts_obj.data[:], dtype=float)[:, 1]
```

iii. The agent identified this as the dataset's side-camera tongue trace and used likelihood to distinguish visible from retracted/unreliable frames.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood above 0.9 are visible. Cumulative sums/counts yield the mean visible y in each (trial-clipped) 50 ms bin; bins without a visible frame remain NaN/not-visible.

ii.
```python
visible = vdata[:, 2] > LIKELIHOOD_THRESH
s_y = ysum[vidx[:, 1:]] - ysum[vidx[:, :-1]]
ybin = np.where(n_y > 0, s_y / np.maximum(n_y, 1), np.nan)
```

iii. The notes say likelihood is strongly bimodal, making 0.9 insensitive versus lower cutoffs, and averaging ~15 frames is less noisy than selecting one.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th/60th percentiles are computed per session over finite retained-trial bin means. Values below p40, from p40 through p60, and above p60 map to 0/1/2; missing bins map to 3. Fewer than 10 visible values makes every bin class 3.

ii.
```python
p40, p60 = np.percentile(vis_vals, [40, 60])
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)
tongue_cls[fin & (ybin < p40)] = 0
tongue_cls[fin & (ybin >= p40) & (ybin <= p60)] = 1
tongue_cls[fin & (ybin > p60)] = 2
```

iii. The agent interpreted “over the session” as the distribution of the same binned visible quantity ultimately classified and validated a 40/20/40 visible-bin split.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are searched at the same absolute go-aligned edges (clipped to trial boundaries), producing exactly one tongue class per neural bin.

ii.
```python
vidx = np.searchsorted(vts, edges_clipped)
outputs[:, 3, :] = tongue_cls
```

iii. Video, spikes, and events use the same session clock; shared edges provide direct alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Structural mismatches raise errors; un-QC'd/no-unit sessions and sessions with fewer than two trials are skipped; invalid regions/coordinates remove units; absent ephys and zero-spike trials are removed; a missing tone has a 1.85 s fallback; missing tongue frames become category 3; worker exceptions are logged and excluded.

ii.
```python
if len(go_times) != n_trials_all: raise RuntimeError(...)
if n_units == 0: return None
tone_onset = np.where(bad_tone, go - 1.85, tone_onset)
tongue_cls = np.full((n_trials, NBINS), 3, dtype=np.int64)
```

iii. The agent distinguished absence of recording (exclude) from meaningful lack of visibility (explicit class), added defensive checks, and documented two trailing zero-spike trials and zero actual tone fallbacks.

## 10-a. What are the most time-consuming steps of the code?

i. NWB reading, ragged per-unit spike binning, video loading/binning, and final serialization are the main costs. The full parallel conversion took 23.3 s and writing the 11.89 GB pickle another 22.1 s.

ii.
```python
timing['neural'] = time.time() - t1
timing['input'] = time.time() - t1
timing['output'] = time.time() - t1
with Pool(min(args.nproc, len(files))) as pool:
```

iii. The notes expected I/O and spike searches to dominate and used per-stage timings plus multiprocessing to verify acceptable runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit loop remains because spike arrays are ragged, though all trials/bins are vectorized inside it. The choice loop and photostimulation-event loop could be further vectorized; observation-interval intersection also loops over every unit. Region mapping loops over good units because it applies string rules.

ii.
```python
for ui in unit_ids: ... obs_trial &= m
for k, ui in enumerate(unit_ids): ... np.searchsorted(st, flat_edges)
for s_on, s_off, w in zip(stim_on, stim_off, which): ...
for i in range(n_trials): ...
```

iii. The agent explicitly described the neural ragged loop as unavoidable and considered remaining loops small; cumulative-sum video binning removed the expensive frame/bin loops.

## 10-c. What processing does the code repeat multiple times?

i. Per-unit reading/searching repeats for neural binning and again for whole-trial spike totals; observation intervals are independently read for every retained unit. Plot mode rereads unit spike arrays and recomputes display summaries. Otherwise each session is opened once and shared grids are module constants.

ii.
```python
pos = np.searchsorted(st, flat_edges)
tb = np.searchsorted(st, trial_bounds)
for ui in aux['unit_ids'][:nshow]:
    st = np.asarray(aux['spike_index'][int(ui)], dtype=float) - g
```

iii. The agent claimed core quantities were computed once; the extra whole-trial spike pass was intentionally added to detect stopped recordings, and plotting is optional validation work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes mouse descriptions, detailed timing/coverage/count diagnostics, absolute go times, original trial indices, file paths, tongue percentiles, and optional plots; most enter intermediate results/metadata or logs but are not decoder features. It also computes `edges_clipped`/`observed` largely for missing-coverage diagnostics and plots.

ii.
```python
mouse_name = str(nwb.subject.description)
'timing': timing, 'frac_bins_observed': float(np.mean(observed)),
'trial_index': trial_idx, 'go_times': go, 'tongue_pct': (float(p40), float(p60))
```

iii. The agent regarded provenance and validation diagnostics as useful rather than waste, and optional processing plots were specifically requested by the workflow; none are consumed by decoder training.
