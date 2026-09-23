# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data/sub-*/*.nwb`, sorts the paths, and processes each NWB file as one session. Each worker opens the file with `pynwb.NWBHDF5IO`, then reads `nwb.units`, `nwb.trials`, `BehavioralEvents`, `BehavioralTimeSeries`, and `nwb.electrodes`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    trials = nwb.trials.to_dataframe()
    be = nwb.acquisition['BehavioralEvents']
```

```python
if args.jobs > 1 and len(files) > 1:
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.jobs) as pool:
        for i, res in enumerate(
                pool.imap(_worker, [(f, s, plot_dir) for f, s in zip(files, show_flags)])):
            results.append(res)
```

iii. In `CONVERSION_NOTES.md`, the AI says the released DANDI dataset is one NWB file per session under `sub-<subject_id>/`, so a sorted glob is the complete session list. It also explicitly justifies use of `pynwb` as required by the task.

## 1-b. How are the data split into subjects?

i. Subjects are taken per session from `nwb.subject.description` when present, otherwise `nwb.subject.subject_id`. After session processing, the AI builds a sorted unique `subjects` list and `subject_idx` from those strings.

ii. 
```python
subject = nwb.subject.description or nwb.subject.subject_id
```

```python
subjects = sorted({s['subject'] for s in sessions})
subj_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subj_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes state that `nwb.subject.description` contains the lab mouse ID such as `SC015`, while `subject_id` is the numeric DANDI subject id such as `440956`. The AI chose the lab ID for the output `subjects` field because it is the mouse identifier used in the papers.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. Session identity is `nwb.identifier`, and the session ordering in the output follows the sorted NWB path list.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
session_id = nwb.identifier
```

```python
'session_info': [s['info'] for s in sessions],
```

iii. The notes repeatedly describe the NWB layout as one file per session and say no further session grouping is needed.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials.to_dataframe()`. The AI checks that the number of `go_start_times` events equals the number of trial rows, then uses a Boolean `keep` mask to choose the subset of trial rows that survive curation.

ii. 
```python
trials = nwb.trials.to_dataframe()
n_trials_all = len(trials)
...
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == n_trials_all, \
    f"{session_id}: {len(go_all)} go cues for {n_trials_all} trials"
```

```python
keep = ephys_covered.copy()
keep &= (auto_all == 0) & (free_all == 0)
trial_idx = np.where(keep)[0]
```

iii. The notes say the trials table is the authoritative trial definition and that there is exactly one go cue per trial, making the mapping unambiguous.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps a trial only if it is covered by `units.obs_intervals`, has `auto_water == 0`, and has `free_water == 0`. After spike binning, it also drops trials whose full recorded population is all-zero in the decoder window, and it drops sessions with fewer than 2 usable trials.

ii. 
```python
obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)
...
ephys_covered = np.zeros(n_trials_all, dtype=bool)
j = np.clip(np.searchsorted(start_all, obs[:, 0] + 1e-6) - 1, 0, n_trials_all - 1)
matched = np.isclose(start_all[j], obs[:, 0]) & np.isclose(stop_all[j], obs[:, 1])
ephys_covered[j[matched]] = True
```

```python
keep = ephys_covered.copy()
keep &= (auto_all == 0) & (free_all == 0)
...
if n_trials < 2:
    return {'session_id': session_id, 'skipped': f'only {n_trials} usable trials'}
```

```python
has_spikes = rates.sum(axis=(0, 2)) > 0
...
if n_empty:
    rates = rates[:, has_spikes, :]
    trial_idx = trial_idx[has_spikes]
```

iii. In Decision D3 of the notes, the AI says it deliberately keeps photostim, early-lick, and ignore trials because those are decoder inputs/outputs, but excludes auto/free-water trials because reward is decoupled from behavior. The notes also justify all-zero-trial removal as a fix for trials where recording stopped mid-trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units['spike_times']` for units whose `classification` is `'good'`, together with `BehavioralEvents/go_start_times` to position the decoder window.

ii. 
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
```

```python
for i in good:
    st = np.asarray(units['spike_times'][int(i)], dtype=np.float64)
    ...
    spike_lists.append(st)
```

```python
go_all = np.asarray(be['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. The notes say the reference pipeline uses classifier-approved single units only, and that spike times are the canonical neural representation available in NWB.

## 2-b. How is the `neural` data processed?

i. The AI converts each good unit’s spike times into 50 ms non-overlapping firing-rate bins in Hz. It flattens all trial bin edges into one query array, uses `np.searchsorted` per unit, takes `np.diff` to get spike counts per bin, and divides by the bin width. No smoothing, normalization, or baseline subtraction is applied.

ii. 
```python
def bin_spike_rates(spike_times_list, go_times):
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), n_trials, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
        out[i] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

```python
rates = bin_spike_rates(spike_lists, go)
```

iii. The notes explicitly say this is meant to match the reference `sliding_histogram(..., rate=True)` logic, with the only required change being 50 ms bins instead of the paper’s original analysis binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units.classification == 'good'`. Sessions with zero such units are skipped entirely. It does not apply a separate firing-rate threshold.

ii. 
```python
classification = np.asarray(units['classification'][:])
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

```python
'neuron_curation': 'units.classification == "good" (region-specific logistic-regression QC '
                   'classifier of Chen, Liu et al. 2023).',
```

iii. The notes’ Decision D1 says this matches the reference classifier-based QC, while Decision D4 says the paper’s later 2 Hz cutoff is analysis-specific and should not be used as dataset curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset by adding a fixed vector of relative bin edges to each trial’s absolute go-cue time and binning spikes against those absolute edges.

ii. 
```python
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0
```

```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
```

iii. The notes say all NWB timestamps share a session-wide clock, so no separate cross-stream alignment or interpolation is needed beyond anchoring bins to the per-trial go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 80 bins of 50 ms each, spanning 2.5 s before to 1.5 s after the go cue. There is no further temporal rebinning or smoothing.

ii. 
```python
T_PRE = 2.5
T_POST = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((T_PRE + T_POST) / BIN_SIZE))
```

iii. The notes say this is a required deviation from the paper’s original analysis settings because the task instructions explicitly asked for 50 ms bins over `[-2.5, +1.5]` s.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` and the trial go-cue times. The AI takes the last sample onset at or before each trial’s go cue.

ii. 
```python
sample_t = np.asarray(be['sample_start_times'].timestamps[:], dtype=np.float64)
sample_t = np.sort(sample_t)
si = np.searchsorted(sample_t, go, side='right') - 1
tone_onset = np.where(si >= 0, sample_t[np.clip(si, 0, len(sample_t) - 1)], np.nan)
```

iii. In Decision D7, the notes say early licks replay the sample and delay epochs, so the final sample onset before go is the instructive tone onset for that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes `go_minus_tone = go - tone_onset`, then adds that offset to the go-aligned bin centers so each bin stores elapsed seconds since the instruction tone. If the chosen tone onset is missing, non-finite, or earlier than trial start, it falls back to the session’s median go-to-tone interval, or 1.85 s if needed.

ii. 
```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < start_all[trial_idx])
if np.any(bad_tone):
    fallback = np.nanmedian(go[~bad_tone] - tone_onset[~bad_tone]) if np.any(~bad_tone) else 1.85
    tone_onset[bad_tone] = go[bad_tone] - fallback
go_minus_tone = go - tone_onset
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```

iii. The notes justify the overall representation by arguing that a go-relative clock would be nearly identical across trials, while tone-to-go interval carries real trial context. The fallback behavior appears to be the AI’s robustness measure for anomalous files.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same 80 go-aligned bin centers as the neural firing rates, so each timepoint lines up exactly with the corresponding neural bin.

ii. 
```python
time_from_tone = (BIN_CENTERS_REL[None, :] + go_minus_tone[:, None]).astype(np.float32)
```

```python
input_trials = [np.stack([time_from_tone[t], photostim[t]]).astype(np.float32)
                for t in range(n_trials)]
```

iii. The notes say all streams use the same go-cue-relative bin grid, so alignment is built into the representation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, rather than from the trial-table `photostim_onset` and `photostim_duration` columns.

ii. 
```python
ps_start = np.asarray(be['photostim_start_times'].timestamps[:], dtype=np.float64)
ps_stop = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=np.float64)
order = np.argsort(ps_start)
ps_start, ps_stop = ps_start[order], ps_stop[order]
```

iii. The notes say these event streams are the raw session-clock photostim events and verify they match the expected late-delay 0.5 s photoinhibition pattern.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI turns photostimulation into a binary time series over the neural bins. It computes each bin center in absolute session time and sets the bin to `1.0` when that center lies in `[photostim_start, photostim_stop)`, otherwise `0.0`.

ii. 
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
if len(ps_start):
    abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]
    k = np.searchsorted(ps_start, abs_centers, side='right') - 1
    valid = k >= 0
    kk = np.clip(k, 0, len(ps_start) - 1)
    on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
    photostim[on] = 1.0
```

iii. The notes justify this as the required decoder input format: a time-varying on/off indicator, not a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim alignment is done by comparing absolute session-time photostim intervals against the absolute session-time centers of the go-aligned neural bins.

ii. 
```python
abs_centers = go[:, None] + BIN_CENTERS_REL[None, :]
...
on = valid & (abs_centers >= ps_start[kk]) & (abs_centers < ps_stop[kk])
```

iii. The notes say photostim events and go cues live on the same NWB time base, so no additional offset correction is necessary.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated raw column. It is reconstructed from `trials.outcome` and `trials.trial_instruction`.

ii. 
```python
outcome = outcome_all[trial_idx]
instr = instr_all[trial_idx]
```

```python
choice_str = np.where(outcome == 'ignore', 'no lick',
                      np.where(outcome == 'hit', instr, opposite))
```

iii. The notes say the AI cross-checked that reconstructing choice from outcome plus instructed side agrees with choice inferred from lick timestamps almost all the time.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps a hit to the instructed side, a miss to the opposite side, and an ignore to `'no lick'`. It codes those as `0=left`, `1=right`, `2=no lick`, then repeats the per-trial value across all 80 bins.

ii. 
```python
opposite = np.where(instr == 'left', 'right', 'left')
choice_str = np.where(outcome == 'ignore', 'no lick',
                      np.where(outcome == 'hit', instr, opposite))
choice = np.select([choice_str == 'left', choice_str == 'right'], [0, 1], default=2).astype(np.int64)
```

```python
np.full(N_BINS, choice[t], dtype=np.int64)
```

iii. The notes say this coding is consistent with the task’s categorical output requirements, and repeating across bins lets all outputs share the same `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from `trials.outcome`.

ii. 
```python
outcome = outcome_all[trial_idx]
```

iii. The notes state that the NWB trials table already stores the exact three outcome labels the task requires.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore'`, `'miss'`, and `'hit'` to `0`, `1`, and `2` respectively, then repeats the per-trial code across all bins.

ii. 
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss'], [0, 1], default=2).astype(np.int64)
```

```python
np.full(N_BINS, outcome_code[t], dtype=np.int64)
```

iii. The notes say this is a direct categorical encoding of the existing trial-table outcome.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from `trials.early_lick`.

ii. 
```python
early = early_all[trial_idx]
```

iii. The notes say the NWB trial table already stores this trial-level behavioral label explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'early'` to `1` and everything else to `0`, then repeats the trial-level code across all bins.

ii. 
```python
early_code = (early == 'early').astype(np.int64)
```

```python
np.full(N_BINS, early_code[t], dtype=np.int64)
```

iii. The notes say this follows the decoder spec of a binary categorical output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue position comes from `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps plus column 1 of the data array for y-position, with column 2 used as a visibility/likelihood score.

ii. 
```python
tongue = nwb.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']
ttimes = np.asarray(tongue.timestamps[:], dtype=np.float64)
tdata = np.asarray(tongue.data[:], dtype=np.float64)
```

```python
visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. The notes describe this as the side-view DeepLabCut tongue tracking stream present across the sessions.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first stable-sorts the tongue timestamps if they are not monotonic, then marks frames as visible when likelihood exceeds `0.9`. It uses `bin_visible_mean` to average visible tongue y values inside each go-aligned 50 ms bin, leaves bins with no visible frames as missing, and finally passes those means to `discretise_tongue`.

ii. 
```python
if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
    o = np.argsort(ttimes, kind='stable')
    ttimes, tdata = ttimes[o], tdata[o]
visible = tdata[:, 2] > TONGUE_LIKELIHOOD_THRESH
mean_y, vis_counts = bin_visible_mean(ttimes, tdata[:, 1], visible, go)
```

```python
def bin_visible_mean(times, values, visible, go_times):
    tv = times[visible]
    yv = values[visible]
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
    csum = np.concatenate([[0.0], np.cumsum(yv.astype(np.float64))])
    counts = np.diff(idx, axis=1)
    sums = np.diff(csum[idx], axis=1)
    ...
```

iii. In Decisions D8 and D9, the notes justify the `0.9` threshold by saying the likelihood distribution is strongly bimodal, and say the discretization should be based on visible binned y values from kept trials in that session.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes the 40th and 60th percentiles of the visible binned `mean_y` values for that session, then assigns class `0` below the 40th percentile, class `2` above the 60th percentile, class `1` in between, and class `3` when no visible frames fell in that bin.

ii. 
```python
def discretise_tongue(mean_y, counts):
    vis = counts > 0
    out = np.full(mean_y.shape, 3, dtype=np.int64)
    if not np.any(vis):
        return out, (np.nan, np.nan)
    p40, p60 = np.percentile(mean_y[vis], [TONGUE_PCT_LOW, TONGUE_PCT_HIGH])
    y = mean_y[vis]
    cls = np.where(y < p40, 0, np.where(y > p60, 2, 1))
    out[vis] = cls
    return out, (float(p40), float(p60))
```

iii. The notes say this matches the requested per-session 40th/60th percentile discretization and uses an explicit `"not visible"` class because the decoder task required that rather than imputation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue frames are aligned to the same go-cue-centered 50 ms bin grid as neural activity. `bin_visible_mean` converts each trial’s absolute go time to absolute bin edges and uses those edges to accumulate visible-frame means.

ii. 
```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
```

iii. The notes say video, events, and spikes all use the same session clock in NWB, so this shared binning grid is sufficient for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. Sessions with no good units are skipped. Non-monotonic spike times or camera timestamps are sorted. Missing or implausible tone onsets are imputed from the median session go-to-tone interval. Bins with no visible tongue frames become class `3`. Trials with zero spikes across the full simultaneously recorded population are dropped. Truncated post-trial windows are left as zero firing rate and `"not visible"` tongue bins.

ii. 
```python
if len(good) == 0:
    return {'session_id': session_id, 'skipped': 'no good units'}
```

```python
if st.size > 1 and not np.all(np.diff(st) >= 0):
    st = np.sort(st)
...
if ttimes.size > 1 and not np.all(np.diff(ttimes) >= 0):
    o = np.argsort(ttimes, kind='stable')
```

```python
bad_tone = ~np.isfinite(tone_onset) | (tone_onset < start_all[trial_idx])
if np.any(bad_tone):
    fallback = np.nanmedian(go[~bad_tone] - tone_onset[~bad_tone]) if np.any(~bad_tone) else 1.85
    tone_onset[bad_tone] = go[bad_tone] - fallback
```

```python
has_spikes = rates.sum(axis=(0, 2)) > 0
...
tongue_class, pcts = discretise_tongue(mean_y, vis_counts)
```

iii. The notes justify these as fixes for known quirks in the NWB release: partially recorded sessions, duplicated video timestamps in two sessions, partially missing video, and rare trials with no usable neural data.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB I/O, reading spike trains for all good units, spike binning, and final pickle writing as the main costs. It also records timing per session for trial handling, spike reading, spike binning, region assignment, inputs, outputs, and assembly.

ii. 
```python
timing = OrderedDict()
...
timing['trials'] = time.time() - t_start
...
timing['read_spikes'] = time.time() - t0
...
timing['bin_spikes'] = time.time() - t0
...
timing['outputs'] = time.time() - t0
...
'timing': {k: round(v, 3) for k, v in timing.items()},
```

```python
with open(args.outfile, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly say the runtime is dominated by opening NWB files, reading spike/video arrays, per-unit `searchsorted`, and writing the large pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI tried to vectorize nearly all trial/timepoint work. Neural binning is vectorized across all trials at once, and tongue averaging is vectorized with `searchsorted` plus cumulative sums. The main remaining unavoidable Python loop is over units, because spike times are ragged per unit; plotting code also loops over a handful of example trials when `--show-processing` is enabled.

ii. 
```python
for i, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    out[i] = np.diff(idx, axis=1)
```

```python
tv = times[visible]
yv = values[visible]
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
idx = np.searchsorted(tv, edges).reshape(n_trials, N_BINS + 1)
csum = np.concatenate([[0.0], np.cumsum(yv.astype(np.float64))])
```

iii. The notes say the major performance work was replacing naive per-trial/per-bin loops with vectorized `searchsorted`/cumsum logic and parallelizing across sessions.

## 10-c. What processing does the code repeat multiple times?

i. In the core conversion path, the AI mostly computes each quantity once per session and reuses module-level bin definitions. Optional `--show-processing` mode does additional plotting passes over the already computed arrays for a few example trials.

ii. 
```python
BIN_EDGES_REL = -T_PRE + np.arange(N_BINS + 1) * BIN_SIZE
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2.0
```

```python
if show_processing:
    _plot_processing(plot_dir, session_id, go, trial_idx, rates, time_from_tone,
                     photostim, choice, outcome_code, early_code, tongue_class,
                     mean_y, vis_counts, pcts, ttimes, tdata, visible,
                     ps_start, ps_stop, tone_onset, region_idx,
                     start_all[trial_idx], stop_all[trial_idx])
```

iii. The notes say the conversion is intended as a single pass over each NWB file, with per-session tongue percentiles computed in that same pass rather than through a second global scan.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core converter tries to avoid discarded processing, but it still computes optional diagnostic plots, rich per-session timing/info dictionaries, and summary statistics that are useful for validation rather than for downstream decoder training itself.

ii. 
```python
info = {
    'session_id': session_id,
    ...
    'timing': {k: round(v, 3) for k, v in timing.items()},
    'total_time': round(time.time() - t_start, 2),
}
```

```python
if show_processing:
    _plot_processing(...)
```

iii. The notes downplay discarded work and say every computed field is either preserved or used for validation. In practice, the extra timing and plotting work serves diagnosis rather than the final decoder inputs/outputs.
