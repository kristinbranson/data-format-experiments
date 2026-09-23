# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the Zhang BWM freeze CSV to enumerate 459 session IDs and probe rows, then uses a per-process `ONE` client and brainbox `SpikeSortingLoader`/`SessionLoader` to load spike sorting, trials, wheel, and camera motion energy. Sessions are processed in parallel; failures are skipped.

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))
_ONE = ONE(base_url=ONE_BASE_URL, silent=True, tables_dir=ONE_TABLES_DIR)
sp, cl, ch = ssl.load_spike_sorting()
sess_loader = SessionLoader(one=one, eid=eid)
```

iii. The notes say raw data are never opened directly, the freeze is the paper's session set, and ONE/brainbox preserve dataset/revision resolution. The agent omitted a raw-ephys sampling-rate query because it was irrelevant and could require remote streaming.

## 1-b. How are the data split into subjects?

i. Subject names come from the freeze CSV. Converted sessions retain `subject`; assembly makes a sorted unique subject list and maps every session into it.

ii.
```python
tasks.append((eid, rows[['pid', 'probe_name']].copy(),
              rows.subject.iloc[0], rows.lab.iloc[0], i < n_show))
subjects = sorted({r['subject'] for r in ok})
subject_idx = np.array([subject_index[r['subject']] for r in ok])
```

iii. The agent treats the freeze's subject field as the authoritative unique mouse ID.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the freeze is one session. Probe rows sharing an `eid` are grouped and processed together; successful results are restored to freeze order.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
rows = bwm[bwm.eid == eid]
ok.sort(key=lambda r: order[r['eid']])
```

iii. The agent notes that `eid` is already the session identifier and that probes in one session share behavior and should be merged.

## 1-d. How are the data split into trials?

i. `load_trials_and_mask` returns a trials table with one row per trial. Retained row indices define 2-second windows around each row's `stimOn_times`; per-trial arrays are emitted from those indices.

ii.
```python
trials, mask = load_trials_and_mask(...)
cand = np.nonzero(trials_mask)[0]
interval_begs = stim_on_all[cand] - 0.5
```

iii. The notes regard the trials table as the native trial split and retain original `trial_idx` for auditability.

## 1-e. How are trials filtered based on quality controls?

i. The reference helper removes missing key fields, reaction times outside 0.08–2 s, trials longer than 10 s, and no-choice trials. The agent additionally requires complete, finite wheel and camera windows, at least two surviving trials per session, and removes trials with no spikes across the retained population.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
keep_trial &= beh_good[name]
has_spikes = binned.any(axis=(1, 2))
```

iii. The agent says the helper reproduces paper/reference curation; behavior NaNs are dropped because categorical imputation would invent labels. It interpreted an all-zero population window as recording dropout.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices derive from every probe's `spikes.times` and `spikes.clusters`; cluster metrics/labels and anatomical acronyms determine which units survive and their brain-region indices.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
```

iii. The agent follows the reference spike-sorting loaders and omits raw AP data because spike times and assignments are sufficient.

## 2-b. How is the `neural` data processed?

i. Probes are merged, retained clusters remapped contiguously, and spikes counted in 100 non-overlapping 20-ms bins from −0.5 to +1.5 s. The saved float32 values are raw counts, not rates or standardized activity.

ii.
```python
flat = spike_clusters[i0:i1].astype(np.int64) * n_bins + bin_idx
counts = np.bincount(flat, minlength=n_neurons * n_bins)
out[k] = counts.reshape(n_neurons, n_bins)
```

iii. The notes argue that the released cache stores counts and standardizes only during model fitting, so conversion should not divide or z-score.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains clusters with `label >= 1`, removes Beryl `root` and `void`, and drops sessions with fewer than five remaining neurons.

ii.
```python
good = clusters['label'].to_numpy() >= 1
in_brain = ~np.isin(beryl_all, ('root', 'void'))
keep = good & in_brain
if n_neurons < MIN_NEURONS_PER_SESSION: ...
```

iii. The agent prioritizes the data paper's 75,708 well-isolated neurons and grey-matter restriction over the method cache's all-cluster default; five neurons is borrowed from a regional analysis criterion and applied session-wide.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins at `stimOn_times - 0.5`; spikes are sliced on the shared session clock and binned through `stimOn_times + 1.5`.

ii.
```python
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
i0s = np.searchsorted(spike_times, interval_begs, side='left')
```

iii. The agent says stimulus onset is mandated by the decoder task and used by the released caching code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: 100 bins cover the 2-second window. Spike events are histogrammed directly; there is no later temporal rebinning or smoothing.

ii.
```python
'binsize': 0.02,
N_BINS = int(np.ceil(2.0 / 0.02))
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. The agent cites the released reference configuration and the results description of 2-second trials with 20-ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from each trial's `stimOn_times` conceptually, but numerically is a fixed relative grid at the right edges of the neural bins, −0.48 through +1.50 s.

ii.
```python
stim_on_all = trials['stimOn_times'].to_numpy()
bin_times = -0.5 + 0.02 * np.arange(1, N_BINS + 1)
```

iii. The agent chose the reference behavior interpolation grid and describes it as time relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed arithmetic progression of bin right edges is generated and copied to every trial.

ii.
```python
inputs[:, 0, :] = bin_times[None, :]
```

iii. The notes justify right edges as matching `get_behavior_per_interval`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Input sample k labels the right edge of neural spike bin k, while that neural value counts the preceding half-open 20-ms interval.

ii.
```python
# right edge of each spike bin, relative to stimulus onset
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
```

iii. The agent considered this consistent with the released continuous-behavior grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the unfiltered trials table's `probabilityLeft`; every value change starts a new block.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Since no explicit block ID exists, the agent reconstructs blocks from the probability held constant within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code assigns a zero-based position relative to the first row of each block, computes it before filtering, selects retained rows, and broadcasts each scalar over all 100 bins.

ii.
```python
changed[1:] = ~same
block_id = np.cumsum(changed) - 1
return np.arange(len(p)) - first_of_block[block_id]
inputs[:, 1, :] = tib[:, None]
```

iii. Computing before filtering preserves the mouse's true experimental trial position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice` for retained trials.

ii.
```python
choice = trials['choice'].to_numpy()[trial_idx]
```

iii. The agent empirically checked the IBL sign convention using high-contrast stimulus sides.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` is mapped to 0 (left), `-1` to 1 (right), and the label is broadcast across time. No-response choices were already filtered.

ii.
```python
choice_lbl = ((1 - choice) / 2).astype(np.int64)
outputs[:, 0, :] = choice_lbl[:, None]
```

iii. This implements the requested labels and the verified IBL convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[trial_idx]
```

iii. The field is the task's blockwise left-stimulus prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 and broadcast across time; unexpected values make the session fail.

ii.
```python
for val, lbl in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_lbl[np.isclose(pleft, val)] = lbl
outputs[:, 1, :] = prior_lbl[:, None]
```

iii. This mapping is specified directly by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()` supplies wheel timestamps and filtered velocity derived from wheel position; speed is the absolute velocity.

ii.
```python
sess_loader.load_wheel()
'times': sess_loader.wheel['times'].to_numpy(),
'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
```

iii. The agent follows `load_target_behavior('wheel-speed')` and the reference definition of speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The loader interpolates/smooths wheel position to obtain velocity. The conversion takes its magnitude, checks window coverage/NaNs, linearly interpolates to neural-bin right edges, then discretizes pooled retained session values.

ii.
```python
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs, 0.02, 100)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. The notes say this mirrors the reference helper, except finite categorical targets motivate dropping rather than imputing NaNs.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles pooled over all retained timepoints in each session; `np.digitize` produces low/medium/high labels 0/1/2.

ii.
```python
q1, q2 = np.quantile(values.reshape(-1), TERTILES)
labels = np.digitize(values, [q1, q2], right=False)
```

iii. Per-session tertiles balance classes and avoid comparing session-specific measurement scales.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are interpolated at the right edge of every 20-ms spike-count bin using the same stimulus-relative 2-second trial window.

ii.
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The agent follows the released behavior interpolation helper's grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses camera `whiskerMotionEnergy` and timestamps loaded by `SessionLoader`, preferring `leftCamera` and falling back to `rightCamera`.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sess_loader.load_motion_energy(views=[view])
    df = sess_loader.motion_energy[cam]
```

iii. The left source matches the method code; fallback retains sessions lacking left-camera data.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI signal is used without filtering/normalization, checked for coverage and NaNs, interpolated linearly to right edges, and discretized with session-level tertiles.

ii.
```python
'values': df['whiskerMotionEnergy'].to_numpy()
me_lbl, me_thr = discretize_tertiles(me)
```

iii. The agent says arbitrary camera/ROI units favor within-session thresholds and no extra transformation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel speed, pooled retained values within each session are cut at the 33.3rd and 66.7th percentiles into 0/1/2.

ii.
```python
me_lbl, me_thr = discretize_tertiles(me)
outputs[:, 3, :] = me_lbl
```

iii. Tertiles provide the required three approximately balanced categorical classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at each spike bin's right edge on the same stimulus-onset window.

ii.
```python
idxs_beg = np.searchsorted(times, interval_begs, side='right')
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The agent relies on synchronized IBL session clocks and the reference behavior grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing camera sessions, too-few-neuron sessions, sessions with fewer than two valid trials, unexpected priors, and other exceptions are skipped and recorded. Trials with missing key fields, incomplete/NaN behavior, or zero population spikes are removed. Right camera is a fallback; degenerate tertiles use unique available edges.

ii.
```python
except Exception as e:
    return {'eid': eid, 'error': f'{type(e).__name__}: {e}'}
if np.any(np.isnan(v)): continue
edges = np.unique([q1, q2])
```

iii. The agent prioritizes finite labels, robust completion of the full run, and explicit `skipped_sessions` metadata over imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting dominates (3890 worker-seconds, about 8.82 s/session), followed by behavior and trial loading; binning costs little. Pickling the 11.7-GB result also takes about 18 s.

ii.
```python
timings['load_spikes'] = time.time() - t0
sp, cl, ch = ssl.load_spike_sorting()
```

iii. The full-run timing report supports the agent's conclusion that disk I/O, not numerical binning, is the bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial loops remain in `bin_spikes` and `interpolate_behavior`; the prior-label loop has only three iterations. The session loop is parallelized. Searchsorted and bincount vectorize most work inside each trial.

ii.
```python
for k in range(n_intervals):
    ...
    counts = np.bincount(flat, minlength=n_neurons * n_bins)
for k in range(n):
    vals[k] = interp1d(...)(x)
```

iii. The notes report only ~0.02 and ~0.06 s/session for spike and behavior binning, so further vectorization would not materially improve total runtime.

## 10-c. What processing does the code repeat multiple times?

i. Every session independently creates a ONE singleton per worker, loads each probe, performs similar per-trial coverage/interpolation loops twice (wheel and whisker), and computes/retains overlapping masks. Diagnostic mode also recomputes summary rates for plotting.

ii.
```python
for name, d in beh.items():
    v, g, r = interpolate_behavior(...)
for _, row in probe_rows.iterrows():
    ssl.load_spike_sorting()
```

iii. The agent accepts repetition because streams differ in sampling and sessions are independent; process-local clients avoid unsafe shared connections.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges broader spike and cluster structures before selecting only times, assignments, labels, and acronyms; computes `beh_reasons`, detailed timings, thresholds, lab, and extensive session audit metadata that the decoder does not consume. Optional figures calculate PSTHs and plots solely for validation.

ii.
```python
beh_vals, beh_good, beh_reasons = {}, {}, {}
result = {..., 'timings': timings, 'total_time': ...}
if show_processing: make_processing_figure(...)
```

iii. These discarded intermediates were intentionally retained during conversion for diagnostics, provenance, sanity checks, and error reporting rather than decoder features.
