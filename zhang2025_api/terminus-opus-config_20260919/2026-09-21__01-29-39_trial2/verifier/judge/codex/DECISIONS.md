# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the Brain-Wide Map release CSV to obtain 459 session IDs, then processes each session through a remote-mode `ONE` client. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads every probe returned by `eid2pid`. Sessions are processed in a 12-worker pool (or sequentially for samples/plots). It does not read ALF data files directly, but the session roster itself comes from the CSV rather than a ONE dataset-availability search.

ii.
```python
bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(bwm.eid.unique())
...
one = get_one()
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
...
pids, pnames = one.eid2pid(eid)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The notes say remote-mode ONE was necessary to resolve revised datasets from cached REST responses while remaining offline. The release CSV was treated as the authoritative BWM session list, and multiprocessing was used because sessions are independent.

## 1-b. How are the data split into subjects?

i. Subjects are read from `bwm_release.csv`, associated with each session by `eid`, deduplicated and sorted, and represented by a per-session integer `subject_idx`.

ii.
```python
eid2subject = bwm.groupby('eid').subject.first().to_dict()
subjects = sorted({eid2subject[r['eid']] for r in ok})
subj_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subj_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64)
```

iii. The notes report 139 subjects in the release and state that the CSV supplies the subject mapping; no subject identity is inferred from paths.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV is one session. `convert_session` returns one session record, successful records are sorted by `eid`, and the outer lists in the target dictionary follow that order.

ii.
```python
eids = list(bwm.eid.unique())
...
ok.sort(key=lambda r: r['eid'])
'neural': [r['neural'] for r in ok]
```

iii. The agent regarded the BWM release and ONE `eid` as already defining session boundaries.

## 1-d. How are the data split into trials?

i. The trials table already has one row per trial. Retained row indices and `stimOn_times` define the trials; neural and behavioral streams are cut into the fixed window from 0.5 s before through 1.5 s after each onset.

ii.
```python
align_all = trials[ALIGN_TIME].to_numpy()
idx_trials = np.where(mask)[0]
align = align_all[idx_trials]
...
for k in range(n_trials):
    neural.append(binned_spikes[k])
```

iii. The notes identify the reference trial definition as alignment to `stimOn_times` with a two-second window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded for reaction time outside 0.08–2 s, duration from go cue to feedback over 10 s, missing values in six specified columns, or no choice. The agent additionally requires the window to fall inside every probe's spike-time coverage, requires full non-NaN wheel and camera coverage, and drops trials with zero spikes across all retained neurons. Sessions must retain at least two trials.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {min_rt})'
query += f' | (firstMovement_times - stimOn_times > {max_rt})'
query += f' | (feedback_times - goCue_times > {max_trial_len})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
...
mask = mask & covered
good = wheel_valid & me_valid
nonzero = binned_spikes.sum(axis=(1, 2)) > 0
```

iii. The primary mask is described as a copy of reference `load_trials_and_mask`. Coverage filters prevent incomplete aligned streams; zero-spike trials were removed because the agent considered them degenerate for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each probe's `spikes.times` and `spikes.clusters`. Cluster `label` and `acronym` (after merging cluster/channel information) determine which neurons survive and their regions.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
spike_times_l.append(spikes['times'])
spike_clu_l.append(spikes['clusters'] + offset)
```

iii. The notes identify these as the reference spike-sorting variables and use cluster metadata only for curation and anatomy.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting cluster IDs, spikes are time-sorted, retained cluster IDs are made contiguous, and spikes are counted in 100 non-overlapping 20 ms bins per trial. The saved values are float32 spike counts, not smoothed, normalized, or divided by bin width.

ii.
```python
spike_clu_l.append(spikes['clusters'] + offset)
...
b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
idx = spike_clusters[s:e].astype(np.int64) * n_bins + b
cnt = np.bincount(idx, minlength=n_clusters * n_bins)
out[k] = cnt.reshape(n_clusters, n_bins)
```

iii. The agent says this mirrors `bin_spiking_data`/`bincount2D`, deliberately leaves counts unnormalized, and relies on decoder-side PCA rather than z-scoring.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with QC label at least 1 and Beryl region other than `void` or `root` are kept. Sessions with fewer than five such neurons are skipped, and trials with no spikes from the retained population are dropped.

ii.
```python
keep = (labels >= QC_LABEL) & (~np.isin(beryl, EXCLUDE_REGIONS))
keep_ids = np.where(keep)[0]
if len(keep_ids) < MIN_NEURONS:
    return dict(eid=eid, skipped='too few good neurons')
```

iii. The notes justify `label == 1` from the data paper's well-isolated-unit criteria and exclusion of `void/root` as a grey-matter restriction. The five-neuron cutoff is attributed to BWM analysis criteria, though it is stricter than the target format requires.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `trials.stimOn_times`; bins span `[onset-0.5, onset+1.5)` on the shared session clock.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
begs = align_times + t_start
ends = align_times + t_end
```

iii. This exactly follows the decoding parameters in the reference repository and the instruction to align on stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins in two seconds. Spikes are directly counted in those bins; there is no later temporal rebinning.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent cites the reference parameters of a two-second interval and 20 ms bin size.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed alignment window and bin size relative to each trial's `stimOn_times`, rather than read as a separate raw column.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. The notes describe time since stimulus onset as a required decoder input and choose bin-center times from -0.49 to 1.49 s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The 100 bin centers are calculated once and copied as the first input row for every trial.

ii.
```python
inputs.append(np.stack([bin_centres.astype(np.float32),
                        np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
```

iii. The fixed relative-time grid directly expresses the requested continuous, time-varying input.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each value labels the center of the corresponding neural spike-count bin, so the first neural bin `[-0.5,-0.48)` is labeled -0.49 s and the last is labeled 1.49 s.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. The agent explicitly chose bin centers as the appropriate timestamp for binned neural observations.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw trial sequence's `probabilityLeft`; any change in that value marks a new block.

ii.
```python
tnb_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The trials table has no explicit block-number field, while `probabilityLeft` is constant within blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter resets on each probability change and increments otherwise. It is computed before trial filtering, selected using retained raw indices, and broadcast across all 100 time bins.

ii.
```python
newblock[1:] = pl[1:] != pl[:-1]
for i in range(len(pl)):
    c = 0 if newblock[i] else c + 1
    idx[i] = c
```

iii. Computing it on raw trials preserves the animal's true block position when intervening low-quality trials are removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`, where +1 denotes left and -1 denotes right.

ii.
```python
choice_raw = trials['choice'].to_numpy()[idx_trials]
```

iii. The agent verified the IBL convention against correct left- and right-stimulus trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The sign is recoded to left = 0 and right = 1, then the per-trial class is broadcast over all time bins. No-choice trials were already filtered.

ii.
```python
choice_lab = (choice_raw < 0).astype(np.int64)
np.full(NBINS, choice_lab[k], dtype=np.int64)
```

iii. This implements the target's specified binary encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft` for each retained trial.

ii.
```python
pleft_raw = trials['probabilityLeft'].to_numpy()[idx_trials]
```

iii. The notes confirm that the raw column contains exactly 0.2, 0.5, and 0.8.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 and broadcast over the trial. Unexpected values receive -1 and are removed.

ii.
```python
prior_lab = np.select([pleft_raw == 0.2, pleft_raw == 0.5, pleft_raw == 0.8],
                      [0, 1, 2], default=-1).astype(np.int64)
```

iii. This is the exact mapping required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is the absolute value of the wheel velocity produced by `SessionLoader.load_wheel()` from raw wheel position and timestamps.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].to_numpy()
wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. The notes identify `abs(velocity)` as the reference definition of wheel speed; the loader performs upstream interpolation/filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute velocity is linearly interpolated for every trial at 100 points from 20 ms after the window start through the window end. Trials with incomplete or NaN signal coverage are removed. The retained session-wide samples are then discretized by tertiles.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
out[k] = np.interp(grid[k], tt, vv)
wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
```

iii. Right-edge interpolation is said to copy reference `get_behavior_per_interval`; per-session tertiles were chosen to handle scale differences and balance the three decoder classes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Thresholds are the 1/3 and 2/3 quantiles over all retained wheel samples in a session. Values at or below the first threshold are 0, above the first are 1, and above the second are 2.

ii.
```python
q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
lab[x > q1] = 1
lab[x > q2] = 2
```

iii. The agent chose session tertiles because the task requests three bins, session scales vary, and balanced classes make decoding accuracy interpretable.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Both use the same stimulus onset and 100-element trial window, but wheel values are sampled at bin right edges (-0.48 through 1.50 s) whereas neural bins are represented by center-time inputs (-0.49 through 1.49 s).

ii.
```python
begs = align_times + t_start
grid = np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None]
```

iii. The agent says this is the reference behavior interpolation grid and treats each right-edge value as corresponding to its neural bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the `whiskerMotionEnergy` trace and timestamps loaded from the left camera when available, falling back to the right camera.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sl.load_motion_energy(views=[view])
    me_times = df['times'].to_numpy()
    me_vals = df['whiskerMotionEnergy'].to_numpy()
```

iii. The reference prefers the left camera, while fallback retains sessions that only have right-camera motion energy.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is not filtered or normalized. It is linearly interpolated at the same right-edge grid as wheel speed, invalid-coverage trials are removed, and samples are discretized using session-wide tertiles.

ii.
```python
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
me_lab, me_q = discretize_tertiles(me_binned, allrows)
```

iii. The agent states that the released motion-energy signal needs no additional processing and uses tertiles to accommodate arbitrary camera-dependent scales.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 1/3 and 2/3 quantiles over all retained session samples define low (0), medium (1), and high (2).

ii.
```python
ref = x[valid_rows].ravel()
q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
```

iii. Session tertiles were selected for balanced classes and robustness to camera gain, lighting, and ROI-scale differences.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares the stimulus onset and number of bins, but is sampled at neural-bin right edges rather than the bin centers used by the time input.

ii.
```python
grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
out[k] = np.interp(grid[k], tt, vv)
```

iii. The agent considers the reference right-edge behavioral samples aligned one-for-one with the neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Trial-table NaNs trigger trial exclusion. Behavioral windows with no samples, inadequate edge coverage, or internal NaNs are invalidated. Missing camera views trigger a left-to-right fallback; sessions with no camera, too few trials, too few neurons, or exceptions are skipped. Unknown prior values are removed. Worker exceptions are caught and reported rather than aborting the full run.

ii.
```python
if np.any(np.isnan(vv)):
    valid[k] = False
...
except Exception as e:
    traceback.print_exc()
    return dict(eid=eid, skipped=f'exception: {e}')
```

iii. The notes emphasize retaining only complete, decoder-safe arrays and document the camera fallback and minimum-session requirements.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting for all probes dominates per-session runtime; full-release conversion is also driven by per-session I/O and multiprocessing overhead. Optional diagnostic plotting can be costly but is not used in the full run.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
timing['spikes'] = time.time() - t0
```

iii. The notes report roughly 3–8 seconds per session with spike sorting dominant, while binning takes only hundredths of a second.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `bin_spikes` and `bin_behavior` loop over trials, `trial_number_in_block` loops over trials, output construction loops over trials, and probe loading loops over insertions. The first three could be further vectorized, though probe I/O inherently occurs per probe.

ii.
```python
for k in range(n_trials):
    ...
for i in range(len(pl)):
    c = 0 if newblock[i] else c + 1
for k in range(n_trials):
    neural.append(binned_spikes[k])
```

iii. The agent notes that it already replaced the reference's per-trial process pools with `searchsorted`, `np.interp`, and `bincount`; remaining loops were considered cheap and clearer.

## 10-c. What processing does the code repeat multiple times?

i. Every session constructs a new ONE client and `BrainRegions`, loads/merges each probe, performs several successive trial masks/slices, and separately interpolates wheel and whisker traces through the same `bin_behavior` logic. It also repeatedly builds constant time, choice, prior, and trial-number arrays for each trial.

ii.
```python
one = get_one()
br = BrainRegions()
...
wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
me_binned, me_valid = bin_behavior(me_times, me_vals, align)
```

iii. This repetition follows the per-session worker design and shared helper functions; the notes prioritize independent parallel sessions and correctness over eliminating small allocations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_spike_sorting()` uses its default attributes and hash behavior even though conversion ultimately needs only spike times/clusters plus cluster/channel metadata. The code computes and retains continuous binned behavior until class labels are formed, then discards it from the pickle; it also tracks timings and raw trial indices only for diagnostics. With `--show-processing`, it produces extensive plots that are not decoder inputs.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
...
wheel_binned, wheel_valid = bin_behavior(...)
me_binned, me_valid = bin_behavior(...)
...
timing=timing,
trial_idx=idx_trials,
```

iii. Continuous traces are necessarily intermediate inputs to discretization, while timing/index metadata and plots support validation. The notes explicitly identify raw AP sampling-rate loading in the original reference as unnecessary and avoid that step, but do not optimize the spike loader down to only the two spike arrays.
