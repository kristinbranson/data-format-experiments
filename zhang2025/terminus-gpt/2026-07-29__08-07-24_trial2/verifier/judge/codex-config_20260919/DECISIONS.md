# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively searches `data/one_cache` for every trials parquet file, treats its ancestor above `alf` as a session, and then directly loads trials, wheel, camera, spike, cluster, and anatomy files with pandas/NumPy. It neither uses ONE's release index nor restricts a datalimit cache using `DATALIMIT_SUBSET.csv`. The full run found and kept 461 sessions.

ii.
```python
trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
...
return pd.read_parquet(trial_file)
...
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
```

iii. The notes identify the data as an IBL ONE-cache directory tree and justify direct loading from its session-level ALF arrays. They say sessions are the appropriate top-level unit and probes should be merged within a session.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the session directory path. Unique subject strings are sorted, and each session receives an index into that list.

ii.
```python
subject = session_path.parts[-3]
subjects = sorted({p['subject'] for p in processed})
subject_idx = np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64)
```

iii. The AI relied on the documented cache layout `<lab>/Subjects/<subject>/<date>/<session>/alf` and reported 141 subjects.

## 1-c. How are the data split into sessions?

i. Each unique directory containing an `alf` folder and a trial table is one session; each is processed independently and appended in sorted path order.

ii.
```python
sessions.append(sess.parent)
out = sorted(set(sessions))
for i, sess in enumerate(sessions, 1):
    p = process_session(sess, show_processing=args.show_processing)
```

iii. The notes state that the raw cache and reference analyses are session-organized, so session is used as the converted dataset's top-level unit.

## 1-d. How are the data split into trials?

i. Each retained row of the trial table is one trial. Its `stimOn_times` value defines a window and produces one neural, input, and output matrix.

ii.
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
    ...
    trial_mats.append(mat)
```

iii. The AI notes that `stimOn_times` provides direct trial alignment and that every converted trial should become a neuron-by-time matrix.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have nonmissing stimulus onset, choice in `{-1, 1}`, and nonmissing prior. After binning, trials with no spikes from any retained neuron are also removed; sessions with fewer than two remaining trials are dropped. No 80 ms–2 s reaction-time mask or wheel/camera window-coverage mask is applied.

ii.
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. The notes say no-response trials were excluded to ensure valid binary choice and all-zero neural trials were removed to eliminate verifier warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from per-probe `spikes.times.npy` and `spikes.clusters.npy`. Cluster metrics determine retained units, while cluster-channel and channel brain-location arrays supply region labels.

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
metrics = pd.read_parquet(cand[-1])
```

iii. The notes identify spike-sorted units and binned spikes as the electrophysiology representation, with probes merged within a session.

## 2-b. How is the `neural` data processed?

i. Retained clusters are remapped to consecutive session-wide indices across probes. For every trial, spikes in −0.2 to 1.0 s relative to stimulus onset are counted into 20 ms bins. The stored values remain spike counts; they are not divided by bin width to obtain Hz.

ii.
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
...
np.add.at(mat, (clu[good], tb[good]), 1)
```

iii. The notes explicitly chose binned spike counts, saying the papers/reference decode from binned spikes, and chose to merge probes within each session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cluster is retained when `noise_cutoff < 20` if that column exists and `label >= 1` if that column exists. It is not filtered for Beryl `void`, and region names remain raw CCF IDs. Spike membership is tested against retained cluster row indices.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
```

iii. The AI sought to reproduce the paper's well-isolated population and cited amplitude, noise-cutoff, and refractory-period criteria; in code it used the available aggregate `label` plus `noise_cutoff`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is shifted by that trial's `stimOn_times`, and spikes in the fixed relative window are binned, so zero denotes stimulus onset.

ii.
```python
rel = spike_times - s
mask = (rel >= t0) & (rel < t1)
```

iii. The notes call stimulus alignment an intentional task-specific requirement even where paper analyses used other events.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, yielding 60 bins over −0.2 to 1.0 s. Raw spikes are counted directly in these bins; no later neural rebinning or smoothing occurs.

ii.
```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

iii. The notes cite the paper's 20 ms nonoverlapping bins, but do not justify changing the reference conversion's two-second −0.5 to 1.5 s window.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the `stimOn_times` alignment event and the generated neural-bin edges rather than from a separate sampled raw variable.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. The notes say `stimOn_times` supports the required alignment and that relative bin times form the input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Adjacent bin edges are averaged, producing centers from −0.19 through 0.99 s, and that same vector is copied into every trial.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
inp = np.vstack([centers, ...])
```

iii. The AI describes this as a common per-bin relative-time representation for stimulus-aligned trials.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the exact centers of the edges used to bin neural spikes, so each input sample corresponds to the same neural column.

ii.
```python
neural, edges = bin_spikes_for_trials(...)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. The notes report a sanity check that the converted time vector exactly matched expected stimulus-aligned bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is inferred from consecutive runs of equal `probabilityLeft` in the already-filtered trial table.

ii.
```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The notes state that block position must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop resets a counter to 1 when the prior changes and increments thereafter. Thus it is one-based, and because it runs after trial filtering it renumbers retained trials and can create false block boundaries or close gaps. The scalar is repeated across all time bins.

ii.
```python
if i == 0 or v != prev:
    c = 1
else:
    c += 1
out[i] = c
```

iii. The AI intended a continuous per-trial position within runs of constant prior, replicated through time; it did not discuss the indexing or post-filtering consequences.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trials table's `choice` column.

ii.
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. The notes say the IBL sign convention was checked against raw trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL `+1` (left) maps to 0 and `−1` (right) maps to 1; zero/no-response trials were previously removed. The trial category is repeated over all time bins.

ii.
```python
out[arr == 1] = 0
out[arr == -1] = 1
np.full_like(centers, choice[i], dtype=np.int64)
```

iii. This implements the requested left/right coding and the notes report a raw-data spot check.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from trial-table `probabilityLeft`.

ii.
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. The AI identifies this column as the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to categories 0, 1, and 2, respectively, and the category is repeated across time.

ii.
```python
m = {0.2: 0, 0.5: 1, 0.8: 2}
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. This is the mapping explicitly required by the task, and the notes report checking it against raw data.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
return np.load(posf), np.load(tsf)
```

iii. The notes identify raw wheel position/timestamps as the source and call for differentiation and interpolation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Timestamps are sorted and duplicates removed. Absolute finite-difference position velocity is assigned to interval midpoints, then linearly interpolated to trial-bin centers. Unlike `SessionLoader`, this does not first interpolate position at 1 kHz or apply the 20 Hz Butterworth filter.

ii.
```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
speed_mid = np.abs(dp / dt).astype(np.float32)
ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
```

iii. The notes say sorting/deduplication fixed runtime warnings and regard finite-difference absolute velocity as wheel speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles are computed from all finite aligned samples in a session. Values at or below the lower threshold are 0, above the lower are 1, and above the upper are 2.

ii.
```python
q1, q2 = np.quantile(x[finite], [1/3, 2/3])
y[x > q1] = 1
y[x > q2] = 2
```

iii. The AI chose session-level tertiles to create three categorical movement levels as required.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Speed is linearly interpolated at `stimOn_times + centers`, the same relative bin centers as neural columns.

ii.
```python
t = s + centers
y = np.interp(t, timestamps, values)
```

iii. The notes say wheel traces should be stimulus-aligned on the common 20 ms grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It loads available left and right `Camera.ROIMotionEnergy.npy` arrays and their corresponding `_ibl_*Camera.times.npy` timestamps. If both exist, their aligned traces are averaged; otherwise the available side is used.

ii.
```python
if left_me and left_t: streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
if right_me and right_t: streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
...
me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) ...]
```

iii. The notes identify released whisker-pad ROI motion energy as the source and characterize averaging both available sides as a sensible consistent rule.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each released motion-energy trace is linearly interpolated to trial bin centers; two cameras are averaged after alignment. No filtering or normalization is applied before session-level discretization.

ii.
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. The notes say to use the video ROI motion-energy stream and interpolate it to the common stimulus-aligned bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel speed, session-wide 1/3 and 2/3 quantiles define categories with strict `>` comparisons.

ii.
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
discretize_with_thresholds(me_trials[i], me_q1, me_q2)
```

iii. Tertiles were chosen to satisfy the required three-bin categorical output.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated at each trial's stimulus time plus the neural-bin centers.

ii.
```python
t = s + centers
y = np.interp(t, timestamps, values)
```

iii. The notes describe a common stimulus-aligned grid and cite above-chance decoding as an alignment sanity check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions missing required trial columns, usable spikes, or at least two retained trials are skipped. Missing wheel or camera streams are silently replaced by zero traces; malformed sessions are caught by a broad exception and skipped. Duplicate wheel timestamps are removed. Missing trial values are filtered. Interpolation outside stream support clamps to endpoint values because no explicit coverage check is made.

ii.
```python
if wheel_pos is None:
    wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
...
except Exception as e:
    print(f'[WARN] failed session {sess}: {e}')
```

iii. The notes document fixes for duplicate wheel timestamps, no-response trials, and all-zero neural trials, but do not justify treating an absent behavioral stream as genuine zero behavior.

## 10-a. What are the most time-consuming steps of the code?

i. Repeatedly loading large per-probe spike arrays and, especially, scanning every session-wide spike array once for every trial dominate. The full sequential conversion took about 14,653 seconds, with late sessions taking up to 232 seconds.

ii.
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. The notes merely list “Code inefficiencies identified” without details; runtime logs provide the main evidence.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The block-position loop, per-spike cluster remapping list comprehension, per-trial full-spike scan/binning, per-trial interpolation, category mapping, and per-trial assembly loops are candidates. Most importantly, sorted spike times could be sliced with `searchsorted` instead of scanning all spikes per trial.

ii.
```python
sc = np.array([remap[c] for c in sc], dtype=np.int64)
for s in stim_on:
    rel = spike_times - s
```

iii. The AI did not provide a substantive efficiency justification; the full-run timing shows the cost of the approach.

## 10-c. What processing does the code repeat multiple times?

i. It recursively searches each probe directory separately for every required file, scans the entire spike-time array for every trial, rebuilds the same time input for every trial, and performs nearly identical interpolation and tertile processing for wheel and motion energy.

ii.
```python
np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
...
for s in stim_on:
    rel = spike_times - s
```

iii. No explicit justification was documented for these repetitions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It imports Matplotlib unconditionally and, when requested, renders diagnostic plots not used in the dataset. It loads/constructs local `pks`, reads cluster channel/anatomy details to generate 545 raw CCF-ID region categories that the decoder does not use as prediction variables, and computes the trial-wide neural matrix before discarding all-zero trials. It also repeatedly copies constant per-trial choice, prior, and block values across 60 bins, though the target format permits per-trial values.

ii.
```python
pks = sorted(probe_dir.rglob('pykilosort'))
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. Plotting was offered as a processing diagnostic; the other discarded or redundant work was not justified in the notes.
