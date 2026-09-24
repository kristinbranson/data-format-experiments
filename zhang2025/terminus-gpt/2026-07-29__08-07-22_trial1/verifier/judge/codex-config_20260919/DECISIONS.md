# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `data/one_cache/Brainwidemap/sessions.pqt`, iterates every metadata row, constructs the corresponding session directory directly, and opens ALF parquet/NumPy files. Sessions lacking required local files are skipped. It does not use ONE searches to preselect sessions with all required datasets.

ii.
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
```
```python
return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"
```

iii. The notes identify the data as a native IBL ONE/ALF cache and say the conversion should follow IBL loading conventions, but the implemented script uses direct filesystem discovery. The notes report that sample mode scanned metadata until it found two valid sessions.

## 1-b. How are the data split into subjects?

i. The subject string comes from each session-index row. A first-seen-order subject list and map are built, and one `subject_idx` is appended per retained session.

ii.
```python
subject = str(row['subject'])
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The agent treated the index's `subject` field as the unique mouse identifier, consistent with its notes that the cache is organized by lab and subject.

## 1-c. How are the data split into sessions?

i. Each row of `sessions.pqt` is treated as one session and becomes one element of the outer `neural`, `input`, and `output` lists if loading succeeds.

ii.
```python
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
    if loaded is None:
        continue
    all_neural.append(neural)
```

iii. The notes recognize the ONE cache's session hierarchy and the target format's session-level outer list.

## 1-d. How are the data split into trials?

i. The trials parquet table supplies one row per trial. After masking and resetting its index, the agent loops over `stimOn_times` and creates one neural/input/output array per retained row.

ii.
```python
trials = trials.loc[mask].reset_index(drop=True)
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(..., st))
```

iii. The notes say trial tables and event-aligned intervals should define trials, matching the IBL organization.

## 1-e. How are trials filtered based on quality controls?

i. Trials are initially retained when `stimOn_times`, `choice`, and `probabilityLeft` are finite. Later, trials whose mapped choice or prior is non-finite are removed. There is no 80 ms–2 s reaction-time mask and no requirement that wheel/camera streams span the full trial window.

ii.
```python
need = ['stimOn_times', 'choice', 'probabilityLeft']
for c in need:
    if c in trials.columns:
        mask &= np.isfinite(trials[c].to_numpy())
```
```python
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
```

iii. The notes state an intention to use the reference trial mask and exclude missing outputs, but the final code only implements finite-value checks. The critical review discusses all-zero neural trials, not the omitted reaction-time/coverage criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays are derived from each probe's `spikes.times.npy` and `spikes.clusters.npy`; `clusters.metrics.pqt` is used for unit QC. Channel files are named but not used to derive neural values.

ii.
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
cm = base / 'clusters.metrics.pqt'
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. The notes explicitly map `spikes.times`, `spikes.clusters`, and curated cluster metadata to `neural`, following the reference spike-binning approach.

## 2-b. How is the `neural` data processed?

i. Good-cluster spikes from all probes are remapped into a shared unit index, concatenated, sliced around each stimulus, and counted into 20 ms bins with `np.add.at`. Counts are saved as float32 without division by bin width, so they are spike counts rather than the reference firing rates in Hz. Concatenated probes are not globally time-sorted, although `searchsorted` assumes they are.

ii.
```python
return np.concatenate(all_times), np.concatenate(all_clusters), region_names
```
```python
lo = np.searchsorted(spike_times, edges[0], side='left')
hi = np.searchsorted(spike_times, edges[-1], side='right')
np.add.at(mat, (sc, tbin), 1)
```

iii. The notes planned “bin spike counts in 20 ms bins” and merging probes after curation. They do not justify omitting the reference conversion to Hz or the missing global sort.

## 2-c. How is the `neural` data filtered based on quality controls?

i. If cluster metrics have `label`, only rows equal to 1 are kept; otherwise `ks2_label == 'good'` is used. If neither is available, all observed clusters are retained. No anatomical `void` exclusion is performed and every retained neuron's region is recorded as `unknown`.

ii.
```python
if 'label' in cols:
    good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
elif 'ks2_label' in cols:
    good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
if good_ids is None:
    good_ids = np.unique(spikes_c)
```

iii. The notes chose “good units only” to follow IBL QC. They acknowledge the final limitation that brain regions remain `unknown`, but do not discuss the permissive all-cluster fallback or lack of `void` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial bin edges are offsets from that trial's `stimOn_times`, aligning neural activity to stimulus onset. The chosen window is −0.2 to 1.0 s.

ii.
```python
T_START = -0.2
T_END = 1.0
edges = stim_time + build_time_edges()
```

iii. Stimulus-onset alignment was explicitly required and selected in the notes. The notes do not justify changing the reference decoding window from −0.5…1.5 s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 60 bins over the 1.2 s window. Raw spikes are binned directly; there is no further neural resampling or rebinning.

ii.
```python
BIN_SIZE_S = 0.02
return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

iii. The notes cite the papers' 20 ms population and wheel analyses and chose a common 20 ms grid for all streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the predefined offsets around each trial's `stimOn_times`; because every trial is stimulus-aligned, the same vector of bin centers is reused.

ii.
```python
centers = build_time_centers().astype(np.float32)
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
```

iii. The notes describe a stimulus-onset-aligned continuous time axis replicated across trials.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Edges from −0.2 to 1.0 s in 20 ms increments are averaged pairwise to obtain centers (−0.19 through 0.99 s), then copied into each trial input.

ii.
```python
e = build_time_edges()
return (e[:-1] + e[1:]) / 2
```

iii. The agent justified bin centers as the continuous time-since-onset input, though not its shorter window.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the centers of the exact edges used for neural spike binning, so input column `t` describes neural bin `t`.

ii.
```python
edges = stim_time + build_time_edges()
input_trials.append(np.vstack([centers, ...]))
```

iii. The planned common grid was intended to keep neural, input, and output streams aligned.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive runs of the filtered trials' `probabilityLeft` values; a probability change starts a new block.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

iii. The notes explain that no native block ID exists and identify block-prior run lengths as the source.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop assigns 1 to the first trial and increments while `probabilityLeft` is unchanged, resetting to 1 after a change. It is computed after filtering, so removed trials do not advance the count, and is broadcast across all time bins.

ii.
```python
run = 1
out[0] = 1
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
    out[i] = run
```

iii. The notes planned a custom run-length transform and broadcasting, but do not justify the one-based count or counting after trial removal; the reference uses zero-based positions computed before filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column.

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. The notes identify `trials.choice` as the source and the raw-label sanity check compared it against the converted first trial.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw `+1` is mapped to left/category 0 and `−1` to right/category 1; other values become NaN and are dropped. The category is repeated over all 60 time bins.

ii.
```python
return 0 if v == 1 else 1 if v == -1 else np.nan
```
```python
np.full(centers.shape, int(choice[i]), dtype=np.int64)
```

iii. This mapping and broadcasting were selected to satisfy the required binary categorical output and uniform tensor shapes.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials table's `probabilityLeft` column.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. The notes identify the IBL block prior as the direct source.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 map to 0, 1, and 2; other values become NaN and are dropped. The category is repeated across the trial.

ii.
```python
if np.isclose(v, 0.2):
    return 0
if np.isclose(v, 0.5):
    return 1
if np.isclose(v, 0.8):
    return 2
```

iii. This is the mapping explicitly required by the task and recorded in the notes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived directly from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
return np.load(ts), np.load(pos)
```

iii. The notes map cached wheel position/velocity to wheel speed and cite the paper's wheel variable.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Absolute first differences of position divided by timestamp differences are assigned to interval midpoints. Those samples are averaged within each 20 ms trial bin. Unlike the reference `SessionLoader`, this does not interpolate position to 1 kHz or apply its 20 Hz Butterworth filter.

ii.
```python
mids = ts[:-1][good] + dt[good] / 2
vv = np.abs(dp[good] / dt[good]).astype(np.float32)
```
```python
out[i] = np.nanmean(values[m])
```

iii. The notes intended reference-consistent wheel processing and 20 ms averaging, but do not document the omitted IBL interpolation/filtering.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All finite aligned wheel samples in a session are pooled; the session's 1/3 and 2/3 quantiles define categories 0, 1, and 2. Missing per-bin values are replaced by the lower threshold before digitization, placing them in the middle class under `right=False`.

ii.
```python
wq = np.quantile(all_wheel_cont, [1/3, 2/3])
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0]), wq, right=False)
```

iii. The notes chose three bins as required, with thresholds to be distribution-driven. Later notes say NaNs were filled to force integer categorical outputs; they do not justify assigning missing samples to the middle category.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel midpoint samples are selected relative to `stimOn_times` and averaged within the same stimulus-relative 20 ms edges as neural activity.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. A common stimulus-onset grid was selected to align behavior and neural tensors bin-for-bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The agent loads available `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` arrays, truncates them to a common length, and averages them. It does not load `_ibl_<side>Camera.times.npy` directly.

ii.
```python
for p in [left_p, right_p]:
    if p is not None and p.exists():
        vals.append(np.load(p).astype(np.float32))
n = min(len(v) for v in vals)
arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
```

iii. The notes planned cached video-derived whisker motion energy, but the final notes acknowledge that explicit ROI timestamps were not loaded and timestamps are approximated.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Aside from averaging available cameras, the released motion-energy values are not filtered or normalized. They are averaged within 20 ms bins using guessed timestamps. Missing bins are later imputed at the lower quantile.

ii.
```python
ms = bin_signal(me_ts, me, st + centers)
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0]), mq, right=False)
```

iii. The notes say to use video-derived variables and discretize them. They explicitly acknowledge approximate timing as a known limitation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Finite aligned samples are pooled within each session, and the 1/3 and 2/3 quantiles define categories 0–2. Missing samples are replaced by the lower threshold and therefore assigned category 1.

ii.
```python
mq = np.quantile(all_me_cont, [1/3, 2/3])
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0]), mq, right=False)
```

iii. Equal-frequency, session-wise categories implement the required three bins; NaN filling was added after validation complained about non-integer outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The values are binned relative to each `stimOn_times` on the common grid, but timestamps are inferred from a camera-features column when available or linearly spread between the first and last trial windows otherwise. Thus nominal tensor bins align, but physical camera/neural alignment is not reliable.

ii.
```python
me_ts = guess_motion_timestamps(session_dir, len(me))
return np.linspace(t0, t1, n, dtype=np.float32)
```

iii. The notes explicitly state that whisker timestamps are approximated because explicit ROI timestamps were not loaded.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing sessions, trial tables, spikes, or sessions with fewer than two finite trials are skipped. Missing cluster QC falls back to all clusters. Missing camera timestamps are guessed. Missing behavioral bins are retained and imputed at the lower quantile rather than dropping the affected trial. Exceptions reading feature tables are silently ignored.

ii.
```python
if not session_dir.exists(): return None
if trials is None or 'stimOn_times' not in trials.columns: return None
if good_ids is None: good_ids = np.unique(spikes_c)
```
```python
except Exception:
    pass
```

iii. The notes report the timestamp approximation and say sample-stage NaN outputs were “fixed” by filling them before discretization. This prioritizes format validation over faithful handling of missing measurements.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading large spike arrays and then repeatedly binning spikes and behavioral samples per trial; the notes observed roughly 35–75 seconds per valid session and projected several hours for a full run.

ii.
```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(...))
```

iii. The runtime estimates in the notes identify session scanning/conversion as costly, but no detailed profiler results were recorded.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidate loops include session-row iteration, the Python remapping of every spike cluster, the per-trial neural/behavior loop, `bin_signal`'s per-bin masks, the block-number loop, and the second per-trial output loop. The cluster remap and repeated bin masks are especially avoidable with indexed arrays/bincount-style reductions.

ii.
```python
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
for i in range(len(centers)):
    m = idx == i
```

iii. The notes contain placeholders for inefficiencies/speedups and do not document a vectorization analysis.

## 10-c. What processing does the code repeat multiple times?

i. It loads the trials table once in `load_session` and potentially again in `guess_motion_timestamps`; computes bin membership separately for wheel and motion energy on every trial; scans each binned trace once to collect finite values and again to digitize; and performs two separate trial loops for inputs/neural traces and outputs.

ii.
```python
trials = load_trials(session_dir)
# guess_motion_timestamps may call:
trials = load_trials(session_dir)
```

iii. The notes do not discuss these repeated operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `wheel_bins_all` and `me_bins_all` are computed but never used. `st = st[good]` is assigned after bin indices are computed but not read afterward. The script also locates `clusters.channels.npy` without using it and collects `region_names` only to replace every region by `unknown`/zero.

ii.
```python
wheel_bins_all = discretize_three_bins(all_wheel_cont)
me_bins_all = discretize_three_bins(all_me_cont)
```
```python
st = st[good]
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The notes do not identify these dead computations; their development section leaves “Code inefficiencies identified” as a placeholder.
