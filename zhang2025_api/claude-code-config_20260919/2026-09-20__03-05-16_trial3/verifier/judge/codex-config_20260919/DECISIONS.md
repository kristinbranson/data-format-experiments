# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds 459 session jobs from the methods repository's `bwm_release.csv`, then loads trials, wheel, camera motion energy, spikes, clusters, and channels through ONE-backed `SessionLoader` and `SpikeSortingLoader`. It may first rebuild local ONE index tables when absent.

ii. `bwm = pd.read_csv(BWM_FREEZE, index_col=0)`; `sl = SessionLoader(one=one, eid=eid); sl.load_trials()`; `ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)`

iii. It says this uses the same freeze and loaders as the reference pipeline, preserves the real eid/pid mapping offline, and obeys the requirement that scientific data access go through ONE/brainbox.

## 1-b. How are the data split into subjects?

i. Subject names come from `bwm_release.csv`; converted sessions retain their subject, and sorted unique names plus a per-session index are assembled at the end.

ii. `subjects = sorted({r['subject'] for r in ok})`; `subject_idx = np.array([sub2idx[r['subject']] for r in ok], dtype=np.int64)`

iii. The freeze already contains the authoritative subject associated with every eid.

## 1-c. How are the data split into sessions?

i. Rows of the freeze are grouped by `eid`; all probe rows for an eid become one session job and successful jobs become list elements in the output.

ii. `for eid, g in bwm.groupby('eid', sort=True): ... jobs.append((eid, g['subject'].iloc[0], probes))`

iii. An eid is the native ONE session identifier, and pooling its probes follows the reference.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one row per trial. Retained row indices address aligned windows, and session arrays are finally converted to lists of per-trial matrices.

ii. `keep = np.flatnonzero(mask)`; `t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]`; `'neural': [[r['neural'][k] ...] ...]`

iii. The trials table already defines trial boundaries/events, so no inferred segmentation is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI calls the repository's `load_trials_and_mask` with RT 0.08–2 s, default NaN exclusions, maximum trial length 10 s, no-choice exclusion, and retention of unbiased trials. It additionally requires wheel/camera window coverage and spike-recording coverage, and drops sessions with fewer than two retained trials.

ii. `load_trials_and_mask(... min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default', max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True, ...)`; `valid = beh_good & in_rec`

iii. It attributes the mask to the paper/reference function and says coverage checks mirror `align_spike_behavior` and prevent fabricated interpolation or silent all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices derive from every probe's `spikes.times` and `spikes.clusters`; cluster `label` and channel-derived `acronym` determine unit retention and region metadata.

ii. `spikes_list.append({'times': spikes['times'], 'clusters': spikes['clusters'].astype(np.int64)})`; `clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`

iii. Only times and cluster assignments are needed for counts, avoiding unused spike amplitudes/depths.

## 2-b. How is the `neural` data processed?

i. Probes are merged, retained clusters are reindexed, spikes are sorted, and a vectorized `np.bincount` creates raw float32 spike counts per neuron per 20-ms bin. Unlike the human solution, the AI does not divide counts by 0.02 to obtain Hz.

ii. `spikes, clusters = merge_probes(spikes_list, clusters_list)`; `binned.reshape(-1)[:] = np.bincount(...)`; `'neural': binned`

iii. The notes explicitly choose raw counts because reference normalization occurs later and the supplied decoder projects the data; they also claim bit-identical count binning to the reference binner.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains clusters with `label >= 1` and excludes both Beryl `root` and `void`, then drops sessions with no retained unit.

ii. `good_unit = (label >= GOOD_UNIT_LABEL) & ~np.isin(beryl, NON_GREY)` where `NON_GREY = ('root', 'void')`

iii. It follows the data paper's well-isolated/grey-matter analysis and argues this avoids an approximately eightfold expansion. The human reference excludes `void` but deliberately keeps `root`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each window begins at `stimOn_times - 0.5` and ends at +1.5 s; spike bin indices are computed relative to that start.

ii. `t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]`; `rel = spike_times[idx] - t_begs[trial_id]`

iii. This directly uses the requested stimulus-onset event and the reference decoding window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 100 non-overlapping 20-ms bins over two seconds. Spikes are counted directly into them; no smoothing or later rebinning is applied.

ii. `BINSIZE = 0.02`; `N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))`

iii. These are the exact caching parameters and model resolution documented by the reference.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed window/bin parameters relative to each trial's `stimOn_times`, rather than from a sampled raw stream.

ii. `bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE)`

iii. The task asks for elapsed time and the alignment event defines zero.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes the 100 bin centers from -0.49 through 1.49 s and repeats the same vector for every trial.

ii. `inputs[:, 0, :] = bin_centers[None, :]`

iii. It describes bin centers as the natural timestamps for the neural bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Element i labels neural spike-count bin i with that bin's center.

ii. `inputs[:, 0, :] = bin_centers[None, :]`

iii. This is intended to share the neural grid exactly.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from runs of equal `trials.probabilityLeft` values in the full trials table.

ii. `change[1:] = pl[1:] != pl[:-1]`; `block_id = np.cumsum(change) - 1`

iii. No explicit block id exists, while the task prior is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI subtracts each block's starting row from the row index, producing a zero-based count before filtering, then broadcasts the retained count across time.

ii. `np.arange(len(pl)) - starts[block_id]`; `inputs[:, 1, :] = tib_all[keep][:, None]`

iii. Computing before exclusion preserves the animal's true trial position when intervening bad trials are removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column.

ii. `choice = map_choice(trials['choice'].to_numpy()[keep])`

iii. This is the released behavioral choice variable used by the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. `+1` is mapped to 0/left and `-1` to 1/right; no-choice trials were already removed, and the value is broadcast across 100 bins.

ii. `out[np.asarray(choice) == 1] = 0`; `out[np.asarray(choice) == -1] = 1`; `outputs[:, 0, :] = choice[:, None]`

iii. The sign mapping was checked against stimulus side/correct trials and matches the requested coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii. `prior = map_prior(trials['probabilityLeft'].to_numpy()[keep])`

iii. This field is the block prior in the IBL task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 and broadcast over time.

ii. `out[np.isclose(pl, 0.2)] = 0`; `out[np.isclose(pl, 0.5)] = 1`; `out[np.isclose(pl, 0.8)] = 2`

iii. This is the mapping explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from wheel timestamps and position as transformed by `SessionLoader.load_wheel()` into velocity; the absolute velocity is used.

ii. `sl.load_wheel()`; `return (sl.wheel['times'].to_numpy(), np.abs(sl.wheel['velocity'].to_numpy()))`

iii. It matches the reference's `wheel-speed` target loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. SessionLoader interpolates/filters the wheel to obtain velocity; the AI takes its magnitude and linearly interpolates it at 100 right-edge query times for each trial.

ii. `grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)`; `interp = np.interp(query.ravel(), target_times, target_vals)`

iii. The right-edge grid is intended to reproduce `get_behavior_per_interval` exactly.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two per-session tertiles over all retained trial/time values define low, medium, and high; explicit fallback edges handle ties.

ii. `edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])`; `np.digitize(values, edges, right=False)`

iii. Per-session tertiles balance classes and accommodate mouse/session scale differences.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Both use the same stimulus-aligned two-second window, but wheel samples are evaluated at each neural bin's right edge while input time labels use centers.

ii. `query = t_begs[good][:, None] + grid[None, :]` with `grid = 0.02 ... 2.0`

iii. The AI says this mirrors the methods-code behavioral helper. The human reference instead evaluates behavior at bin centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses released left-camera whisker ROI motion energy and timestamps, falling back to the right camera.

ii. `sl.load_motion_energy(views=[view])`; `me['whiskerMotionEnergy'].to_numpy()`

iii. This follows the reference preference and handles sessions lacking the left stream.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is otherwise unchanged and linearly interpolated onto the same right-edge grid used for wheel speed.

ii. `me_vals, me_good = bin_behavior(mt, mv, t_begs)`

iii. The notes state no extra filtering/normalization is appropriate and claim equivalence to the reference helper.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses per-session tertiles over retained values, including the same degenerate-edge fallback.

ii. `me_cls, me_edges = discretize_tertiles(me_vals)`

iii. Motion energy is camera/ROI dependent, so session-specific balanced categories are justified.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It shares the stimulus-aligned window but is sampled at neural-bin right edges, not the centers used by the human solution.

ii. `grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)`

iii. The AI chose right edges to reproduce the repository's behavior helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Default NaN trial fields, uncovered/NaN behavior windows, windows outside the retained spike train, absent camera streams, no-unit sessions, and sessions below two trials are dropped. Per-session conversion catches exceptions and records skip reasons. Degenerate tertile edges receive deterministic fallbacks.

ii. `valid = beh_good & in_rec`; `if mt is None: res['skip'] = 'no whisker motion energy (neither camera)'`; `except Exception as e: res['skip'] = ...`

iii. It prefers exclusion to inventing data and preserves a full audit trail in metadata/logs.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging large spike objects, binning spikes, parallel session conversion, and serializing the approximately 11.7-GB pickle dominate.

ii. `ssl.download_spike_sorting_object('spikes')`; `binned = bin_spikes(...)`; `with Pool(args.n_workers)`

iii. The notes identify spike I/O as the major cost and report about 70 seconds for the full parallel conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI specifically replaces the reference's per-trial spike-binning and behavior-interpolation loops with session-wide vectorized operations. Remaining loops over probes, sessions, and output list assembly are structurally natural; diagnostic plotting loops are optional.

ii. `flat = (...)`; `np.bincount(flat, ...)`; `query = t_begs[good][:, None] + grid[None, :]`

iii. It reports the vectorized spike binner as about 100 times faster and bit-identical at the count stage.

## 10-c. What processing does the code repeat multiple times?

i. A new SessionLoader is constructed for trials, wheel, motion energy, and optional plots; each probe separately loads/merges sorting metadata. Wheel and whisker independently call the same behavior-binning and tertile routines. Assembly loops separately over neural/input/output trials.

ii. `sl = SessionLoader(one=one, eid=eid)` appears in `_convert_session`, `load_wheel_speed`, and `load_whisker_me`; both traces call `bin_behavior` and `discretize_tertiles`.

iii. The AI favors small reusable loaders and notes that one ONE/BrainRegions object is cached per worker.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal runs retain diagnostic-only raw wheel/ME arrays, bin starts, tertile edges, trial indices, timing data, and cluster metadata in intermediate results; most are reduced to metadata or discarded after assembly. It also sorts retained spikes even when probe merge may already yield time order and downloads/loads cluster/channel fields beyond the two spike arrays.

ii. `order = np.argsort(spike_times_k, kind='stable')`; res stores `'wheel_raw'`, `'me_raw'`, `'t_begs'`, and `'timing'`.

iii. These support validation, plotting, provenance, and safe vectorized search; the notes contrast this with larger genuinely unused reference loads that the AI avoided.
