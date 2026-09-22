# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses the local ONE cache for the actual arrays, but selects the public BWM release from `/app/code/code_zhang2025/data/bwm_release.csv`. It optionally restricts that list with `/app/DATALIMIT_SUBSET.csv`, then processes every selected session in parallel. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads every released probe listed for that session. Results are sorted by `eid` before assembly.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(_worker, task): task[0] for task in tasks}
```

iii. The trajectory shows that the agent identified the BWM release table as the curated population and used ONE loaders so dataset names, revisions, and clock synchronization were handled by IBL tooling. It tested four sessions, then ran the full conversion with 24 workers.

## 1-b. How are the data split into subjects?

i. Subject labels come directly from the release CSV. After conversion, unique subject strings are sorted and each retained session gets an integer `subject_idx`.

ii.
```python
tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
              str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The agent treated the release table's subject field as the authoritative mouse identifier; no subject ID was inferred from paths or filenames.

## 1-c. How are the data split into sessions?

i. The release CSV's unique `eid` values define sessions. Probe rows are grouped by `eid`, each session is converted independently, and its probes are merged into one neural population.

ii.
```python
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
by_eid = bwm_df.groupby('eid')
rows = by_eid.get_group(eid)
```

iii. The agent reasoned that an IBL experiment ID is already the session unit and that probes in the same session share behavior and should be merged, following `merge_probes`.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one table row per trial. For each retained row, the agent creates a two-second interval beginning 0.5 s before `stimOn_times`, bins neural activity, and emits one neural/input/output array.

ii.
```python
sess_loader.load_trials()
trials = sess_loader.trials
interval_begs_all = align_times + TIME_WINDOW[0]
...
'neural': [binned_spikes[k] for k in range(n_trials)]
```

iii. The trajectory and comments treat the trials table as already trial-split; only temporal slicing around the requested event is needed.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes trials with reaction times outside 0.08–2 s, feedback more than 10 s after go cue, missing `stimOn_times`, choice, feedback time, prior, first movement, or feedback type, or `choice == 0`. It also rejects unknown prior values and trials whose wheel or camera streams fail to cover the full window or contain nonfinite interpolated values. Sessions with fewer than two surviving trials are skipped.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
...
keep &= good
```

iii. The agent explicitly said this reproduced `load_trials_and_mask(..., max_trial_len=10.)` from the methods repository and added stream coverage because every retained trial must have complete time-varying outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `spikes.times` and `spikes.clusters` for every released insertion. The cluster/channel tables contribute cluster quality and anatomical acronym used for filtering and region metadata.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
sc = np.asarray(spikes['clusters'], dtype=np.int64)
st = np.asarray(spikes['times'], dtype=np.float64)
```

iii. The agent identified spike time and cluster assignment as the raw quantities needed for population binning, while merged cluster metadata supplies QC and anatomy.

## 2-b. How is the `neural` data processed?

i. Good units from all probes are compactly renumbered and pooled. Spikes are stably sorted by time and counted per unit in 100 nonoverlapping 20 ms bins for each trial. The saved values are raw spike counts (`float32`), not counts divided by 0.02 s.

ii.
```python
b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
counts = np.bincount(u * N_BINS + b, minlength=flat)
out[k] = counts.reshape(n_units, N_BINS)
```

iii. The agent stated that this matches the repository's `bincount2D` behavior and described the metadata as “spike count per 20 ms bin.” It did not justify departing from the human conversion's Hz scaling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have cluster `label >= 1`, a finite spike time, and a Beryl acronym other than both `root` and `void`. Sessions with no retained unit are dropped.

ii.
```python
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY_MATTER)
...
ok &= np.isfinite(st)
```

iii. The agent tied `label == 1` to the data paper's well-isolated-unit criteria and interpreted both `root` and `void` as non-grey/unassigned locations. The latter interpretation is stricter than the human reference, which only removes `void`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. Neural bin 0 starts at stimulus onset minus 0.5 s and the final bin ends at onset plus 1.5 s.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
interval_begs_all = align_times + TIME_WINDOW[0]
```

iii. The agent cited the reference caching parameters: `align_time='stimOn_times'` and `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins over two seconds. Spike events are newly histogrammed directly into these bins; there is no smoothing or subsequent rebinning/resampling of the neural matrix.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The agent selected the method repository's 20 ms caching bin size and verified 100 timepoints per trial.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the requested `stimOn_times` alignment and the fixed -0.5 to 1.5 s bin grid, rather than another recorded signal.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. The agent used stimulus onset because that is the required alignment event and represented elapsed time at neural-bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs the 100 bin centers from -0.49 through 1.49 s and repeats the same vector for every trial.

ii.
```python
inputs[:, 0, :] = bin_centres[None, :]
```

iii. Its comments explain that a continuous elapsed-time covariate should correspond to the centers of the neural bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `i` is the center of neural bin `i`, whose interval begins at `-0.5 + 0.02*i` seconds.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. The agent intentionally used centers even though its behavioral outputs use right edges.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive values of the trials-table `probabilityLeft` column; a change indicates a new block.

ii.
```python
p_left = trials['probabilityLeft'].to_numpy(dtype=float)
new_block[1:] = ~(p_left[1:] == p_left[:-1])
```

iii. The agent noted that no explicit block ID is required because the prior is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. It computes a zero-based position from the start index of each contiguous block, before filtering trials, then broadcasts the retained trial's scalar across all 100 timepoints.

ii.
```python
block_id = np.cumsum(new_block) - 1
block_start = np.flatnonzero(new_block)
idx_in_block = np.arange(len(p_left)) - block_start[block_id]
inputs[:, 1, :] = n_in_block[:, None]
```

iii. The trajectory records a unit test that exposed and fixed an initial indexing bug. The agent justified pre-filter computation as preserving the animal's real position even when an intervening trial is excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`, whose retained values are +1 (left) and -1 (right).

ii.
```python
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
```

iii. The agent says it verified the IBL sign convention against left-contrast rewarded trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. It maps +1 to 0 and -1 to 1, then repeats the per-trial class over 100 timepoints. No-response trials were already excluded.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. This directly implements the task requirement left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
p_left_all = trials['probabilityLeft'].to_numpy(dtype=float)
```

iii. The agent identified this column as the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2; anything else is excluded. The scalar class is repeated over the trial.

ii.
```python
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code
outputs[:, 1, :] = prior[:, None]
```

iii. This mapping is explicitly required by the task; `isclose` avoids brittle floating-point equality.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses wheel timestamps and the absolute value of the velocity produced by `SessionLoader.load_wheel()` from raw wheel position/timestamps.

ii.
```python
sl.load_wheel()
traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                         np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))
```

iii. The agent followed the repository's `load_target_behavior` convention that speed is absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` interpolates/filter-differentiates wheel position into velocity; the agent takes its magnitude, linearly interpolates it at 100 trial-relative sample times, validates full-window coverage and finiteness, and then discretizes it.

ii.
```python
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
good &= np.all(np.isfinite(binned), axis=1)
```

iii. The agent relied on the standard IBL loader for velocity processing and said right-edge interpolation reproduced `get_behavior_per_interval`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It computes the 1/3 and 2/3 quantiles over all retained wheel samples in each session and applies `np.digitize`, producing 0/1/2. If both thresholds coincide, the upper is moved to the next representable float.

ii.
```python
lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
if not hi > lo:
    hi = np.nextafter(lo, np.inf)
return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))
```

iii. The agent argued session-wise tertiles balance classes and avoid encoding session-specific measurement scale or movement vigor.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is stimulus-aligned to the same two-second window, but sampled at the right edge of every neural bin (`-0.48, -0.46, ..., 1.5`), not at the neural bin centers.

ii.
```python
offsets = (np.arange(N_BINS) + 1) * BINSIZE
sample_times = interval_begs[:, None] + offsets[None, :]
```

iii. The agent explicitly chose right edges because the methods repository's interpolation helper used `beg + binsize` through `end`; this differs by 10 ms from the human conversion.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `whiskerMotionEnergy` and frame `times` from left-camera ROI motion energy, with right camera as a fallback.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    sl_me.load_motion_energy(views=[view])
    df = sl_me.motion_energy[cam]
```

iii. The agent followed the reference preference for the left side view while allowing sessions containing only the right camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI trace is used without normalization or smoothing, linearly interpolated at the trial sample times, checked for coverage/nonfinite values, and discretized per session.

ii.
```python
whisker = (df['times'].to_numpy(dtype=np.float64),
           df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
```

iii. The agent found no additional processing in the referenced behavior pipeline.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same session-wise 1/3 and 2/3 quantile thresholds and 0/1/2 digitization used for wheel speed are applied.

ii.
```python
whisk_lvl, whisk_thr = discretize_tertiles(
    beh_binned['whisker-motion-energy'][keep])
```

iii. The agent specifically noted that camera view, frame rate, illumination, and ROI size make a global raw threshold inappropriate.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It uses the same stimulus-aligned window as neural data but is interpolated at each bin's right edge, 10 ms later than its corresponding neural-bin center.

ii.
```python
sample_times = interval_begs[:, None] + offsets[None, :]
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. As for wheel speed, the agent justified this by the original repository interpolation helper, whereas the human reference uses bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trial fields and incomplete/nonfinite behavioral windows are filtered. Nonfinite spike times and empty probes are ignored. Sessions with fewer than two valid trials or no retained units are skipped. In addition, `_worker` catches every session exception, records the traceback-derived reason, and continues; skipped sessions are listed in metadata.

ii.
```python
if len(spikes) == 0 or len(clusters) == 0:
    continue
...
except Exception as exc:
    return eid, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
```

iii. The agent prioritized completing the release despite isolated bad sessions. The trajectory separately investigated a zero-spike trial and concluded it reflected a real recording gap rather than a conversion bug.

## 10-a. What are the most time-consuming steps of the code?

i. Loading and merging large spike-sorting arrays is the dominant per-session work; full conversion also spends time binning spikes and serializing the large pickle. The agent parallelized sessions with 24 workers for the production run.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
...
with ProcessPoolExecutor(max_workers=args.workers) as pool:
```

iii. The trajectory's long-timeout conversion and subsequent verification indicate that disk-heavy spike loading/full-dataset processing were treated as the costly stages.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes` could be vectorized by assigning spikes to trial/bin indices in bulk. The probe/session loops are heterogeneous I/O work and are less naturally vectorized; sessions are parallelized instead. The small prior mapping loop could be replaced by a lookup but is immaterial.

ii.
```python
for k in range(n_trials):
    t = spike_times[i0[k]:i1[k]]
    ...
    counts = np.bincount(u * N_BINS + b, minlength=flat)
```

iii. The agent did not explicitly discuss vectorization in its final rationale. Its code already vectorizes behavioral interpolation across trials and uses `bincount` within each trial, leaving only variable-length spike slices looped.

## 10-c. What processing does the code repeat multiple times?

i. It creates separate `SessionLoader` instances for trials, wheel, and motion energy, and repeatedly creates ONE/atlas state in worker processes. Both behavioral streams pass through the same interpolation/coverage function and the same tertile function. Per-trial output broadcasting also repeats constant choice/prior values.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
...
sl = SessionLoader(one=one, eid=eid)
...
sl_me = SessionLoader(one=one, eid=eid)
```

iii. The trajectory offers no explicit justification beyond modular separation and process isolation. Reusing one session loader could reduce setup, though the repeated operations are small relative to spike I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges `channels` and a full cluster table because QC/anatomy require them, so those are not wholly unnecessary. More plausibly discardable work includes retaining `lab`, per-session thresholds, detailed session metadata, skipped-session diagnostics, and sorting unique region lists solely for metadata. `np.clip` on bins is redundant after half-open `searchsorted` bounds except for numerical edge cases.

ii.
```python
return {'eid': eid, 'subject': subject, 'lab': lab, ...
        'wheel_speed_thresholds': wheel_thr,
        'whisker_motion_energy_thresholds': whisk_thr}
```

iii. The agent did not identify unnecessary work in the trajectory. These additions aid provenance and debugging but are not consumed by the decoder itself.
