# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the ONE API pointed at a local cache (`/app/data/one_cache`). It reads the BWM release CSV (`/app/code/code_zhang2025/data/bwm_release.csv`) to get the list of session eids and their associated probe insertions. When a `DATALIMIT_SUBSET.csv` file exists, sessions are restricted to those listed there. For each session, `SessionLoader` loads trials, wheel, and motion energy, while `SpikeSortingLoader` loads the spike sorting for each probe insertion listed in the release CSV.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
if os.path.exists(DATALIMIT_SUBSET_CSV):
    subset = pd.read_csv(DATALIMIT_SUBSET_CSV)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    allowed = set(subset[col].astype(str))
    eids = [e for e in eids if e in allowed]
```
```python
sl = SessionLoader(one=one, eid=eid)
ssl = SpikeSortingLoader(pid=str(row['pid']), one=one, eid=eid, pname=row['probe_name'])
```

iii. The AI explored the data directory structure, the ONE cache, and the BWM release CSV. It chose to use the release CSV to enumerate sessions and their probes, rather than `one.search()`, because the CSV provides the canonical list of released sessions with their probe insertion IDs.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the BWM release CSV. Each session's subject name comes from the CSV row. At assembly, unique subjects are sorted alphabetically and each session is assigned a `subject_idx`.

ii.
```python
tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
              str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
```
```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The subject identity is directly available in the release CSV, so no parsing of paths or additional lookups is needed.

## 1-c. How are the data split into sessions?

i. A session is identified by its `eid` in the BWM release CSV. Each unique eid is processed independently. The release CSV is grouped by eid to get the probe insertions for each session.

ii.
```python
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
by_eid = bwm_df.groupby('eid')
for eid in eids:
    rows = by_eid.get_group(eid)
```

iii. Sessions are already the unit of organization in the BWM release.

## 1-d. How are the data split into trials?

i. The trials table, loaded via `SessionLoader.load_trials()`, has one row per trial. This table is the basis for splitting.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials = sess_loader.trials
```

iii. No decision to make; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI reimplements the reference `load_trials_and_mask` function with the same default parameters plus `max_trial_len=10.0`. Trials are excluded if: (1) reaction time (firstMovement_times - stimOn_times) < 0.08s or > 2.0s, (2) feedback_times - goCue_times > 10s, (3) any of stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/feedbackType is NaN, (4) choice == 0 (no response). Additionally, trials are excluded if the wheel or whisker traces don't cover the decoding window (within one bin tolerance) or have NaN values.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'
mask = ~trials.eval(query)
```
```python
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
# ... then also:
keep &= good  # from bin_behaviour coverage check
```

iii. The AI explicitly followed the reference code's `load_trials_and_mask` with `max_trial_len=10.0`, which is how `prepare_data` calls it in the reference `0_data_caching.py`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignment for each spike). The cluster table provides quality labels (`label`) and anatomical locations (`acronym`).

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Spike times and cluster IDs are the standard arrays for constructing neural activity matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms non-overlapping bins over the trial window (-0.5 to 1.5s around stimulus onset), producing 100 time bins per trial. The counts are stored as float32 but are **not** divided by the bin width -- the neural data is in units of spike counts per bin, not firing rate in Hz. When a session has multiple probes, their units are merged into a single population.

ii.
```python
def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    out = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
    # ...
    b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
    counts = np.bincount(u * N_BINS + b, minlength=flat)
    out[k] = counts.reshape(n_units, N_BINS)
    return out
```
```python
binned_spikes = bin_spikes(spike_times, spike_units, n_units, interval_begs)
# Note: no division by BINSIZE
```

iii. The AI's metadata describes the neural units as "spike count per 20 ms bin". The reference Zhang code also uses spike counts (via `bincount2D`) and does not divide by bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are kept (well-isolated units). Additionally, units whose Beryl acronym is in `('root', 'void')` are excluded -- both non-brain (`void`) and unassigned (`root`) regions.

ii.
```python
NON_GREY_MATTER = ('root', 'void')
beryl = np.asarray(brain_regions.acronym2acronym(
    clusters['acronym'].to_numpy(), mapping='Beryl'))
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY_MATTER)
```

iii. The AI describes this as "Data-paper inclusion criteria for neurons: well isolated (all three RIGOR single-unit metrics passed, which ibllib encodes as label == 1) and located in grey matter." The code comments state it restricts to grey matter by excluding both `root` and `void`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to stimulus onset (`stimOn_times`). The interval begins at `stimOn_times - 0.5s` and spans 2s (100 bins of 20ms). Spike times within each trial's window are binned relative to the interval start.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs_all = align_times + TIME_WINDOW[0]
# ...
interval_begs = interval_begs_all[keep]
binned_spikes = bin_spikes(spike_times, spike_units, n_units, interval_begs)
```

iii. All streams share the same session clock (IBL synchronization), so alignment is simply subtracting the stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins per 2s trial window. No rebinning is applied; spikes are directly counted into the 20ms bins.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. These are exactly the reference caching parameters from `0_data_caching.py`: `binsize=0.02`, `time_window=(-0.5, 1.5)`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table. The input is the center of each 20ms bin in the window from -0.5 to 1.5s around stimulus onset.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The bin centers represent the time from stimulus onset at the midpoint of each neural bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond constructing the bin center times. The values are the same for every trial: `[-0.49, -0.47, ..., 1.49]` in seconds relative to stimulus onset.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. This is a deterministic grid defined by the binning parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The input time values are the centers of the same bins used for neural data, so they are aligned by construction.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres[None, :]
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A change in `probabilityLeft` marks the start of a new block.

ii.
```python
def trial_number_in_block(trials):
    p_left = trials['probabilityLeft'].to_numpy(dtype=float)
    new_block = np.ones(len(p_left), dtype=bool)
    new_block[1:] = ~(p_left[1:] == p_left[:-1])
    block_id = np.cumsum(new_block) - 1
    block_start = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(p_left)) - block_start[block_id]
    return idx_in_block.astype(np.float32)
```

iii. The trials table carries no explicit block identifier, so blocks must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based position of each trial within its block of constant `probabilityLeft`. Computed on the full trials table before any filtering, so the value reflects the animal's actual position in the block. NaN values in `probabilityLeft` break block continuity (NaN != NaN), but those trials are excluded by the mask.

ii.
```python
n_in_block = trial_number_in_block(trials)[keep]
inputs[:, 1, :] = n_in_block[:, None]
```

iii. The AI's docstring states: "Computed on the *full* trials table, before any trial is excluded, so that the value reflects the animal's actual position in the block rather than a position in the filtered sequence."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table. IBL codes choice as +1 for left and -1 for right.

ii.
```python
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
```

iii. The AI verified the convention: "trials with contrastLeft > 0 and feedbackType == +1 all have choice == +1".

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded to left=0, right=1. The AI uses `(choice == -1)` to get a boolean (0 for left/+1, 1 for right/-1), then casts to int64. No-response trials (choice==0) are already excluded by the trial mask.

ii.
```python
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
outputs[:, 0, :] = choice[:, None]
```

iii. Matches the instruction specification: left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior_all = np.full(len(trials), -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code
```

iii. The three values are the block priors defined by the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoded from continuous values to integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 using `np.isclose` for floating-point safety. Trials with unrecognized values get -1 and are excluded by the mask (`prior_all >= 0`).

ii.
```python
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
```

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The speed is the absolute value of the velocity computed by `SessionLoader`.

ii.
```python
sl.load_wheel()
traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                         np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))
```

iii. The reference derives `wheel-speed` the same way, as `np.abs` of the velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` internally interpolates the raw wheel position onto a 1000 Hz grid and differentiates with a 20 Hz Butterworth low-pass filter. The speed is the absolute value of that velocity. The trace is then linearly interpolated onto the **right edge** of each 20ms bin for each trial (i.e., `beg + (i+1)*BINSIZE`). Finally discretized into 3 levels at the 1/3 and 2/3 quantiles of the session.

ii.
```python
offsets = (np.arange(N_BINS) + 1) * BINSIZE
sample_times = interval_begs[:, None] + offsets[None, :]
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The AI explicitly states this matches the reference `get_behavior_per_interval` which uses `x_interp = linspace(beg + binsize, end, n_bins)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Split at the 1/3 and 2/3 quantiles of the session's values into 3 levels (0=low, 1=medium, 2=high) using `np.digitize`.

ii.
```python
def discretize_tertiles(values):
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not hi > lo:
        hi = np.nextafter(lo, np.inf)
    return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))
```

iii. Per-session tertiles ensure balanced classes and avoid session-dependent biases in absolute values.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sampled at the right edge of each neural bin, which is offset by one bin width from the bin centers used for the `time_from_stimulus_onset` input. Both are defined relative to stimulus onset.

ii.
```python
offsets = (np.arange(N_BINS) + 1) * BINSIZE
sample_times = interval_begs[:, None] + offsets[None, :]
```

iii. The wheel times are on the same session clock as the neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the left camera (`leftCamera.ROIMotionEnergy`) with the right camera as fallback, plus its timestamps (`_ibl_leftCamera.times`).

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl_me = SessionLoader(one=one, eid=eid)
        sl_me.load_motion_energy(views=[view])
        df = sl_me.motion_energy[cam]
        whisker = (df['times'].to_numpy(dtype=np.float64),
                   df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
        break
    except Exception:
        continue
```

iii. Left camera preferred, right camera as fallback, following the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is with no additional filtering. It is linearly interpolated onto the right edge of each 20ms bin (same as wheel speed), then discretized into 3 levels at session-level 1/3 and 2/3 quantiles.

ii.
```python
binned, good = bin_behaviour(t, v, safe_begs)
# ...
whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])
```

iii. Same interpolation and discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: split at session-level 1/3 and 2/3 quantiles into 3 levels using `np.digitize`.

ii.
```python
whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])
```

iii. Same rationale as wheel speed -- per-session tertiles for balanced classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: the trace is interpolated onto the right edge of each neural bin, aligned to stimulus onset.

ii.
```python
# Same bin_behaviour function used for both wheel and whisker
binned, good = bin_behaviour(t, v, safe_begs)
```

iii. Same session clock, same interpolation grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Trials with NaN in key fields are excluded by the trial mask. (2) Trials whose behavioral traces don't cover the decoding window are excluded. (3) Trials with NaN in the interpolated behavioral values are excluded. (4) Sessions with no well-isolated grey-matter units return None. (5) Sessions with fewer than 2 valid trials are skipped. (6) Probe insertions with empty spike sorting are skipped. (7) Spikes with NaN times are excluded. (8) Sessions that raise exceptions are caught and skipped.

ii.
```python
if spike_times is None:
    return None, 'no well-isolated grey-matter units'
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return None, 'too few trials with usable behaviour'
# Spike NaN check:
ok &= np.isfinite(st)
# Exception handling:
except Exception as exc:
    return eid, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
```

iii. The AI's approach is defensive: any session or trial that cannot be fully processed is dropped rather than producing partial or corrupted data.

## 10-a. What are the most time-consuming steps of the code?

i. Reading spike sorting data from disk is the most expensive operation, as probe spike arrays can be hundreds of megabytes each. The `SpikeSortingLoader.load_spike_sorting()` call dominates per-session runtime.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. The cost is mainly file I/O for large binary arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function loops over trials to bin spikes. This could be vectorized by offsetting spike indices across trials. Similarly, `bin_behaviour` vectorizes the interpolation by flattening all sample times into one `np.interp` call, but it still loops implicitly over trials via the reshape. The `bin_spikes` trial loop is the main candidate for vectorization.

ii.
```python
for k in range(n_trials):
    t = spike_times[i0[k]:i1[k]]
    # ... bin spikes for this trial
```

iii. The loop is kept for clarity; each trial requires slicing a different portion of the spike train.

## 10-c. What processing does the code repeat multiple times?

i. A new `SessionLoader` is created for wheel and for whisker motion energy separately (two separate `SessionLoader(one=one, eid=eid)` calls), though they could share one.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
# ...
sl_me = SessionLoader(one=one, eid=eid)
sl_me.load_motion_energy(views=[view])
```

iii. The duplication is minor -- constructing a SessionLoader is cheap compared to I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores `lab`, `n_probes`, `n_trials_recorded` (total trials before filtering), `wheel_speed_thresholds`, and `whisker_motion_energy_thresholds` in session_info metadata, as well as per-session `brain_regions` lists. This metadata is informational and not used by the decoder. The `skipped_sessions` list is also stored. None of these affect the decoder's operation.

ii.
```python
'session_info': [
    {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
     'n_probes': r['n_probes'], 'n_neurons': r['n_units'],
     'n_trials': r['n_trials'], 'n_trials_recorded': r['n_trials_total'],
     'brain_regions': sorted(set(r['acronyms'].tolist())),
     'wheel_speed_thresholds': r['wheel_speed_thresholds'],
     'whisker_motion_energy_thresholds': r['whisker_motion_energy_thresholds']}
    for r in results],
'skipped_sessions': [{'eid': e, 'reason': str(m).splitlines()[0]}
                     for e, m in sorted(skipped)],
```

iii. The extra metadata is useful for debugging and reproducibility but not consumed by the decoder.
