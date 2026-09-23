# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API in `local` mode to access data. Sessions are identified from a `bwm_release.csv` freeze file (the same one the reference code uses), which lists eids, pids, probe names, and subjects. The AI rebuilt the ONE cache tables from disk files (`build_one_cache.py`) because the staged release tables did not index the on-disk dataset revisions. `SessionLoader` loads trials, wheel, and motion energy; `SpikeSortingLoader` loads spike sorting per probe. The reference `load_trials_and_mask` and `merge_probes` functions are imported directly from the reference code.

ii.
```python
BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'

def build_jobs(sample):
    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    jobs = []
    for eid, g in bwm.groupby('eid', sort=True):
        probes = list(zip(g['pid'], g['probe_name']))
        jobs.append((eid, g['subject'].iloc[0], probes))
    jobs.sort(key=lambda j: j[0])
    ...

_ONE = ONE(base_url='https://openalyx.internationalbrainlab.org',
           silent=True, mode='local')
```

iii. The AI chose `bwm_release.csv` because `one.eid2pid` requires a live Alyx connection (unavailable offline), and `bwm_release.csv` is the same freeze file the reference code reads. Cache tables were rebuilt from disk because the staged tables did not index on-disk revisions.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column in `bwm_release.csv`, grouped per eid. At assembly, unique sorted subject names form the `subjects` list, and `subject_idx` maps each session to its subject.

ii.
```python
subjects = sorted({r['subject'] for r in ok})
sub2idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub2idx[r['subject']] for r in ok], dtype=np.int64),
```

iii. Subject information is directly available from the release freeze file.

## 1-c. How are the data split into sessions?

i. Each eid in `bwm_release.csv` is one session. The jobs list has one entry per eid, grouping its probes together. Sessions are processed independently (in parallel via `multiprocessing.Pool`).

ii.
```python
for eid, g in bwm.groupby('eid', sort=True):
    probes = list(zip(g['pid'], g['probe_name']))
    jobs.append((eid, g['subject'].iloc[0], probes))
```

iii. Sessions are already the natural unit of organization in the BWM release.

## 1-d. How are the data split into trials?

i. The trials table (loaded via `SessionLoader.load_trials()`) has one row per trial. The `load_trials_and_mask` function from the reference code produces a boolean mask over this table.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference code's `load_trials_and_mask` function directly with parameters: `min_rt=0.08`, `max_rt=2.0`, `nan_exclude='default'` (excludes trials with NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), `max_trial_len=10.0` (feedback_times - goCue_times <= 10s), `exclude_nochoice=True`. Additionally, trials are dropped if the wheel or whisker motion energy trace does not cover the trial window, and trials whose window falls outside the recorded spike train are also dropped.

ii.
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
    max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
    sess_loader=sl)
...
wheel_vals, wheel_good = bin_behavior(wt, wv, t_begs)
me_vals, me_good = bin_behavior(mt, mv, t_begs)
beh_good = wheel_good & me_good
...
if len(spike_times_k):
    in_rec = ((t_begs >= spike_times_k[0] - BINSIZE)
              & (t_begs + N_BINS * BINSIZE <= spike_times_k[-1] + BINSIZE))
...
valid = beh_good & in_rec
```

iii. The AI directly imported `load_trials_and_mask` from the reference code to ensure consistency. The `max_trial_len=10.0` parameter matches the reference's `prepare_data` call. Behavior coverage checks mirror the reference's `get_behavior_per_interval` rejection criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe, loaded via `SpikeSortingLoader`. The cluster table supplies the quality label and anatomical acronym for filtering. Probes are merged using the reference's `merge_probes` function.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
ssl.download_spike_sorting_object('spikes')
wanted = [f for f in ssl.files['spikes']
          if f.name.split('.')[1] in ('times', 'clusters')]
spikes = dict(ssl._load_object(wanted))
clusters = ssl.load_spike_sorting_object('clusters')
channels = ssl.load_channels()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. The AI loads only the two needed spike attributes (`times` and `clusters`), avoiding the large `depths` and `amps` arrays.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window `[stimOn + T_START, stimOn + T_END)`. The AI stores raw spike counts (float32), NOT firing rates. The reference code divides by BIN to convert to Hz. Probes are merged using the reference's `merge_probes` function.

ii.
```python
binned = bin_spikes(spike_times_k, spike_clusters_k, len(kept_ids), t_begs)
...
# In bin_spikes:
binned = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
...
flat = (trial_id * n_clusters + spike_clusters[idx]) * N_BINS + b
binned.reshape(-1)[:] = np.bincount(
    flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)
return binned
...
'neural': binned,   # spike counts, NOT divided by BIN
```

iii. The AI verified that its vectorized binner produces bit-identical results to the reference's `bincount2D`. Neural data are stored as spike counts rather than firing rates; the metadata states `'neural_units': 'spike counts per 20 ms bin'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (well-isolated neurons) AND Beryl acronym NOT in `('root', 'void')` are kept. This means both `root` and `void` regions are excluded.

ii.
```python
NON_GREY = ('root', 'void')     # excluded from "grey matter" analyses
...
good_unit = (label >= GOOD_UNIT_LABEL) & ~np.isin(beryl, NON_GREY)
```

iii. The AI justifies this as following the data paper's definition of well-isolated neurons in grey matter. The reference code's `MultiRegionDataModule` also drops `root`/`void`. The AI notes this yields 62,763 neurons across the kept sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window starts at `stimOn_times + TIME_WINDOW[0]` (i.e., `stimOn_times - 0.5`). The spikes are binned relative to this window start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]
...
# In bin_spikes:
rel = spike_times[idx] - t_begs[trial_id]
b = np.floor(rel / BINSIZE).astype(np.int64)
```

iii. All streams (spikes, wheel, whisker) are on the same session clock, synchronized upstream by IBL. Alignment to stimulus onset matches the reference code's `align_time='stimOn_times'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, T = 100 time bins over a 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
```

iii. Matches the reference code's `params['binsize'] = 0.02` and the methods paper's "20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table (which defines the alignment event) and the bin grid parameters. The input is the center of each of the 100 bins.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
inputs[:, 0, :] = bin_centers[None, :]
```

iii. The bin centers represent the time from stimulus onset for each time bin.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The input is analytically computed as the centers of the 100 bins: `T_START + (i + 0.5) * BINSIZE` for i = 0..99, giving values from -0.49 to 1.49 s.

ii.
```python
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. This is a deterministic grid, identical for every trial.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centers are the centers of the same bins the spikes are counted into, so they share the same time axis bin-for-bin.

ii.
```python
# Neural binning uses: t_begs + bin_index * BINSIZE
# Input is: TIME_WINDOW[0] + (i + 0.5) * BINSIZE = t_beg - t_beg + bin_center
bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
```

iii. By construction, each bin center corresponds to the center of the bin where spikes are counted.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `trials['probabilityLeft']`. A block boundary is detected where `probabilityLeft` changes value.

ii.
```python
def trial_number_in_block(probability_left):
    pl = np.asarray(probability_left, dtype=float)
    change = np.ones(len(pl), dtype=bool)
    change[1:] = pl[1:] != pl[:-1]
    block_id = np.cumsum(change) - 1
    starts = np.flatnonzero(change)
    return (np.arange(len(pl)) - starts[block_id]).astype(np.float32)
```

iii. The trials table has no explicit block identifier, so blocks are recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Computed on the FULL trials table BEFORE any trial exclusion, so excluded trials still advance the counter. The value is the 0-based index of the trial within its block. It is broadcast across all 100 time bins.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
inputs[:, 1, :] = tib_all[keep][:, None]
```

iii. Computing on the full table ensures the block counter reflects the animal's true position in the block, not an artifact of trial exclusions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials['choice']`, which is +1 (left), -1 (right), or 0 (no-go). Recoded to 0 (left) and 1 (right). No-go trials (choice=0) are excluded by `exclude_nochoice=True`.

ii.
```python
def map_choice(choice):
    out = np.full(len(choice), -1, dtype=np.int64)
    out[np.asarray(choice) == 1] = 0
    out[np.asarray(choice) == -1] = 1
    return out
...
choice = map_choice(trials['choice'].to_numpy()[keep])
outputs[:, 0, :] = choice[:, None]
```

iii. IBL convention: +1 = left, -1 = right. The task spec says left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple remapping of +1 to 0 and -1 to 1. The value is broadcast across all 100 time bins (per-trial but represented as time-varying).

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. No further processing needed.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials['probabilityLeft']`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
def map_prior(probability_left):
    pl = np.asarray(probability_left, dtype=float)
    out = np.full(len(pl), -1, dtype=np.int64)
    out[np.isclose(pl, 0.2)] = 0
    out[np.isclose(pl, 0.5)] = 1
    out[np.isclose(pl, 0.8)] = 2
    return out
...
prior = map_prior(trials['probabilityLeft'].to_numpy()[keep])
outputs[:, 1, :] = prior[:, None]
```

iii. The three values are the block prior specified in the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Remapping 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 using `np.isclose` for floating-point comparison. Broadcast across all 100 time bins.

ii.
```python
out[np.isclose(pl, 0.2)] = 0
out[np.isclose(pl, 0.5)] = 1
out[np.isclose(pl, 0.8)] = 2
```

iii. Using `np.isclose` is more robust than exact equality for floating-point values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The speed is `abs(velocity)`.

ii.
```python
def load_wheel_speed(one, eid):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    return (sl.wheel['times'].to_numpy(),
            np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. Same source as the reference's `load_target_behavior(one, eid, 'wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` interpolates the raw wheel position to 1000 Hz and computes velocity with a 20 Hz Butterworth low-pass filter. Speed = |velocity|. The speed trace is then interpolated onto the per-trial bin grid using `np.interp`, then discretized into 3 per-session tertile bins.

ii.
```python
wheel_vals, wheel_good = bin_behavior(wt, wv, t_begs)
...
# In bin_behavior:
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
...
wheel_cls, wheel_edges = discretize_tertiles(wheel_vals)
```

iii. The interpolation grid uses `np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)`, which evaluates the trace at the RIGHT EDGE of each bin (t_beg + 0.02, t_beg + 0.04, ..., t_beg + 2.0), matching the reference code's `get_behavior_per_interval` which uses `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 classes using per-session tertiles (33.3rd and 66.7th percentiles of all binned values in that session). The function handles degenerate cases where tertile edges are tied.

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])
    if edges[0] == edges[1]:
        # Degenerate handling...
    return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. Per-session tertiles ensure balanced classes (each ~1/3 of values) and handle the fact that wheel speed scale varies between mice.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto `t_beg + np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)`, which are the RIGHT EDGES of the neural bins. This means behavior bin k corresponds to the right edge of neural bin k.

ii.
```python
grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)   # right edges
query = t_begs[good][:, None] + grid[None, :]
```

iii. This matches the reference code's `get_behavior_per_interval`, which uses `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `{left,right}Camera.ROIMotionEnergy` and `_ibl_{left,right}Camera.times`, loaded via `SessionLoader.load_motion_energy()`. Left camera preferred, right camera as fallback.

ii.
```python
def load_whisker_me(one, eid):
    for view in ('left', 'right'):
        try:
            sl = SessionLoader(one=one, eid=eid)
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[f'{view}Camera']
            return (me['times'].to_numpy(),
                    me['whiskerMotionEnergy'].to_numpy(), view)
        except Exception:
            continue
    return None, None, None
```

iii. Same camera preference as the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is with no additional filtering. It is interpolated onto the per-trial bin grid (right edges, same as wheel), then discretized into 3 per-session tertile bins.

ii.
```python
me_vals, me_good = bin_behavior(mt, mv, t_begs)
me_cls, me_edges = discretize_tertiles(me_vals)
outputs[:, 3, :] = me_cls
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical to wheel speed: per-session tertiles (33.3rd and 66.7th percentiles), with degenerate-case handling.

ii.
```python
me_cls, me_edges = discretize_tertiles(me_vals)
```

iii. Per-session because whisker motion energy is in arbitrary camera/ROI-dependent units not comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto the right edges of each neural bin.

ii.
```python
# Same bin_behavior function used for both wheel and whisker
me_vals, me_good = bin_behavior(mt, mv, t_begs)
```

iii. Same alignment as wheel speed, using the reference code's `get_behavior_per_interval` approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Sessions with no whisker video are skipped (14 of 459). (2) Trials where the behavior trace does not cover the window are dropped per `bin_behavior`'s coverage check. (3) Trials whose window falls outside the spike recording are dropped. (4) Sessions with no surviving well-isolated grey-matter units or fewer than 2 valid trials are skipped. (5) Any exception in processing a session is caught and the session is skipped, not aborting the run.

ii.
```python
# Behavior coverage check in bin_behavior:
good = (nonempty
        & np.isfinite(t_begs) & np.isfinite(t_ends)
        & (np.abs(t_begs - first) <= BINSIZE)
        & (np.abs(t_ends - last) <= BINSIZE))

# Session-level skip:
if res['n_units_kept'] == 0:
    res['skip'] = 'no well-isolated grey-matter units'
    return res

# Exception handler:
except Exception as e:
    res['skip'] = f'{type(e).__name__}: {e}'
```

iii. The AI documented 15 skipped sessions (14 no whisker video, 1 no valid trials after coverage checks), 89 dropped trials (2 wheel, 84 whisker, 3 outside recording).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk is the most expensive step (~1.8 s per session). The AI reduced this by loading only `spikes.times` and `spikes.clusters` instead of all spike attributes. The full conversion takes ~70 s with 32 worker processes.

ii.
```python
ssl.download_spike_sorting_object('spikes')
wanted = [f for f in ssl.files['spikes']
          if f.name.split('.')[1] in ('times', 'clusters')]
spikes = dict(ssl._load_object(wanted))
```

iii. The AI provided detailed timing breakdowns: trials+mask 0.2s, behavior 0.6s, spike load 1.8s, spike binning 0.4s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI fully vectorized both spike binning and behavior interpolation. `bin_spikes` uses a single `np.bincount` over all trials of a session instead of per-trial calls. `bin_behavior` uses a single `np.interp` over all trials' query points. No per-trial loops remain.

ii.
```python
# Vectorized spike binning (all trials at once):
flat = (trial_id * n_clusters + spike_clusters[idx]) * N_BINS + b
binned.reshape(-1)[:] = np.bincount(
    flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)

# Vectorized behavior interpolation:
query = t_begs[good][:, None] + grid[None, :]
interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
```

iii. The AI documented ~100x speedup on binning and ~50x on behavior interpolation compared to the reference code's per-trial approach.

## 10-c. What processing does the code repeat multiple times?

i. `SessionLoader` is instantiated multiple times per session: once in `convert_session` for trials, once in `load_wheel_speed`, once in `load_whisker_me`, and potentially again in `plot_processing`. The trials loading is not repeated (only done once), but the SessionLoader construction overhead is repeated.

ii.
```python
# In convert_session:
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()

# In load_wheel_speed (separate function):
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()

# In load_whisker_me (another separate function):
sl = SessionLoader(one=one, eid=eid)
sl.load_motion_energy(views=[view])
```

iii. This is a minor inefficiency; a single SessionLoader could load all three.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `wheel_raw`, `me_raw`, `t_begs`, `wheel_edges`, `me_edges`, and `trial_idx` in each session result for diagnostic purposes (plotting and metadata). These intermediate arrays are not part of the final pickle's neural/input/output structure, though some end up in the metadata's `session_info`. The reference code's `load_trials_and_mask` also computes a full trials table with all columns when only a few are needed.

ii.
```python
res.update({
    ...
    'wheel_raw': wheel_vals.astype(np.float32),
    'me_raw': me_vals.astype(np.float32),
    't_begs': t_begs,
    'wheel_edges': wheel_edges,
    'me_edges': me_edges,
    'trial_idx': keep,
    ...
})
```

iii. The raw behavior values and edges are used for `--show-processing` plots and stored in metadata for reproducibility/auditability.
