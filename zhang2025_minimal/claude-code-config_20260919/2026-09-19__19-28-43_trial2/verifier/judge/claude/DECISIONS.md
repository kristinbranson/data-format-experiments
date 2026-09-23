# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the session list from `bwm_release.csv` (the brain-wide map release freeze CSV shipped with the reference code), which lists every released session along with its probe insertion IDs and probe names. When the datalimit subset file exists (`DATALIMIT_SUBSET.csv`), sessions are restricted to those eids. A ONE client is created pointing at the staged `one_cache` directory (no network access to Alyx needed beyond the initial token). For each session, a `SessionLoader` loads trials, wheel, and motion energy, and a `SpikeSortingLoader` loads spikes per probe.

ii.
```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET = '/app/data/DATALIMIT_SUBSET.csv'

def session_list(args):
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    if os.path.exists(DATALIMIT_SUBSET):
        subset = pd.read_csv(DATALIMIT_SUBSET)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
    sessions = []
    for eid, rows in bwm_df.groupby('eid', sort=False):
        sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                         list(rows.pid), list(rows.probe_name)))
    return sessions
```

iii. The AI chose `bwm_release.csv` because it already lists the 459 sessions of the BWM release freeze with their probe IDs, avoiding the need to search the ONE index and call `eid2pid`. The agent stated: "probe IDs come from the BWM release freeze `bwm_release.csv`. All 459 released sessions were attempted."

## 1-b. How are the data split into subjects?

i. The subject name is read directly from the `bwm_release.csv` table alongside each eid. At assembly, unique subjects are sorted and `subject_idx` maps each session to its subject index.

ii.
```python
sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                 list(rows.pid), list(rows.probe_name)))
# ...
subjects = sorted({r['subject'] for r in results})
subject_lookup = {s: i for i, s in enumerate(subjects)}
```

iii. The subject is already a column in the CSV, so no parsing is needed.

## 1-c. How are the data split into sessions?

i. A session is the unit the release is organized by. The CSV is grouped by `eid` to produce one entry per session (with potentially multiple probes).

ii.
```python
for eid, rows in bwm_df.groupby('eid', sort=False):
    sessions.append((eid, rows.subject.iloc[0], rows.lab.iloc[0],
                     list(rows.pid), list(rows.probe_name)))
```

iii. Sessions are already distinct eids in the release CSV.

## 1-d. How are the data split into trials?

i. The trials table has one row per trial. `SessionLoader.load_trials()` provides the full trials DataFrame.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, trials_mask = load_trials_and_mask(sess_loader)
```

iii. No decision needed; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI re-implements `load_trials_and_mask` from the reference code (`code_zhang2025/src/utils/ibl_data_utils.py`) with the arguments used by `prepare_data`: reaction time between 0.08 s and 2.0 s, `feedback_times - goCue_times <= 10 s`, no NaN in six trial event columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`), and `choice != 0` (exclude no-choice trials). Additionally, trials where wheel speed or whisker motion energy traces do not cover the trial window or contain NaN are dropped.

ii.
```python
def load_trials_and_mask(sess_loader):
    trials = sess_loader.trials
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()
```

```python
keep = trials_mask & wheel_ok & whisker_ok
```

iii. The AI stated it followed "the `load_trials_and_mask` criteria used by `prepare_data`" from the reference code, including the `max_trial_len=10` filter on feedback time minus go cue time, which the human reference does not include.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignment for each spike), loaded via `SpikeSortingLoader`. The cluster table provides anatomical labels (acronyms) mapped to the Beryl atlas.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. Spike times and cluster IDs are directly available from the spike sorting loader.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset), giving one count per unit per bin. Counts are stored as-is (uint16 cast to float32); they are **not** divided by the bin width, so the neural data is in spike counts, not firing rates. When a session has multiple probes, their units are merged (re-indexed continuously) and sorted by time, following the reference code's `merge_probes`. Only units that fired at least one spike in any trial are included in the output.

ii.
```python
def bin_spikes(spike_times, spike_clusters, interval_begs, interval_ends):
    unit_ids = np.unique(spike_clusters)
    n_units = len(unit_ids)
    binned = np.zeros((n_trials, n_units, N_BINS), dtype=np.uint16)
    # ...
    bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
    counts = np.bincount(rows * N_BINS + bin_idx[keep], minlength=n_units * N_BINS)
    binned[trial] = counts.reshape(n_units, N_BINS)
    return binned, unit_ids
```

```python
neural.append(binned[i].astype(np.float32))
```

iii. The agent stated: "spike counts are binned 'using all neurons, sorted by Kilosort, from each session'" following the methods paper. The agent verified: "spike binning identical: True" against the reference code's `get_spike_data_per_interval`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality control filtering is applied.** The AI uses `qc=None` (the default of `load_spiking_data` in the reference code), keeping all spike-sorted units regardless of their quality label. Units that never fire are dropped by `np.unique(spike_clusters)`. Void brain regions are **not** filtered out.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
# No label filtering; all clusters are kept
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

```python
unit_ids = np.unique(spike_clusters)  # only units that fired
```

iii. The agent justified this explicitly: "`prepare_data` -> `load_spiking_data` in the methods-paper code runs with `qc=None`, and the paper states spike counts are binned 'using all neurons, sorted by Kilosort, from each session'. This is the one place the data paper differs (it restricts its analyses to RIGOR well-isolated units); I followed the decoding pipeline being replicated."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The interval for each trial is `[stimOn_times + TIME_WINDOW[0], stimOn_times + TIME_WINDOW[1]]`, i.e. -0.5 to 1.5 s from stimulus onset. Spikes within this interval are binned relative to the interval start.

ii.
```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs = align_times_filled + TIME_WINDOW[0]
interval_ends = align_times_filled + TIME_WINDOW[1]
# ...
bin_idx = np.floor((spike_times[sl] - interval_begs[trial]) / BINSIZE).astype(np.int64)
```

iii. All timestamps share one session clock, so subtracting the onset aligns the data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The 2 s window is divided into 100 bins of 20 ms each. No rebinning or smoothing is applied.

ii.
```python
BINSIZE = 0.02              # seconds
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The agent noted this matches the `params` dict of `src/0_data_caching.py` and the methods paper's description of "20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which is the alignment event. The input values are the bin centres of the 100 bins spanning -0.5 to 1.5 s.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
# ...
inp[0] = bin_centres
```

iii. The bin centres define the time axis of the decoding window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing; the bin centres are computed directly from the window and bin size parameters.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centres correspond to the centres of the same bins the spikes are counted in, so they share the same time axis.

ii.
```python
bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in its value marks the start of a new block.

ii.
```python
def trial_number_in_block(trials):
    pleft = trials['probabilityLeft'].to_numpy()
    new_block = np.ones(len(pleft), dtype=bool)
    new_block[1:] = pleft[1:] != pleft[:-1]
    block_id = np.cumsum(new_block) - 1
    block_starts = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(pleft)) - block_starts[block_id]
    return idx_in_block.astype(np.float64)
```

iii. The trials table has no block identifier, so blocks are recovered from changes in the prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based index of each trial within its block is computed on the full (unfiltered) trials table, so excluded trials still advance the count. The result is the animal's real position in the block.

ii.
```python
in_block = trial_number_in_block(trials)[keep]
```

iii. The agent's docstring states: "Computed on the full (unfiltered) trials table so that the count reflects the animal's actual position in the block, independent of the trial exclusions."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
```

iii. IBL codes +1 for left choice and -1 for right choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded to 0 for left (was +1) and 1 for right (was -1). No-response trials (choice == 0) are excluded by the trial mask.

ii.
```python
choice = np.where(trials['choice'].to_numpy()[keep] > 0, 0, 1).astype(np.int64)
```

iii. The instructions specify left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[keep]
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int64)
```

iii. The three values are the block priors the task holds constant within a block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped to categorical: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The AI uses `np.isclose` for floating-point comparison and `np.select` rather than a dictionary map.

ii.
```python
prior = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                   np.isclose(pleft, 0.8)], [0, 1, 2], default=-1).astype(np.int64)
```

iii. The instructions give this mapping directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps loaded by `SessionLoader.load_wheel()`. The velocity is computed internally by `SessionLoader` (Gaussian-smoothed differentiation), and the speed is the absolute value of that velocity.

ii.
```python
def load_behavior_trace(sess_loader, name):
    if name == 'wheel-speed':
        if sess_loader.wheel.empty:
            sess_loader.load_wheel()
        return (sess_loader.wheel['times'].to_numpy(),
                np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The reference code derives wheel-speed as `np.abs` of the velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader` interpolates wheel position onto an even grid and differentiates with a Butterworth low-pass filter to get velocity; (2) the absolute velocity trace is interpolated at the **right edge** of each of the 100 bins for each trial (using `scipy.interpolate.interp1d` with linear interpolation and `fill_value='extrapolate'`); (3) discretized into 3 classes at the within-session tertiles.

ii.
```python
def bin_behavior(target_times, target_vals, interval_begs, interval_ends):
    x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
    y = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
    binned[trial] = y
    good[trial] = True
    return binned, good
```

```python
def discretize(values, n_classes=N_BEHAVIOR_CLASSES):
    quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
    edges = np.maximum.accumulate(quantiles)
    return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. The agent stated this "exactly reproducing `get_behavior_per_interval`" and verified "behavior mask match: True values close: True" against the reference code. The right-edge interpolation matches the methods paper's code.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 classes at the within-session tertiles (quantiles at 1/3 and 2/3 of all wheel-speed values in the session). The AI adds `np.maximum.accumulate` to ensure strictly increasing bin edges when there are ties (e.g., many zero-speed samples).

ii.
```python
def discretize(values, n_classes=N_BEHAVIOR_CLASSES):
    quantiles = np.quantile(values, np.arange(1, n_classes) / n_classes)
    edges = np.maximum.accumulate(quantiles)
    return np.digitize(values, edges, right=False).astype(np.int64), edges
```

iii. Session-wise boundaries keep labels comparable across sessions since absolute values differ between animals and cameras.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated at the **right edge** of each 20 ms bin, while the neural data represents spike counts in each 20 ms bin. The bin edges are `[interval_beg + BINSIZE, interval_beg + 2*BINSIZE, ..., interval_end]`, so the behavioral value represents the instantaneous speed at the end of each bin.

ii.
```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
```

iii. The agent followed the reference code's `get_behavior_per_interval` which samples at right edges. The agent verified its implementation against the reference code.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy of the left camera (`leftCamera.ROIMotionEnergy`), falling back to the right camera when the left is unavailable. Loaded via `SessionLoader.load_motion_energy`.

ii.
```python
def load_behavior_trace(sess_loader, name):
    if name == 'whisker-motion-energy':
        for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
            try:
                sess_loader.load_motion_energy(views=[view])
                me = sess_loader.motion_energy[key]
                return (me['times'].to_numpy(),
                        me['whiskerMotionEnergy'].to_numpy())
            except Exception:
                continue
```

iii. Left camera preferred, right as fallback, following the reference code's logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is with no filtering or normalization. It is interpolated at the **right edge** of each 20 ms bin (same as wheel speed) and then discretized into 3 classes at within-session tertiles.

ii.
```python
whisker_times, whisker_vals = load_behavior_trace(sess_loader, 'whisker-motion-energy')
whisker, whisker_ok = bin_behavior(whisker_times, whisker_vals,
                                   interval_begs, interval_ends)
whisker_class, whisker_edges = discretize(whisker[keep])
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 classes at the within-session tertiles using `np.quantile` and `np.digitize`.

ii.
```python
whisker_class, whisker_edges = discretize(whisker[keep])
```

iii. Same justification as wheel speed (session-wise boundaries for comparability).

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at the right edge of each 20 ms bin, so the behavioral value represents the instantaneous motion energy at the end of each bin.

ii.
```python
x = np.linspace(interval_begs[trial] + BINSIZE, interval_ends[trial], N_BINS)
```

iii. Follows `get_behavior_per_interval` from the reference code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) NaN alignment times are replaced with 0.0 so searchsorted/interpolation stay well-defined (those trials are already excluded by the mask). (2) Trials whose wheel or whisker traces don't cover the trial window, contain NaN, or have no samples are marked as not-ok and dropped. (3) Sessions with fewer than 2 surviving trials raise a RuntimeError and are recorded as failures. (4) The AI explicitly disallows NaN in the output (unlike the reference code which uses `allow_nans=True`), dropping affected trials instead.

ii.
```python
align_times_filled = np.where(np.isnan(align_times), 0.0, align_times)
```

```python
if len(v) == 0:
    continue
if np.abs(interval_begs[trial] - t[0]) > BINSIZE:
    continue
if np.abs(interval_ends[trial] - t[-1]) > BINSIZE:
    continue
if np.any(~np.isfinite(y)):
    continue
```

```python
if keep.sum() < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(f'only {keep.sum()} trials left after filtering')
```

iii. The agent noted: "the reference code runs with `allow_nans=True` and then ... ignores the behaviour masks entirely; NaNs are not allowed in the target format, so trials whose wheel/whisker traces are missing, short, or NaN are dropped. 15 sessions were excluded this way."

## 10-a. What are the most time-consuming steps of the code?

i. Loading and merging spike sorting data from disk, which involves reading hundreds of megabytes of spike arrays per probe. The `load_spike_sorting()` call and subsequent `merge_clusters` are the dominant costs.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. The cost is dominated by file I/O for the large spike arrays.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: one in `bin_spikes` that iterates over trials to bin spikes, and one in `bin_behavior` that iterates over trials to interpolate behavioral traces. Both could potentially be vectorized by offsetting indices across trials.

ii.
```python
for trial in range(n_trials):
    sl = slice(i_beg[trial], i_end[trial])
    # ...
    binned[trial] = counts.reshape(n_units, N_BINS)
```

```python
for trial in range(n_trials):
    t = target_times[idxs_beg[trial]:idxs_end[trial]]
    # ...
    binned[trial] = y
```

iii. Both loops operate on variable-length slices per trial, making full vectorization non-trivial. The cost is small relative to I/O.

## 10-c. What processing does the code repeat multiple times?

i. A `BrainRegions()` instance is created inside `process_session` for each session, and the ONE client is re-created for each session via `get_one()`. The BrainRegions initialization involves loading atlas data.

ii.
```python
from iblatlas.regions import BrainRegions
beryl = BrainRegions().acronym2acronym(acronyms, mapping='Beryl')
```

```python
one = get_one()
```

iii. Each session is processed independently (possibly in parallel), so some repeated initialization is expected.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads all spike-sorted units without QC filtering, resulting in ~600k units versus ~76k with QC. Many of these poorly isolated units add noise rather than signal. The Beryl region mapping is computed for all units including those in `void` regions outside the brain. The code also computes `merge_clusters` with full DataFrame conversion for each probe, though only the acronym column is used downstream.

ii.
```python
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
# Only acronym is used later:
acronyms = clusters['acronym'].to_numpy()[unit_ids]
```

iii. The agent chose to keep all units to match the methods paper's approach, accepting the overhead.
