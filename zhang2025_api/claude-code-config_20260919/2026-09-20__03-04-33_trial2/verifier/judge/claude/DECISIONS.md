# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) shipped with the reference code to get the list of session EIDs. It then uses the ONE API (`ONE(base_url=..., silent=True, cache_dir=...)`) in remote mode against a local cache. For each session, it calls `SessionLoader` for trials/wheel/motion-energy and `SpikeSortingLoader` for spikes/clusters. It also directly imports and calls the reference code's `load_trials_and_mask` and `merge_probes` functions from `utils.ibl_data_utils`.

ii.
```python
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
...
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(pd.unique(bwm_df.eid))
...
one = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
```

```python
from utils.ibl_data_utils import load_trials_and_mask, merge_probes
```

iii. The AI chose to reuse the reference code's own session list (`bwm_release.csv`) and its `load_trials_and_mask` / `merge_probes` utilities directly. This ensures consistency with the reference pipeline. The ONE API is used in remote mode because the local release table doesn't resolve the dataset revisions staged on disk.

## 1-b. How are the data split into subjects?

i. Subject names are extracted from the `bwm_release.csv` dataframe, which maps each EID to its subject. After conversion, subjects are sorted and indexed.

ii.
```python
subject_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
...
subjects = sorted({subject_of_eid[res['eid']] for res in results})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The subject information comes directly from the release CSV, avoiding any need to parse paths or query the API separately.

## 1-c. How are the data split into sessions?

i. Sessions are already the unit of the release. The list of unique EIDs from `bwm_release.csv` defines the sessions. Each EID is processed independently.

ii.
```python
eids = list(pd.unique(bwm_df.eid))
```

iii. No splitting needed; sessions are inherent to the data organization.

## 1-d. How are the data split into trials?

i. Trials come from the trials table loaded by `SessionLoader.load_trials()` (called internally by `load_trials_and_mask`), which has one row per trial.

ii.
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```

iii. The trials table is inherently one-row-per-trial. No additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference code's `load_trials_and_mask(max_trial_len=10.0)` which excludes: NaN in key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), reaction time outside [0.08, 2.0] s, choice == 0, and feedback_times - goCue_times > 10 s. Additionally, the AI applies its own filters: `wheel_ok` and `me_ok` (behavioral coverage), and `has_neural` (neural data gap detection).

ii.
```python
trials, mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
...
has_neural = neural_data_available(all_spike_times, align_times)
keep_trials = mask & wheel_ok & me_ok & has_neural
```

iii. The AI documents that this mirrors `align_spike_behavior` from the reference: keep trials that pass the trials mask AND have valid behavioral traces. The neural gap check is the AI's own addition to handle sessions with recording dropouts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from `SpikeSortingLoader.load_spike_sorting()`, plus the cluster table for quality labels and anatomical locations.

ii.
```python
ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
sp, cl, ch = ssl.load_spike_sorting()
...
cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```

iii. Standard IBL spike sorting data loaded through the brainbox API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s), giving spike counts per neuron per bin. The counts are stored as uint8 (not converted to firing rates). When a session has multiple probes, neurons are pooled using the reference's `merge_probes` function.

ii.
```python
binned = bin_spiking_data_fast(spike_times, spike_clusters, n_neurons,
                               align_times[keep_trials])
...
out = np.zeros((len(align_times), n_clusters, NBINS), dtype=np.uint8)
...
counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
np.clip(counts, 0, 255, out=counts)
out[k] = counts.reshape(n_clusters, NBINS).astype(np.uint8)
```

iii. The AI chose to store raw spike counts (as the reference cached dataset does) rather than converting to firing rates. The reference code's cached HF dataset stores "raw spike counts as np.ubyte". However, the final output is cast to float32 when assembling: `np.ascontiguousarray(neural[k], dtype=np.float32)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` ("well-isolated neurons") are kept. Additionally, clusters with Beryl acronym in `('root', 'void')` are excluded (designated as non-grey-matter).

ii.
```python
NON_GREY_MATTER = ('root', 'void')
...
good = clusters['label'].to_numpy() >= QC_LABEL
grey = ~np.isin(beryl, NON_GREY_MATTER)
keep = np.nonzero(good & grey)[0]
```

iii. The AI documents this as implementing the data paper's restriction to "grey matter" regions. Both `root` and `void` are excluded because they represent recording sites not in designated grey matter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. For each trial, spikes within the window [stimOn - 0.5, stimOn + 1.5] are selected and binned relative to stimOn_times.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align_times = trials[ALIGN_TIME].to_numpy()
...
beg = align_times + TIME_WINDOW[0]
end = align_times + TIME_WINDOW[1]
...
b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
```

iii. All clocks are already synchronized in IBL data, so alignment is simply subtracting the stimulus onset time from spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, producing 100 time bins over the 2 s window. No rebinning or interpolation is applied to the neural data.

ii.
```python
BINSIZE = 0.02
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # -> 100
```

iii. Matches the reference code parameter `'binsize': 0.02` and the method paper description of "T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table (the alignment event). The input is the centre of each of the 100 bins relative to stimulus onset.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
...
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. Defined by the binning grid; same values for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing; it's the bin centres of the time grid, computed as `TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)`.

ii.
```python
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
```

iii. A deterministic function of the binning parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centres are the temporal midpoints of the same bins used for spike counting, so they are inherently aligned.

ii.
```python
inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
```

iii. By construction, the time input and neural data share the same temporal grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` marks a new block boundary.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        new_block[1:] = ~(p[1:] == p[:-1])
    block_id = np.cumsum(new_block) - 1
    starts = np.nonzero(new_block)[0]
    within = np.arange(len(p)) - starts[block_id]
    return within.astype(np.float32), block_id
```

iii. The trials table has no explicit block identifier, so blocks are recovered from where `probabilityLeft` changes.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The 0-based index of each trial within its block is computed on the complete trials table before any trial exclusion, so the numbering reflects the actual block structure the mouse experienced.

ii.
```python
in_block, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
inp[:, 1, :] = in_block[keep_trials][:, None]
```

iii. Computing trial-in-block before filtering preserves the real block position. The value is broadcast across all time bins (per-trial constant).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, where IBL codes +1 for leftward and -1 for rightward choice.

ii.
```python
choice = tr['choice'].to_numpy()
choice_out = (choice < 0).astype(np.int64)   # left -> 0, right -> 1
```

iii. Verified empirically that +1 is leftward by checking correct trials with left stimuli.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoding: `choice == +1` (left) maps to 0, `choice == -1` (right) maps to 1. No-response trials (choice == 0) are already excluded by the trials mask.

ii.
```python
choice_out = (choice < 0).astype(np.int64)
```

iii. Simple binary recoding as specified in the instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pleft = tr['probabilityLeft'].to_numpy()
prior_out = np.full(n_keep, -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_out[np.isclose(pleft, value)] = code
```

iii. The three values are mapped to categorical codes as specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapping: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Uses `np.isclose` for floating-point comparison. Raises an error if any unexpected values are found.

ii.
```python
if np.any(prior_out < 0):
    bad = np.unique(pleft[prior_out < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
```

iii. Includes validation that all values are one of the three expected values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`), loaded via `SessionLoader.load_wheel()`. The speed is the absolute value of the velocity.

ii.
```python
sess_loader.load_wheel()
out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                      np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. `SessionLoader` internally interpolates the raw wheel position onto a ~1 kHz grid and computes velocity with a Butterworth low-pass filter.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader.load_wheel()` interpolates position to ~1 kHz and computes velocity with a 20 Hz Butterworth filter; (2) `|velocity|` gives speed; (3) the speed trace is linearly interpolated onto the right edge of each 20 ms bin via `interp1d(..., kind='linear', fill_value='extrapolate')` evaluated at `np.linspace(beg + binsize, end, n_bins)`.

ii.
```python
def bin_behaviour_per_trial(target_times, target_vals, align_times):
    ...
    x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
    vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
    ...
```

iii. The AI explicitly follows the reference `get_behavior_per_interval` function, which samples at the right edge of each bin.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertile discretization: the 33.3rd and 66.7th percentiles of all retained (trial, time-bin) samples of that session are used as thresholds. Values are classified into 3 bins (0=low, 1=medium, 2=high) using `np.searchsorted(thresholds, values, side='right')`.

ii.
```python
def discretize_tertiles(values):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
    labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
    return labels, thresholds
```

iii. Per-session thresholds are used because the signals are in session-specific units; a global threshold would map whole sessions into one class.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sampled at the right edge of each 20 ms bin (`np.linspace(beg + binsize, end, n_bins)`), which means the behavior at time-bin k corresponds to the interval `[beg + k*binsize, beg + (k+1)*binsize]`. This is slightly different from the neural bin centres but follows the reference code's `get_behavior_per_interval`.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)
...
x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
```

iii. Follows the reference code's convention of sampling behavior at the right edge of each bin.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `ROIMotionEnergy` from the side camera (left preferred, right as fallback), with corresponding `_ibl_<side>Camera.times`. The motion energy is the `whiskerMotionEnergy` column.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        times = me['times'].to_numpy()
        vals = me['whiskerMotionEnergy'].to_numpy()
```

iii. Left camera is preferred (60 Hz); right camera (150 Hz) is used as fallback, matching the reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is resampled onto the right edge of each 20 ms bin using `interp1d`, then discretized into 3 per-session tertile bins.

ii.
```python
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
...
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical to wheel speed: per-session 33.3rd and 66.7th percentile thresholds, producing 3 classes (0=low, 1=medium, 2=high).

ii.
```python
me_lab, me_thr = discretize_tertiles(me_kept)
```

iii. Same rationale: session-specific units require per-session normalization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: sampled at the right edge of each 20 ms bin, following the reference `get_behavior_per_interval`.

ii.
```python
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
```

iii. Follows the reference convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) If a probe's spike sorting fails to load, it's skipped. (2) `load_trials_and_mask` excludes trials with NaN in key columns. (3) `bin_behaviour_per_trial` rejects trials where behavioral data doesn't span the window or contains NaN. (4) `neural_data_available` rejects trials overlapping gaps in the spike train (>0.5 s inter-spike interval). (5) Sessions with fewer than `MIN_NEURONS=5` well-isolated neurons or `MIN_TRIALS=2` usable trials raise errors and are skipped. (6) Entire sessions that fail for any reason are logged and excluded.

ii.
```python
if n_neurons < MIN_NEURONS:
    raise RuntimeError(...)
...
keep_trials = mask & wheel_ok & me_ok & has_neural
n_keep = int(keep_trials.sum())
if n_keep < MIN_TRIALS:
    raise RuntimeError(...)
```

iii. The AI's approach is thorough, handling missing data at every level from individual trials to entire sessions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading the spike sorting from disk (`SpikeSortingLoader.load_spike_sorting()`), which involves reading large spike arrays. The AI uses `ProcessPoolExecutor` with up to 24 workers to parallelize session processing.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
...
with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futures = {pool.submit(_worker, eid): eid for eid in remaining}
```

iii. File I/O for spike data dominates; parallelization mitigates this.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: `bin_spiking_data_fast` iterates over trials to bin spikes, and `bin_behaviour_per_trial` iterates over trials to interpolate behavioral traces. Both could potentially be vectorized.

ii.
```python
for k in range(len(align_times)):  # in bin_spiking_data_fast
    ...
for k in range(n):  # in bin_behaviour_per_trial
    ...
```

iii. The per-trial loops are kept because each trial operates on a different slice of the data. The cost is modest relative to I/O.

## 10-c. What processing does the code repeat multiple times?

i. The `bin_behaviour_per_trial` function is called twice with the same structure (once for wheel speed, once for whisker motion energy), performing the same interpolation logic. The `searchsorted` to find trial boundaries is computed separately in `bin_spiking_data_fast` and `bin_behaviour_per_trial`.

ii.
```python
wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
```

iii. The repetition is minimal and the function is reusable.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive metadata per session (lab, date, fraction correct, wheel/ME thresholds, timing info, cluster counts) that is stored in metadata but not used by the decoder. It also computes `all_spike_times` from the full unfiltered spike train for the neural gap detection, which requires loading and sorting all spikes regardless of quality.

ii.
```python
all_spike_times = spikes['times']  # pooled, unfiltered: used for data-gap test
...
'frac_correct': float((trials['feedbackType'].to_numpy() == 1).mean()),
'timing': timing,
```

iii. The extra metadata is useful for documentation and debugging but adds processing overhead.
