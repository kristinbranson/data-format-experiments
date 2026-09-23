# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API exclusively, never reading files directly. It discovers sessions from a `bwm_release.csv` freeze file shipped with the reference code (listing 459 eids), rather than querying the ONE search index with dataset requirements. For each eid, it loads trials via `load_trials_and_mask` (imported from the reference code), wheel via `SessionLoader.load_wheel`, whisker motion energy via `SessionLoader.load_motion_energy`, and spikes via `SpikeSortingLoader.load_spike_sorting` plus `merge_probes` (also imported from the reference code).

ii.
```python
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
```
```python
sess_loader, trials, trials_mask = load_trials(one, eid)
beh = load_continuous_behaviors(one, eid, sess_loader)
spikes, clusters = load_session_spikes(one, eid, probe_rows)
```

iii. The AI chose to start from the BWM freeze CSV (which lists all 459 released eids with their probe insertions) rather than using `one.search` with dataset requirements. This provides the probe-to-session mapping directly and avoids needing to separately check for the presence of each required dataset file. The AI documented this in CONVERSION_NOTES.md Step 6.

## 1-b. How are the data split into subjects?

i. Subject names are taken from the `subject` column of `bwm_release.csv`. At assembly, subjects are the sorted unique names across all converted sessions, and `subject_idx` maps each session to its index in that list.

ii.
```python
rows = bwm[bwm.eid == eid]
subject = rows.subject.iloc[0]
```
```python
subjects = sorted({r['subject'] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
```

iii. The subject identity comes directly from the freeze file; no path parsing is needed.

## 1-c. How are the data split into sessions?

i. Each eid in the freeze file is one session. The code iterates over unique eids. Sessions are processed in parallel and then sorted back into freeze order.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
```

iii. Sessions are already the unit of organization in the BWM release.

## 1-d. How are the data split into trials?

i. The trials table has one row per trial. The code loads it via `load_trials_and_mask`, which returns the full table plus a boolean mask.

ii.
```python
sess_loader, trials, trials_mask = load_trials(one, eid)
```

iii. No splitting is needed; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference code's `load_trials_and_mask` function (imported directly from `ibl_data_utils.py`), which applies: no NaN in key trial event times (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`), reaction time in [0.08, 2.0] s, `feedback_times - goCue_times <= 10 s`, and `choice != 0`. Beyond that, the AI additionally drops trials where the continuous behavior (wheel speed or whisker motion energy) window lacks coverage or contains NaN (via `interpolate_behavior`). Finally, trials where the entire neural population records zero spikes are dropped (16 trials total).

ii.
```python
from utils.ibl_data_utils import load_trials_and_mask
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                    sess_loader=sess_loader)
```
```python
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs, ...)
keep_trial &= beh_good[name]
```
```python
has_spikes = binned.any(axis=(1, 2))
# ... drops zero-spike trials
```

iii. The AI explicitly imported and called the reference code's own trial mask function with the same arguments. Additional behavior-coverage checks mirror the reference's `get_behavior_per_interval` skip rules. Zero-spike trial dropping was added after the first full run produced 38 warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe, loaded via `SpikeSortingLoader.load_spike_sorting`. The cluster table provides quality labels and anatomical locations for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
```

iii. Same spike arrays as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms non-overlapping bins over the 2 s trial window (stimOn + [-0.5, 1.5)), giving one count per unit per bin. The counts are stored as **raw spike counts** (float32), NOT divided by the bin width. When a session has multiple probes, their units are merged via the reference code's `merge_probes` function. No smoothing, z-scoring, or other normalization is applied.

ii.
```python
binned = bin_spikes(spike_times, spike_cl, n_neurons, interval_begs,
                    PARAMS['binsize'], N_BINS)
# ...
'neural': [binned[k] for k in range(n_trials)],
```
The `bin_spikes` function:
```python
rel = spike_times[i0:i1] - interval_begs[k]
bin_idx = np.floor(rel / binsize).astype(np.int64)
np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
flat = spike_clusters[i0:i1].astype(np.int64) * n_bins + bin_idx
counts = np.bincount(flat, minlength=n_neurons * n_bins)
out[k] = counts.reshape(n_neurons, n_bins)
```

iii. The AI noted that the reference Zhang et al. code stores raw spike counts (as ubyte CSR), and that z-scoring happens at model-fit time. The metadata says `neural_units: 'spike counts per 20 ms bin'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` (IBL well-isolated neurons) are kept. Additionally, clusters whose Beryl acronym is in `{'root', 'void'}` are dropped. Sessions with fewer than 5 surviving neurons (`MIN_NEURONS_PER_SESSION = 5`) are skipped entirely.

ii.
```python
NON_REGION_ACRONYMS = ('root', 'void')
MIN_NEURONS_PER_SESSION = 5
```
```python
beryl_all = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
good = clusters['label'].to_numpy() >= 1
in_brain = ~np.isin(beryl_all, NON_REGION_ACRONYMS)
keep = good & in_brain
```
```python
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'error': f'only {n_neurons} well-isolated grey-matter neurons ...'}
```

iii. The AI justified `label >= 1` as the data paper's definition of well-isolated neurons. Dropping both `root` and `void` was justified because "root is not a brain region" and void means outside the brain. The minimum 5 neurons per session was adopted from the data paper's per-region floor and also resolved the all-zero-neural warnings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times` with a window of [-0.5, +1.5] s. Spike times are selected within this window using `np.searchsorted` on the interval beginning (`stimOn + time_window[0]`), and the relative time within the window is computed by subtracting the interval start.

ii.
```python
PARAMS = {'align_time': 'stimOn_times', 'time_window': (-0.5, 1.5)}
interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]
```
```python
rel = spike_times[i0:i1] - interval_begs[k]
bin_idx = np.floor(rel / binsize).astype(np.int64)
```

iii. This matches both the reference code parameters and the Decoder Task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial. No rebinning or interpolation is applied.

ii.
```python
PARAMS = {'interval_len': 2.0, 'binsize': 0.02}
N_BINS = int(np.ceil(PARAMS['interval_len'] / PARAMS['binsize']))  # 100
```

iii. Matches the reference code exactly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin grid definition, specifically the right edge of each 20 ms spike bin relative to stimulus onset. The variable is `stimOn_times` (for the alignment event) combined with the bin parameters.

ii.
```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
inputs[:, 0, :] = bin_times[None, :]
```

iii. The AI used the right edge of each bin, consistent with how the reference code's `get_behavior_per_interval` defines its interpolation grid: `np.linspace(t_beg + binsize, t_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The time values are computed from constants: `time_window[0] + binsize * arange(1, 101)`, giving values from -0.48 to 1.50 s (right bin edges).

ii.
```python
bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
```

iii. The AI chose right bin edges to match the behavior interpolation grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input represents the right edge of each spike bin. The neural data is binned at these same bins. The continuous behaviors are interpolated onto these same right edges. So the time input, neural data, and behavior outputs all share the same temporal grid, but the time values correspond to the right edge rather than the center of each bin.

ii.
```python
# Neural bins: [t_beg + k*binsize, t_beg + (k+1)*binsize)
# Time input: t_beg + (k+1)*binsize  (right edge)
# Behavior: interpolated to t_beg + (k+1)*binsize  (right edge)
```

iii. The AI chose consistency with the reference code's behavior interpolation grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `trials.probabilityLeft`, which is constant within a block. A change in `probabilityLeft` (including transitions to/from NaN) marks a new block boundary.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    changed = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        same = (p[1:] == p[:-1]) | (np.isnan(p[1:]) & np.isnan(p[:-1]))
        changed[1:] = ~same
    block_id = np.cumsum(changed) - 1
    first_of_block = np.zeros(block_id[-1] + 1, dtype=np.int64)
    starts = np.nonzero(changed)[0]
    first_of_block[:] = starts
    return np.arange(len(p)) - first_of_block[block_id]
```

iii. Computed on the unfiltered trials table so that a dropped trial still advances the count.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Block boundaries are detected where `probabilityLeft` changes (or transitions to/from NaN). Within each block, the trial's 0-based index is computed as its position minus the position of the first trial of that block. This is computed on the unfiltered trials table, then the values for retained trials are selected.

ii.
```python
tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ... later, after filtering:
tib = tib_all[trial_idx].astype(np.float32)
inputs[:, 1, :] = tib[:, None]
```

iii. Computing on unfiltered trials preserves the animal's real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which is +1 (left), -1 (right), or 0 (no response).

ii.
```python
choice = trials['choice'].to_numpy()[trial_idx]
choice_lbl = ((1 - choice) / 2).astype(np.int64)
```

iii. No-response trials (choice == 0) are already excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The mapping is `(1 - choice) / 2`: +1 (left) maps to 0, -1 (right) maps to 1. This is broadcast across all 100 time bins.

ii.
```python
choice_lbl = ((1 - choice) / 2).astype(np.int64)
outputs[:, 0, :] = choice_lbl[:, None]
```

iii. Matches the instruction: left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pleft = trials['probabilityLeft'].to_numpy()[trial_idx]
prior_lbl = np.full(n_trials, -1, dtype=np.int64)
for val, lbl in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_lbl[np.isclose(pleft, val)] = lbl
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Each of the three probability values is mapped to a class label using `np.isclose` for floating-point safety. If any unexpected value is found, the session is rejected as an error.

ii.
```python
if np.any(prior_lbl < 0):
    bad = np.unique(pleft[prior_lbl < 0])
    return {'eid': eid, 'error': f'unexpected probabilityLeft values {bad}'}
```

iii. Standard categorical encoding as specified.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The velocity is computed internally by `SessionLoader` (interpolation to 1 kHz + Butterworth low-pass filtering), and wheel speed is the absolute value of velocity.

ii.
```python
sess_loader.load_wheel()
beh['wheel-speed'] = {
    'times': sess_loader.wheel['times'].to_numpy(),
    'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
}
```

iii. Same source as the reference code's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel` interpolates wheel position to 1 kHz and computes velocity with a Butterworth low-pass filter. Speed is `abs(velocity)`. The speed trace is then interpolated onto the right edge of each spike bin using `scipy.interp1d(kind='linear', fill_value='extrapolate')`, restricted to samples strictly inside the interval. Finally, it is discretized into 3 classes at per-session tertiles (33.3/66.7 percentiles).

ii.
```python
def interpolate_behavior(times, values, interval_begs, binsize=0.02, n_bins=N_BINS):
    x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
    vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```
```python
ws_lbl, ws_thr = discretize_tertiles(ws)
```

iii. The interpolation grid and skip rules match the reference code's `get_behavior_per_interval`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles: the 33.3 and 66.7 percentile of all wheel speed values pooled across all retained trials of the session define the thresholds. Values are mapped to {0, 1, 2} using `np.digitize`.

ii.
```python
TERTILES = (1.0 / 3.0, 2.0 / 3.0)

def discretize_tertiles(values):
    flat = values.reshape(-1)
    q1, q2 = np.quantile(flat, TERTILES)
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))
```

iii. Per-session tertiles ensure balanced classes and handle the fact that motion energy units are not comparable across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the right edge of each spike bin, using the same grid as `get_behavior_per_interval`: `linspace(t_beg + binsize, t_end, n_bins)`. This grid is offset by half a bin from the bin centers used for neural data (bin centers vs. right edges).

ii.
```python
x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
```

iii. Matches the reference code's behavior interpolation grid exactly.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (preferred) or `rightCamera.ROIMotionEnergy` (fallback), loaded via `SessionLoader.load_motion_energy`. The left camera is tried first.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        me = {'times': df['times'].to_numpy(),
              'values': df['whiskerMotionEnergy'].to_numpy()}
        break
    except Exception:
        continue
```

iii. Follows the reference code's camera preference (left first, right as fallback).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no additional filtering or normalization). It is interpolated onto the right bin edges using the same `interpolate_behavior` function as wheel speed, then discretized into 3 classes at per-session tertiles.

ii.
```python
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                               PARAMS['binsize'], N_BINS)
me_lbl, me_thr = discretize_tertiles(me)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: per-session tertiles (33.3/66.7 percentile), `np.digitize` to map to {0, 1, 2}. A degenerate-distribution fallback exists for heavily tied distributions.

ii.
```python
me_lbl, me_thr = discretize_tertiles(me)
```

iii. Same as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto right bin edges using `interpolate_behavior`.

ii.
```python
v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                               PARAMS['binsize'], N_BINS)
```

iii. Matches the reference code's behavior grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Sessions without whisker motion energy from either camera are skipped. (2) Sessions with fewer than 5 well-isolated grey-matter neurons are skipped. (3) Trials where behavior coverage is incomplete or contains NaN are dropped. (4) Trials where the entire population records zero spikes over the 2 s window are dropped (16 trials). (5) Sessions with fewer than 2 surviving trials are skipped. (6) Probes whose spike sorting was never released produce an error dict.

ii.
```python
if n_neurons < MIN_NEURONS_PER_SESSION:
    return {'eid': eid, 'error': ...}
```
```python
keep_trial &= beh_good[name]  # drops trials with bad behavior
```
```python
has_spikes = binned.any(axis=(1, 2))
if n_zero_spike:
    binned = binned[has_spikes]
```

iii. The AI documented each type of exclusion and the number of affected sessions/trials in CONVERSION_NOTES.md.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk, which involves reading the two spike arrays (times and clusters) that can be hundreds of megabytes per probe. The AI's timing report shows `load_spikes` at 5.8 s/session, dominating the total ~6.7 s/session.

ii.
```python
timings['load_spikes'] = time.time() - t0
# From conversion output: load_spikes ~5.8s/session
```

iii. Disk I/O for large spike arrays is the bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: one in `bin_spikes` that counts spikes per trial, and one in `interpolate_behavior` that interpolates behavior traces per trial. Both iterate over trials sequentially.

ii.
```python
# bin_spikes loop:
for k in range(n_intervals):
    # ... per-trial spike binning
```
```python
# interpolate_behavior loop:
for k in range(n):
    # ... per-trial interpolation
```

iii. The AI noted these loops but determined they are fast enough (~0.01-0.04 s/session) given that disk I/O dominates.

## 10-c. What processing does the code repeat multiple times?

i. The code calls `SpikeSortingLoader.merge_clusters` for each probe and then `merge_probes` to combine them. This is standard and not redundant. No substantial repeated processing was identified.

ii. N/A - no significant repetition.

iii. The code is structured to process each session once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed per-session metadata (`session_info` including cluster counts, good unit counts, timing breakdowns, tertile thresholds, trial indices, etc.) that is not used by the decoder. The `skipped_sessions` list is also stored. However, this is metadata overhead, not computational overhead on the neural/input/output data.

ii.
```python
'info': {
    'n_probes': int(len(probe_rows)),
    'n_clusters_total': int(len(clusters)),
    'n_good_units': n_good,
    # ... extensive metadata
}
```

iii. The metadata is useful for auditing but not for decoding.
