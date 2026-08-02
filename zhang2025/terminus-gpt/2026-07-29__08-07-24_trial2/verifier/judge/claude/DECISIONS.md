# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds sessions by recursively searching `data/one_cache` for `_ibl_trials.table.pqt` files and walking up to the `alf` parent directory. Each session is then processed individually by `process_session()`, which loads trial tables, spike data, wheel data, and motion energy from the session's ALF directory. The data are loaded directly from local `.npy` and `.pqt` files rather than using the IBL ONE API.

ii.
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    sessions = []
    for p in trial_tables:
        sess = p.parent
        while sess.name != 'alf' and sess != sess.parent:
            sess = sess.parent
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
    return out
```

iii. The AI noted in CONVERSION_NOTES.md that "data/ is an IBL ONE cache-style dataset" and chose to discover sessions by finding trial table files. The reference code uses the ONE API (`one.eid2pid`, `SessionLoader`, etc.) and a CSV freeze file (`bwm_release.csv`) to enumerate sessions; the AI bypasses this and directly scans the filesystem.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the session path: `session_path.parts[-3]` (the subject name in the IBL path structure `lab/Subjects/<subject>/<date>/<number>`). A sorted list of unique subjects is built from all processed sessions.

ii.
```python
subject = session_path.parts[-3]
# in build_dataset:
subjects = sorted({p['subject'] for p in processed})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI inferred subject identity from the directory structure, which matches the IBL path convention. This is consistent with how the reference code obtains subject information.

## 1-c. How are the data split into sessions?

i. Each session corresponds to a unique filesystem path (e.g., `lab/Subjects/subject/date/number`). Sessions are processed one at a time by `process_session()` and collected into a list. Each session becomes one entry in the `neural`, `input`, and `output` lists.

ii.
```python
for i, sess in enumerate(sessions, 1):
    st = time.time()
    try:
        p = process_session(sess, show_processing=args.show_processing)
    except Exception as e:
        print(f'[WARN] failed session {sess}: {e}')
        p = None
    if p is not None:
        processed.append(p)
```

iii. This follows the one-session-per-entry structure required by the target format.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file. Each row in the trial table is a trial. After filtering (see 1-e), each remaining row becomes one trial in the session.

ii.
```python
def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)
```

iii. Consistent with the IBL data architecture where trial information is stored in parquet tables.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by requiring: (1) `stimOn_times` is not NaN, (2) `choice` is either -1 or 1 (excludes no-choice trials where choice==0), and (3) `probabilityLeft` is not NaN. Additionally, trials where the binned neural activity is all zeros are removed. However, the AI does **not** apply the reference paper's reaction time filter (0.08s <= RT <= 2.0s based on `firstMovement_times - stimOn_times`), nor does it exclude trials missing `feedback_times`, `feedbackType`, or `firstMovement_times`.

ii.
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
# ...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. The AI's CONVERSION_NOTES.md mentions the paper's filtering rules (reaction time constraints, NaN exclusions) under Step 3, but the actual code only implements a subset. The reference code (`load_trials_and_mask`) applies min_rt=0.08, max_rt=2.0, and NaN exclusion for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files found per probe under the session's `alf/probe*` directory, along with `clusters.metrics.pqt` for quality filtering and `clusters.channels.npy` for brain region mapping.

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. Consistent with the reference code which loads spike times and cluster IDs via `SpikeSortingLoader.load_spike_sorting()`.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping time bins from -0.2s to 1.0s relative to stimulus onset. Multiple probes within a session are merged by remapping cluster IDs. The result is a (n_neurons, n_timepoints) matrix of spike counts per trial.

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    trial_mats = []
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
        # ... bin into (n_neurons, n_bins) matrix
```

iii. The reference code uses `binsize=0.02` (20ms) and `time_window=(-0.5, 1.5)` (2s window), whereas the AI uses a 1.2s window (-0.2 to 1.0). The reference code bins spike counts using `bincount2D` from `iblutil.numerical`. The time window differs from the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `clusters.metrics.pqt`: clusters are kept if `noise_cutoff < 20` and `label >= 1`. The paper specifies three criteria: amplitude > 50 uV, noise cut-off < 20, and refractory period violation. The AI only checks two of these fields.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
```

iii. The reference code uses `qc=1` which filters on `label >= 1`. The `label` column in the IBL metrics typically encodes the result of all three quality criteria combined, so filtering on `label >= 1` should be functionally equivalent to applying all three criteria. The additional `noise_cutoff < 20` filter is redundant but not harmful if `label` already encodes it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are binned relative to that trial's `stimOn_times` value.

ii.
```python
neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
```

iii. The instructions specify alignment to stimulus onset. The reference code also uses `align_time='stimOn_times'` for choice and prior decoding. This is correct.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (`bin_size=0.02`). The trial window spans -0.2s to 1.0s, producing 60 time bins. No temporal rebinning is applied - raw spike times are directly binned.

ii.
```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

iii. The reference code uses `binsize=0.02` (20ms), consistent with the AI's choice. However, the reference time window is `(-0.5, 1.5)` producing 100 time bins (T=100 as stated in methods.txt: "each divided into 20-ms bins, producing T = 100 time steps"), whereas the AI uses `(-0.2, 1.0)` producing only 60 time bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin edges of the neural data time grid, which are defined relative to stimulus onset. No raw data variable is directly used; it's computed from the bin edges.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
# ...
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. Since trials are aligned to stimulus onset, the bin centers represent time since stimulus onset. This is a reasonable approach.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Bin edges are computed using `np.arange(t0, t1 + 1e-9, bin_size)`, and then bin centers are computed as the midpoint of consecutive edges. The values range from -0.19 to 0.99 (centers of 20ms bins from -0.2 to 1.0).

ii.
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. This is a continuous, time-varying input as required by the instructions.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same bin edges are used for both neural data and the time-since-stimulus-onset input, so they are perfectly aligned by construction.

ii. Same bin edge computation used for neural and inputs - see code in 3-a and 3-b above.

iii. Correct alignment by construction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trial table. Block transitions are detected as changes in `probabilityLeft`, and the trial count within each constant-`probabilityLeft` run is computed.

ii.
```python
def compute_trial_number_in_block(prob_left):
    prob_left = np.asarray(prob_left)
    out = np.zeros(len(prob_left), dtype=np.float32)
    c = 0
    prev = None
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
        else:
            c += 1
        out[i] = c
    return out
```

iii. The approach detects block boundaries by looking for changes in `probabilityLeft`. This is a reasonable proxy for trial-in-block counting.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through trials sequentially. When `probabilityLeft` changes from the previous trial, the counter resets to 1; otherwise it increments. The result is a per-trial scalar that is replicated across all time bins.

ii.
```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
# ...
np.full_like(centers, block_trial[i], dtype=np.float32),
```

iii. This is stored as a continuous per-trial value replicated across time bins, matching the "continuous, per-trial" specification in the instructions.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of the trial table.

ii.
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. Consistent with the reference code which uses `trials_df['choice']`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL convention: choice=1 (left) is mapped to 0, choice=-1 (right) is mapped to 1. The result is a binary per-trial value replicated across all time bins.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0    # left -> 0
    out[arr == -1] = 1   # right -> 1
    return out
```

iii. The instructions specify "left = 0, right = 1". IBL convention is choice=1 for left, choice=-1 for right, so the mapping is correct.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of the trial table.

ii.
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. Consistent with the reference code using `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped as a categorical variable: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The result is replicated across all time bins.

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. Matches the instructions: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    if not (posf.exists() and tsf.exists()):
        return None, None
    return np.load(posf), np.load(tsf)
```

iii. The reference code uses `SessionLoader.load_wheel()` which returns wheel position interpolated to a uniform sampling rate, plus velocity and acceleration computed via Gaussian smoothing. The AI loads raw wheel position and timestamps directly. The reference code computes wheel speed as `np.abs(velocity)` from the SessionLoader, whereas the AI manually differentiates position.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI: (1) sorts wheel timestamps, (2) removes duplicate timestamps, (3) computes speed as `abs(diff(position) / diff(timestamps))`, (4) computes midpoint timestamps, (5) interpolates speed to trial-aligned bin centers using `np.interp`.

ii.
```python
order = np.argsort(wheel_ts)
wheel_ts = np.asarray(wheel_ts)[order]
wheel_pos = np.asarray(wheel_pos)[order]
uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
wheel_ts = wheel_ts[uniq_mask]
wheel_pos = wheel_pos[uniq_mask]
if len(wheel_ts) >= 2:
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)
    speed_mid = np.abs(dp / dt).astype(np.float32)
    ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
    wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The reference code uses `SessionLoader.load_wheel()` which applies Gaussian smoothing to compute velocity, then takes absolute value for speed. The AI uses raw finite differences without any smoothing, producing noisier speed estimates. This is a meaningful difference in processing.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes tertile thresholds (33rd and 67th percentiles) across all wheel speed values in the session, then discretizes: values <= q1 -> 0, q1 < values <= q2 -> 1, values > q2 -> 2.

ii.
```python
def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)

def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
```

iii. The instructions say "Wheel speed discretized into 3 bins, time-varying". The AI uses per-session tertile thresholds for discretization. The output distribution shows {low: 0.418, medium: 0.291, high: 0.291}, which is not exactly 1/3 each due to ties at threshold values.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers used for neural data using `interp_to_trial_bins`, which computes bin centers from the neural bin edges and uses `np.interp` for linear interpolation.

ii.
```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
        out.append(y.astype(np.float32))
    return out
```

iii. The wheel speed is aligned to stimulus onset and sampled at the same time points as neural data, ensuring temporal consistency.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy`, along with corresponding camera timestamp files (`_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`).

ii.
```python
def load_motion_energy(session_path: Path):
    alf = session_path / 'alf'
    left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
    right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
    left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
    right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
    streams = []
    if left_me and left_t:
        streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
    if right_me and right_t:
        streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
    return streams
```

iii. The reference code loads whisker motion energy via `load_target_behavior(one, eid, 'left-whisker-motion-energy')` which accesses `whiskerMotionEnergy` from the SessionLoader. The AI loads the raw ROI motion energy arrays directly.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. If both left and right camera motion energy streams are available, they are averaged. The values are then interpolated to the trial-aligned bin centers using `np.interp`.

ii.
```python
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The reference code loads left whisker motion energy first; if unavailable, falls back to right (`if 'skip' in target_dict.keys(): target_dict = load_target_behavior(one, eid, 'right-whisker-motion-energy')`). The AI instead averages both when available, which differs from the reference approach of using a single camera.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: tertile thresholds (33rd/67th percentiles) computed per session, then discretized into 3 bins (0, 1, 2).

ii. Same `tertile_thresholds` and `discretize_with_thresholds` functions as shown in 9-c.

iii. The instructions specify "Whisker motion energy discretized into 3 bins, time-varying". The output distribution shows {low: 0.356, medium: 0.322, high: 0.322}.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel speed: interpolated to the neural data bin centers using `interp_to_trial_bins`.

ii. Same `interp_to_trial_bins` function as shown in 9-d.

iii. Temporally aligned to stimulus onset, consistent with neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions with no trial table, fewer than 2 trials, or missing required columns are skipped entirely. (2) Trials with NaN `stimOn_times`, invalid `choice`, or NaN `probabilityLeft` are excluded. (3) Trials with all-zero neural activity are excluded. (4) If wheel or motion energy data is unavailable, zeros are substituted. (5) Wheel timestamps are sorted and deduplicated to handle non-monotonic data. (6) Session processing failures are caught and warned.

ii.
```python
if trials is None or len(trials) < 2:
    return None
# ...
wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]  # fallback
# ...
try:
    p = process_session(sess, show_processing=args.show_processing)
except Exception as e:
    print(f'[WARN] failed session {sess}: {e}')
    p = None
```

iii. The AI handles missing data by substituting zeros for missing wheel/motion energy data rather than excluding those trials or sessions. The reference code's `get_behavior_per_interval` function has more sophisticated checking (verifying data isn't NaN, starts/ends within bounds) and produces a mask to exclude bad trials.

## 12-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, individual session processing times range from ~2s to ~134s. The most time-consuming step is likely `load_curated_spikes` which loads and filters spike data from multiple probes, and `bin_spikes_for_trials` which bins spikes trial-by-trial. The full conversion of 461 sessions took approximately 7000+ seconds.

ii.
```python
# No explicit timing within process_session; only outer timing:
print(f'processed {i}/{len(sessions)} sessions; kept {len(processed)}; dt={time.time()-st:.2f}s')
```

iii. The conversion output shows highly variable per-session times, with some sessions taking over 2 minutes, likely due to large numbers of spikes or neurons.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The spike cluster remapping loop (`[remap[c] for c in sc]`) could use `np.vectorize` or a lookup array. (2) The `map_prior` function uses a list comprehension that could use `np.searchsorted` or vectorized mapping. (3) The `bin_spikes_for_trials` function loops over trials sequentially rather than using vectorized binning.

ii.
```python
sc = np.array([remap[c] for c in sc], dtype=np.int64)  # could be vectorized
# ...
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)  # could be vectorized
```

iii. The reference code uses multiprocessing for spike binning (`get_spike_data_per_interval`) and behavior interpolation (`get_behavior_per_interval`). The AI's code is single-threaded.

## 12-c. What processing does the code repeat multiple times?

i. (1) The wheel and motion energy interpolation both call `interp_to_trial_bins` separately, which re-computes bin centers each time. (2) Brain region name lookups are done per-neuron in a loop with repeated list searches. (3) The `tertile_thresholds` function is called separately for wheel and whisker data, doing similar concatenation work each time.

ii.
```python
# bin centers recomputed each call:
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
```

iii. These are minor inefficiencies that don't substantially impact correctness.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code builds a global sorted list of brain region names across all sessions in `build_dataset`, remapping indices, even though brain regions are stored as CCF IDs rather than meaningful names (e.g., `ccf_128` instead of `CA1`). (2) The per-trial `choice` and `prior` values are replicated across all 60 time bins as time-varying signals, even though they are per-trial constants. While not discarded, this increases data size unnecessarily compared to storing them as scalars. (3) Processing plots are generated even for sessions beyond the first 2 when `--show-processing` is used (though this is controlled by the flag).

ii.
```python
# CCF IDs used as region names instead of Beryl/acronym mapping:
reg = f'ccf_{int(reg_ids[ch])}' if reg_ids is not None and ch < len(reg_ids) else f'channel_{ch}'
# Per-trial values replicated across time:
np.full_like(centers, choice[i], dtype=np.int64),
```

iii. The reference code uses `BrainRegions().acronym2acronym(...)` with Beryl mapping to get meaningful region names. The AI uses raw CCF IDs, producing 545 distinct "brain region" names that are numeric IDs rather than anatomical labels.
