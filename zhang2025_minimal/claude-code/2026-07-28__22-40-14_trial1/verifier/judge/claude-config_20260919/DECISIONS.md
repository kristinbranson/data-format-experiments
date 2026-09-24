# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API for data access. It tried `SessionLoader`/ONE first, found that ONE's local-mode search did not resolve the revisioned trials table (`alf/#2025-03-03#/_ibl_trials.table.pqt`), and switched to reading the ONE cache's parquet index tables and the raw `.npy`/`.pqt` files directly off disk. The session list is built by intersecting the `eid` column of the Zhang repo's `bwm_release.csv` (699 insertion rows, 459 unique sessions) with the index of `sessions.pqt` from **`one_cache/2022_Q4_IBL_et_al_BWM`**, the oldest of the three release tables present in the cache (2022_Q4: 354 sessions, 2025_Q3: 459, Brainwidemap: 480). Each session's directory is resolved from `datasets.pqt['session_path']`, and individual files are located by `find_file_with_revision`, which checks the plain `alf/` path first and then scans `#...#` revision sub-directories. Everything downstream (trials, spikes, wheel, camera) is `np.load`/`pd.read_parquet` on those paths. The result was 354 candidate sessions, 323 converted, 107 subjects, 142,224 trials.

Verified consequence: the 105 sessions that the 2022_Q4 table omits are fully present on disk (e.g. `hausserlab/Subjects/PL050/2023-06-15/001/` has `alf/#2025-03-03#/_ibl_trials.table.pqt`, `alf/probe00/pykilosort/#2024-05-06#/spikes.times.npy`, wheel and both camera streams). The AI's notes attribute the shortfall to "105 sessions not in local cache", which is not the case — they were excluded purely by the choice of release table. The human reference, searching the `Brainwidemap` table through ONE, converts 441 sessions / 136 subjects / 188,740 trials.

ii.
```python
CACHE_DIR = '/app/data/one_cache'
TABLES_DIR = '/app/data/one_cache/2022_Q4_IBL_et_al_BWM'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
...
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

```python
def find_file_with_revision(base_dir, filename):
    """Find a file that might be in a revision subdirectory like #2025-03-03#/."""
    direct = base_dir / filename
    if direct.exists():
        return direct
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
            candidate = d / filename
            if candidate.exists():
                return candidate
    return None
```

iii. From the trajectory (step 48): *"The trials table has all the data we need. The issue is that ONE's `SessionLoader` isn't finding the versioned table. Let me write a direct data loading approach."* At step 61 it concluded *"Good - 354 sessions in cache out of 459"* and treated that as the complete available dataset; `CONVERSION_NOTES.md` records it as a known limitation: *"354 of 459 sessions available: 105 sessions not in local cache."* No justification is given anywhere for selecting the `2022_Q4_IBL_et_al_BWM` table over the `2025_Q3` or `Brainwidemap` tables sitting beside it in the same cache directory.

## 1-b. How are the data split into subjects?

i. The subject name is read straight off the release index row for each session (`sessions_df.loc[eid, 'subject']`); no path parsing. Subjects are entered into `subjects` in order of first appearance (not sorted) and `subject_idx` records each session's index into that list. Result: 107 subjects across 323 sessions (reference: 136 across 441), the deficit being entirely a consequence of the session set chosen in 1-a — whole animals (e.g. all the `PL0xx` mice of `hausserlab`) are absent.

ii.
```python
row = sessions_df.loc[eid]
subject = row['subject']
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. Not discussed explicitly in the trajectory; the subject field is taken as authoritative from the cache table, the same premise as the reference ("the API already returns a unique subject id, so nothing has to be derived").

## 1-c. How are the data split into sessions?

i. A session is one `eid` = one row of `sessions.pqt` = one recording directory `lab/Subjects/<subject>/<date>/<number>`; the loop body processes exactly one `eid` at a time and appends one entry per session to `neural`/`input`/`output`. No splitting or merging of sessions is performed. Sessions are processed serially in the order the `bwm_release.csv` lists them, so sessions of the same animal end up adjacent.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    print(f"\n--- Session {eid_idx+1}/{len(available_eids)}: {eid} ---")
    session_path = find_session_path(eid, sessions_df, datasets_df)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI treated the `eid` as the natural session unit throughout; no alternative was considered. (The *set* of sessions is the issue documented in 1-a, not the unit.)

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` has one row per trial, and that row structure is the split. Trial windows are then cut from the continuous streams as `stimOn_times + (-0.5, 1.5)` s, so trials are allowed to overlap if the mouse was fast; no interval/ITI logic is used.

ii.
```python
trials = pd.read_parquet(trials_file)
...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
```

iii. Implicit — the parquet trials table is one row per trial, so no decision was needed. Matches the reference's reasoning.

## 1-e. How are trials filtered based on quality controls?

i. Four filters combined into one boolean mask, plus a fifth applied later:
1. No NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`.
2. Reaction time `firstMovement_times - stimOn_times` in [0.08, 2.0] s.
3. `choice != 0` (no-response trials dropped).
4. (later) the trial window must be covered by **both** the wheel and the camera stream: at least one sample present, no NaNs, and the first/last sample within one bin (20 ms) of the window edges — `combined_mask = wheel_mask & me_mask`.
Sessions left with fewer than 2 complete trials are dropped (30 sessions); one more was dropped for having <5 neurons.
Typical retention was ~60–75 % of raw trials (e.g. `408/565`, `245/425`), and the resulting 440 trials/session average is close to the reference's 428.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

```python
# coverage test inside interpolate_behavior_to_bins
if np.abs(t_beg - local_times[0]) > binsize:
    good_mask[trial_idx] = False
...
combined_mask = wheel_mask & me_mask
if n_final < 2:
    print(f"  SKIP: too few trials with complete data")
```

iii. `CONVERSION_NOTES.md`: *"Reaction time: 0.08 - 2.0 seconds … Source: Reference code `ibl_data_utils.py:load_trials_and_mask()` defaults"*, and for the coverage test *"Quality check: Trials excluded if behavior data doesn't cover the full interval"* citing `get_behavior_per_interval()`. So the trial mask is taken verbatim from the reference repository's own default trial mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Per probe: `spikes.times.npy` and `spikes.clusters.npy` build the array itself; `clusters.metrics.pqt` (`label` column) supplies the quality score used to select units; `clusters.channels.npy` together with `channels.brainLocationIds_ccf_2017.npy` supplies each unit's Allen region id, which is converted to a Beryl acronym for `brain_region_idx`. Files are read from `alf/<probe>/pykilosort/` (or its revision sub-directory).

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()

cluster_brain_ids = channels_brain_ids[clusters_channels]
```

iii. Not argued at length; the AI inspected the on-disk `pykilosort` folder contents (steps 39–50) and reproduced by hand what `SpikeSortingLoader.load_spike_sorting` + `merge_clusters` return, since it had abandoned the loader.

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving units are counted into the 100 × 20 ms bins of each trial window using `searchsorted` to slice the trial's spikes and `np.add.at` to accumulate the (cluster, bin) grid. No smoothing, no z-scoring, **no division by the bin width** — the stored values are raw spike counts (verified: max 5, mean 0.11, integer-valued), whereas the reference stores firing rate in Hz. Arrays are stored as **float64** (the reference uses float32), which doubles the file size for data that only ever takes small integer values. The two probes of a session are merged into one population by offsetting the second probe's cluster indices and re-sorting all spikes by time, matching `merge_probes`.

ii.
```python
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
...
bin_indices = np.minimum(((trial_times - t_beg) / binsize).astype(np.int64), n_bins - 1)
valid = trial_clusters < n_clusters_total
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
all_spike_clusters.append(spike_clusters + cluster_offset)
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times, kind='stable')
```

```python
session_neural.append(neural_trial.astype(np.float64))
```

iii. `CONVERSION_NOTES.md` §4: *"Merge spikes from all probes in a session, reindex cluster IDs … Rationale: 'data from the probes recorded in the same session are not statistically independent'"* — taken from `ibl_data_utils.py:merge_probes()`. The counts-not-rates choice is never stated or defended; it simply follows `bin_spiking_data` in the reference repo, which also returns counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `clusters.metrics['label'] >= 1.0` — the IBL pipeline's "well-isolated neuron" criterion (amplitude, noise cut-off, refractory-period violation), the same threshold as the human reference. Spikes of rejected clusters are removed with `ismember` and the survivors are renumbered 0..n-1. Probes whose files are missing, or with zero good clusters, are skipped; sessions with fewer than `MIN_NEURONS = 5` units are dropped (this fired once). Units whose Beryl acronym is **`void`** (outside the brain) are **not** removed — the converted data contains 202 `void` units (and 7,443 `root`, 35 `x`, 6 `y`); the reference drops `void` and keeps `root`. Resulting mean is 178 units/session (reference 164).

Important inconsistency: the module docstring and the `metadata` written into the pickle both state the opposite of what the code does.

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
            ...
            spike_mask, ib = ismember(spike_clusters, good_cluster_ids)
            spike_times = spike_times[spike_mask]
            spike_clusters = ib.astype(np.int32)
            cluster_brain_ids = cluster_brain_ids[good_cluster_ids]
```

but the same file ships:
```python
- All spike-sorted clusters included (no quality filter, matching reference code's qc=None)   # module docstring
...
'cluster_quality_filter': 'none (all clusters, matching reference code)',                     # written into metadata
```

iii. The trajectory shows the AI flip-flopping (steps 63, 98, 100): it first decided to follow Zhang's `qc=None`, then reversed after seeing the file size — *"the file is already 1.1 GB for 3 sessions … 354 sessions * 300 trials * 1300 clusters * 100 bins * 4 bytes ≈ 55 GB. Still too large. Given the practical constraints, I should filter to good quality clusters. This also matches what the BWM paper does for its decoding analyses."* `CONVERSION_NOTES.md` §3 documents the final choice correctly (*"Filter: Well-isolated neurons (cluster quality label >= 1.0) … The Zhang reference code uses qc=None (all clusters), but we apply quality filtering to match the BWM paper's decoding methodology and keep file sizes manageable"*), but the docstring and metadata string were never updated from the earlier decision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times` of each trial. All IBL streams are already on one synchronised session clock, so alignment is just window arithmetic: the trial window is `[stimOn - 0.5, stimOn + 1.5]` s, spikes in that range are selected by `searchsorted`, and the bin index is computed from `t - t_beg`. Trials with NaN alignment time cannot occur (NaN `stimOn_times` is filtered) but are guarded anyway.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
if np.isnan(t_beg) or np.isnan(t_end):
    continue
```

iii. `CONVERSION_NOTES.md` §1: *"Event: Stimulus onset (`stimOn_times`); Window: -0.5 to 1.5 seconds; Source: Reference code `0_data_caching.py` line 53: `'time_window': (-.5, 1.5)`"*, cross-checked against the methods paper (*"align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset"*).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are histogrammed directly into that grid, so there is no resampling or rebinning of the neural data at any point (the behavioural streams are interpolated onto the same grid — see 7/8).

ii.
```python
BINSIZE = 0.02  # 20 ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
...
'time_bin_size': BINSIZE * 1000,  # 20 ms
```

iii. `CONVERSION_NOTES.md` §2: *"Bin size: 20 ms; Number of bins: 100 per trial; Source: Reference code `0_data_caching.py` line 52: `'binsize': 0.02`; Source: Methods paper: 'divided into 20-ms bins, producing T = 100 time steps'."*

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Only from `stimOn_times` and the fixed window/bin constants — the same value vector is reused for every trial of every session, since every trial is cut to the same relative window. No raw time series is read for it.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. Implicit in the alignment decision (§1 of the notes); the AI treats it as a deterministic function of the window definition rather than data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The chosen values are the **right edges** of the 20 ms bins, i.e. −0.48, −0.46, … , 1.50 s (verified in the pickle), rather than the bin centres (−0.49 … 1.49) used by the human reference; this is the grid `get_behavior_per_interval` in the reference repository uses (`np.linspace(beg + binsize, end, n_bins)`). It is stacked as row 0 of the `(2, 100)` per-trial input array and cast to float64.

ii.
```python
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),                                  # (1, T) time-varying
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)            # (1, T) constant
])  # (2, T)
```

iii. `CONVERSION_NOTES.md` §7: *"Bin centers: `linspace(t_beg + binsize, t_end, n_bins)`; Source: Reference code `ibl_data_utils.py:get_behavior_per_interval()`"* and §8: *"time_since_stimulus_onset: Continuous, time-varying. Linearly spaced from -0.48s to 1.5s."* (The notes call these "bin centers", but they are the right edges.)

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Bin-for-bin: element *k* of the time input corresponds to neural column *k*, which counts spikes in `[stimOn − 0.5 + 0.02k, stimOn − 0.5 + 0.02(k+1))`. The time input labels that bin by its closing edge, so its value sits 10 ms later than the centre of the neural bin it labels — a constant half-bin offset applied uniformly to all trials, sessions, and to both behavioural outputs, which use the identical grid. Column indices are therefore perfectly consistent across `neural`, `input` and `output`.

ii.
```python
# neural bin k spans [t_beg + k*binsize, t_beg + (k+1)*binsize)
bin_indices = np.minimum(((trial_times - t_beg) / binsize).astype(np.int64), n_bins - 1)
```
```python
# same k labelled by the closing edge
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS)
# behaviour sampled on the same grid
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. Consistency with the reference repository's interval functions is the stated reason (`CONVERSION_NOTES.md` §7); the half-bin offset itself is not discussed.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` of the trials table only — the table has no block id, so block boundaries are recovered as the points where `probabilityLeft` changes value (a NaN on either side also starts a new block).

ii.
```python
prob_left = trials_df['probabilityLeft'].values
...
if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
    block_start = i
```

iii. `CONVERSION_NOTES.md` §8: *"trial_number_in_block: Continuous, constant within trial. 0-indexed count from block start."* The block-from-prior inference is implicit and identical to the reference's reasoning.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop over **all** trials of the session (before the quality mask) assigns each trial its 0-indexed position since the last change of `probabilityLeft`; the array is then subset with the trial mask, so a trial that was later discarded still advances the counter and the number reflects the animal's true position in the block. The scalar is tiled across all 100 bins as row 1 of the input array, as a float. Observed range 0–98, identical to the reference.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    prob_left = trials_df['probabilityLeft'].values
    trial_nums = np.zeros(len(trials_df), dtype=np.float64)
    block_start = 0
    for i in range(1, len(prob_left)):
        if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
            block_start = i
        trial_nums[i] = i - block_start
    return trial_nums[mask]
```
```python
trial_nums_in_block = compute_trial_number_in_block(trials, mask)
...
trial_num = trial_nums_in_block[trial_global_idx]
```

iii. Not separately justified in the trajectory; counting on the unfiltered table is the natural reading of "trial number in block" and matches what the reference does with `groupby(block).cumcount()`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 / −1 / 0. Zero (no response) trials are already removed by the trial mask. The AI maps **−1 → 0 (labelled "left")** and **+1 → 1 (labelled "right")**, stating in both the code comment and the notes that "−1 = left, 1 = right" in the raw data.

This mapping is inverted with respect to the IBL convention and to the human reference (`CHOICE = {1.0: 0, -1.0: 1}`). Verified directly from the raw trials tables: on correct (`feedbackType == 1`) high-contrast trials with the stimulus on the **left** (`contrastLeft > 0.2`) every trial has `choice == +1`, and with the stimulus on the right every trial has `choice == −1`. Verified again in the AI's own output: among trials the AI labels prior class 2 (p(left) = 0.8, i.e. left-biased blocks) only 23.4 % carry the label "left", while in prior class 0 (p(left) = 0.2) 74.8 % carry "left" — the labels behave exactly as if they were swapped. The class fractions are the mirror image of the reference's (AI 0.493/0.507 vs reference 0.5075/0.4925).

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
...
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```
```python
'output_values': [
    ['left', 'right'],  # choice: 0=left, 1=right
    ...
```

iii. `CONVERSION_NOTES.md` §9: *"choice: Binary (0=left, 1=right). Original data: -1=left, 1=right."* No verification of the sign convention against the data (e.g. against `contrastLeft`/`feedbackType` or against block bias) appears anywhere in the trajectory; the convention was asserted, not checked.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the recoding described in 5-a: a per-trial scalar, cast to int64 and tiled across all 100 bins so that the output array is uniformly `(4, 100)` and time-varying-compatible.

ii.
```python
output_trial = np.vstack([
    np.full((1, N_TIMEBINS), choice, dtype=np.int64),
    np.full((1, N_TIMEBINS), prior, dtype=np.int64),
    wheel_disc.reshape(1, -1),
    me_disc.reshape(1, -1)
])  # (4, T)
```

iii. The instructions ask for per-trial outputs to be time-varying "if at all possible"; tiling the constant across bins satisfies the shape requirement. Same as the reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 and 0.8 (the block prior the task holds constant within a block). NaN values are already excluded by the trial mask.

ii.
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. `CONVERSION_NOTES.md` §9: *"prior: Categorical 3-class. p(left)=0.2 -> 0, p(left)=0.5 -> 1, p(left)=0.8 -> 2"*, the mapping given in the task instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way recode 0.2→0, 0.5→1, 0.8→2 by explicit comparison, then tiled across the 100 bins as row 1 of the output array. The branch is written as an `else` rather than a third equality test, so any unexpected value would silently become class 2; in practice `probabilityLeft` only ever takes the three canonical values, and the resulting class fractions (0.418 / 0.139 / 0.443) match the reference's (0.4180 / 0.1405 / 0.4415) to within 0.002.

ii.
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
```

iii. Directly from the task instructions; the AI also sanity-checked the resulting distribution against the protocol (`CONVERSION_NOTES.md` Check 3: *"Data paper: '90 unbiased trials' then biased blocks … Our prior distribution: ~15% at 0.5 … 90/~600 total trials ≈ 15% unbiased -> matches"*).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `alf/_ibl_wheel.position.npy` and `alf/_ibl_wheel.timestamps.npy` — the raw, movement-triggered wheel encoder stream. Speed is the absolute value of the velocity derived from them. If either file is absent (17 sessions) the wheel mask is set all-false and the session is dropped.

ii.
```python
pos_file = alf_dir / '_ibl_wheel.position.npy'
ts_file = alf_dir / '_ibl_wheel.timestamps.npy'
if not pos_file.exists() or not ts_file.exists():
    return None, None
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. `CONVERSION_NOTES.md` §7: *"Source: Raw wheel position and timestamps … Matching: ibllib's `interpolate_position` and `velocity_filtered` functions"*, i.e. reproducing what `SessionLoader.load_wheel` would have returned, which is what the reference code's `'wheel-speed'` target uses.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps, the first two reproducing `SessionLoader.load_wheel` exactly by calling the same ibllib primitives: (1) `interpolate_position(timestamps, position, freq=1000)` puts the irregular encoder samples on a uniform 1 kHz grid; (2) `velocity_filtered(pos_interp, 1000)` differentiates with a 20 Hz Butterworth low-pass (ibllib defaults `corner_frequency=20, order=8`) and returns (velocity, acceleration) — the acceleration is discarded; (3) `speed = np.abs(velocity)` in rad/s. The trace is then linearly interpolated (`scipy.interp1d`, `fill_value='extrapolate'`) onto the 100-point per-trial grid, using only samples inside the window ± one bin. This is the same pipeline as the reference, whose per-trial resampling uses `np.interp` (clamping instead of extrapolating at the edges — immaterial, because trials whose samples do not reach the window edges are dropped anyway).

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```
```python
beh_mask = (beh_times >= t_beg - binsize) & (beh_times <= t_end + binsize)
local_times = beh_times[beh_mask]
local_vals = beh_values[beh_mask]
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. Trajectory steps 75–86 show the AI discovering that the raw wheel timestamps are non-uniform (*"The wheel data has non-uniform timestamps. Let me fix the wheel speed computation - need to interpolate to uniform timestamps first like the SessionLoader does"*) and then fixing the `velocity_filtered` call signature and tuple return. `CONVERSION_NOTES.md` §7 records the final pipeline.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three equal-frequency classes with thresholds computed **per session**: the trial-by-bin speed values of all kept trials of that session are concatenated, the 33.33rd and 66.67th percentiles are taken, and `np.digitize` assigns 0/1/2 ("low"/"medium"/"high"). Thresholds therefore differ between sessions. Result: 0.333 / 0.333 / 0.333 of all bins, matching the reference's balanced split exactly.

ii.
```python
for idx in combined_indices:
    if wheel_binned_list[idx] is not None:
        all_wheel_vals.append(wheel_binned_list[idx])
all_wheel_concat = np.concatenate(all_wheel_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. `CONVERSION_NOTES.md` §7: *"Discretization: 3 equal-frequency bins (33rd and 67th percentiles per session)"*, and listed under Known Limitations: *"Bin edges vary between sessions."* Equal-frequency binning is the natural reading of the instruction to "discretize into 3 bins" and keeps the decoder's chance level at 1/3 — the same rationale and the same rule as the reference.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Via the same per-trial grid as everything else: the trace is evaluated at `stimOn + linspace(-0.48, 1.5, 100)`, so element *k* pairs with neural column *k* (with the constant +10 ms half-bin offset described in 3-c). No clock correction is needed because the wheel timestamps are already on the session clock. A trial is kept only if wheel samples actually span the window to within one bin at both ends, so no trial is aligned by extrapolation over a gap.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)   # t_beg = stimOn - 0.5
...
if np.abs(t_beg - local_times[0]) > binsize:   # coverage at window start
    good_mask[trial_idx] = False
if np.abs(t_end - local_times[-1]) > binsize:  # coverage at window end
    good_mask[trial_idx] = False
```

iii. `CONVERSION_NOTES.md` §7 "Behavior Interpolation": *"Method: Linear interpolation to match neural bin times … Quality check: Trials excluded if behavior data doesn't cover the full interval"*, citing `get_behavior_per_interval()`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, falling back to the right camera when the left pair is absent. This is IBL's released per-frame motion energy over the whisker-pad ROI; no video is re-processed. If neither camera is available (17 sessions) the session is dropped.

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or ts_file is None:
    # Fallback to right camera
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. `CONVERSION_NOTES.md` §7: *"Source: `leftCamera.ROIMotionEnergy.npy` (fallback to right camera); Camera: Left camera at 60 Hz, matching 'whisker pad area' definition from data paper"*, noted as *"Matching reference code: `bin_behaviors` for 'whisker-motion-energy'"*, with the resolution/frame-rate difference between cameras flagged under Known Limitations. Same preference order as the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, normalisation or unit conversion. Two small robustness steps are added relative to the reference: the value and timestamp arrays are truncated to their common length (they can differ by a frame), and NaN samples are dropped before interpolation; a session with fewer than 10 valid samples is treated as having no camera. The cleaned trace is then linearly interpolated onto the same 100-point per-trial grid as the wheel, with the same coverage test, and any trial containing a NaN inside its window is dropped.

ii.
```python
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
valid = ~np.isnan(me) & ~np.isnan(times)
if np.sum(valid) < 10:
    return None, None
return times[valid], me[valid]
```
```python
if np.any(np.isnan(local_vals)):
    good_mask[trial_idx] = False
    result.append(None)
    continue
```

iii. Not separately argued; `CONVERSION_NOTES.md` treats the motion-energy trace as a released product to be resampled, exactly as the reference does.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 33.33rd/66.67th percentiles of the concatenated per-session trace, `np.digitize` into 0/1/2 = "low"/"medium"/"high", thresholds recomputed per session. Observed fractions 0.332 / 0.332 / 0.335, essentially identical to the reference's.

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
...
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)  # 0, 1, 2
```

iii. Same justification as 7-c (`CONVERSION_NOTES.md` §7): equal-frequency 3-class split per session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: camera frame times are already on the session clock, so the trace is simply evaluated at `stimOn + linspace(-0.48, 1.5, 100)`, giving bin-for-bin correspondence with the neural columns (with the shared +10 ms half-bin offset). The same coverage test guarantees real frames span the window — important here because the camera runs at 60 Hz (left) or 150 Hz (right), i.e. 1–3 frames per 20 ms bin, and dropped frames would otherwise be silently interpolated across.

ii.
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
...
combined_mask = wheel_mask & me_mask
```

iii. As in 7-d — the same `get_behavior_per_interval`-derived routine is reused for both behavioural streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything problematic is dropped rather than imputed, at the finest level at which it can be dropped:
- Trial level: NaN in any of the six required trials columns; NaN whisker samples inside the window; wheel/camera coverage gaps at the window edges; NaN alignment time.
- Probe level: missing `spikes.times`/`spikes.clusters`/`clusters.channels`/`channels.brainLocationIds` files, or a probe with zero label≥1 clusters, is skipped and the remaining probes still used.
- Session level: trials table unreadable; no `probe*` directory; no usable probe; fewer than 5 units; fewer than 2 valid trials before or after the behavioural mask. 31 of 354 sessions were skipped (30 for too few complete trials, 1 for <5 clusters).
- Array-level repairs: camera value/time arrays truncated to common length; `interp1d` wrapped in `try/except`; `valid = trial_clusters < n_clusters_total` guards against out-of-range cluster ids.
Only 3 trials in the whole 323-session output tripped the verifier's "all neural data is zero" warning (the reference has 17 such trials).

One latent issue: files are resolved by scanning `alf/` for `#...#` directories in ascending sort order and returning the **first** match, i.e. the *oldest* revision, whereas ONE resolves to the *newest*. Only 5 sessions on disk ship two revisions of `_ibl_trials.table.pqt` (`#2024-07-15#` and `#2025-03-03#`) and every `spikes.times.npy` has a single revision, so the practical impact here is negligible, but the rule is the opposite of the reference's.

ii.
```python
if any(f is None for f in [spike_times_file, spike_clusters_file,
                            clusters_channels_file, channels_brain_ids_file]):
    return None, None, None, None
```
```python
if len(probes_data) == 0:
    print(f"  SKIP: no valid probe data"); n_skipped += 1; continue
if n_clusters < MIN_NEURONS:
    print(f"  SKIP: too few clusters ({n_clusters} < {MIN_NEURONS})"); n_skipped += 1; continue
if n_final < 2:
    print(f"  SKIP: too few trials with complete data"); n_skipped += 1; continue
```
```python
valid = ~np.isnan(me) & ~np.isnan(times)
```

iii. Largely undiscussed in the trajectory — the guards were added reactively as sessions failed during the sample runs. `CONVERSION_NOTES.md` reports the outcome (*"323 (31 skipped due to missing data)"*) and the AI checked the survivors against the data paper's expectations (trial counts, ~108 well-isolated neurons per probe, 90 unbiased trials, ~50/50 choice split) as sanity checks.

## 10-a. What are the most time-consuming steps of the code?

i. Measured from the conversion log, the full run took roughly 35 minutes of wall clock for 323 sessions, single-threaded. The dominant costs, in order:
1. **Reading the spike sorting** — `np.load` of `spikes.times.npy` + `spikes.clusters.npy` per probe (hundreds of MB each), plus `ismember` over every spike to apply the QC mask. Same dominant cost the reference identifies.
2. **`bin_spikes_per_trial`** — a Python loop over trials whose accumulation uses `np.add.at`, which is an unbuffered scatter-add and is roughly an order of magnitude slower than `np.bincount` on the same data (the reference uses `bincount`). The AI itself noticed this step was slow (step 92) and rewrote the inner loop once.
3. **`interpolate_behavior_to_bins`** — called twice per session, and for each trial it builds a boolean mask over the **entire** session-length stream. The interpolated wheel trace is 1 kHz, so a 90-minute session is ~5.4 M samples, scanned once per trial: ~2.7 × 10⁹ element comparisons for a 500-trial session, for 100 output points.
4. **Serial execution**: sessions are processed one at a time in a single process (`--n-workers` is parsed but never used). The reference runs 10 sessions in parallel via `ProcessPoolExecutor`.
5. **Pickling 19.8 GB** of float64 neural data in one `pickle.dump`.

ii.
```python
ap.add_argument('--n-workers', type=int, default=1)   # parsed, never used
...
for eid_idx, eid in enumerate(available_eids):        # strictly serial
```
```python
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```
```python
for trial_idx in range(n_trials):
    ...
    beh_mask = (beh_times >= t_beg - binsize) & (beh_times <= t_end + binsize)   # O(len(beh_times)) per trial
```

iii. The AI only addressed this reactively: step 92 — *"the spike binning is very slow for large sessions. Let me optimize it"*, then step 94 — *"the inner loop is slow in pure Python. Let me use numpy more efficiently"*. There is no profiling and no discussion of parallelising sessions; it simply launched the full conversion in the background and polled it for ~35 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
1. `interpolate_behavior_to_bins`'s per-trial loop — the full-array boolean mask should be two `np.searchsorted` calls (what the reference does with `window_slice`), turning an O(n_trials × n_samples) scan into O(n_trials log n_samples). This is the single biggest avoidable cost in the script.
2. `bin_spikes_per_trial`'s per-trial loop — `np.add.at` on a 2-D index pair can be replaced by a flat index plus one `np.bincount` per trial (the reference's formulation), or the trial loop can be collapsed entirely by offsetting each spike's bin index by its trial.
3. `compute_trial_number_in_block`'s Python loop over every trial — one line of pandas (`trials.groupby((p != p.shift()).cumsum()).cumcount()`), as the reference writes it.
4. The final per-trial assembly loop, which re-creates the constant `np.full((1, 100), …)` rows and re-stacks arrays trial by trial; the choice/prior rows could be broadcast once per session.
None of these change the results, and (1) and (3) are the only ones with material cost.

ii.
```python
beh_mask = (beh_times >= t_beg - binsize) & (beh_times <= t_end + binsize)
```
```python
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start
```
```python
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
    input_trial_full = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The AI vectorised only the spike-binning inner loop and only because it was visibly slow (*"Actually, the inner loop is slow in pure Python. Let me use numpy more efficiently"*, step 94). The behavioural loop and the block-counting loop are never revisited.

## 10-c. What processing does the code repeat multiple times?

i. Several small repetitions, none affecting correctness:
- `find_file_with_revision` re-lists the `alf/` (or `pykilosort/`) directory for **every** file it looks up — 5 lookups per probe plus 4 per session, each an `iterdir()` + sort.
- Brain-region resolution is done in two passes (`br.id2acronym(ids)` then `br.acronym2acronym(acronyms, mapping='Beryl')`) where a single id→Beryl lookup would do.
- The behavioural coverage test and the window slicing are re-derived inside `interpolate_behavior_to_bins` for each trial and again for each of the two streams, instead of being computed once per session.
- The percentile thresholds are computed from a fresh concatenation of per-trial lists that were themselves just built trial by trial.
- The sanity-check block at the end re-walks every trial of every session to recompute choice/prior fractions that were already known per session.
- `n_valid_trials`/masks are recomputed as `mask.sum()` and `np.where(combined_mask)[0]` in several places.

ii.
```python
for d in sorted(base_dir.iterdir()):        # re-scanned on every call
```
```python
cluster_acronyms = br.id2acronym(cluster_brain_ids)
beryl_acronyms = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
```
```python
for session in all_output:
    for trial in session:
        all_choices.append(trial[0, 0])
        all_priors.append(trial[1, 0])
```

iii. Not discussed. These are incidental, and the human reference reports "N/A" for this question — the AI's repetitions are of the same negligible magnitude, except for the per-trial stream rescans already counted under 10-a/10-b.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **float64 neural arrays.** The values are small integer spike counts (max 5 in the sample), yet they are stored as float64, doubling the pickle to 19.8 GB for 323 sessions where float32 (the reference's choice) would halve it and int16 would cut it by 4×. Likewise the outputs are stored as int64 where int8 suffices (the reference uses int8).
- **`discretize_to_bins()` is dead code** — a fully written equal-frequency discretisation helper that is never called; the script uses an inline `np.percentile`/`np.digitize` instead.
- **Acceleration** is computed by `velocity_filtered` and immediately thrown away (unavoidable given the ibllib API, but it is ~half that call's work).
- **Spikes are binned for trials that are later dropped.** `bin_spikes_per_trial` runs over all mask-valid trials, and only afterwards is `combined_mask` (wheel + camera coverage) applied; in the 30 skipped sessions the entire binning pass was wasted, and elsewhere a few percent of trials are binned and discarded.
- Minor: `wheel_binned = None` / `me_binned = None` are assigned and never used; the whole `n_skipped`/sanity-check reporting pass; `--n-workers` is parsed and ignored.

ii.
```python
def discretize_to_bins(values, n_bins=3):     # never called anywhere
```
```python
session_neural.append(neural_trial.astype(np.float64))
...
output_trial = np.vstack([... dtype=np.int64 ...])
```
```python
binned_spikes = bin_spikes_per_trial(...)      # all valid trials
...
combined_mask = wheel_mask & me_mask           # trials dropped only afterwards
neural_trial = binned_spikes[trial_global_idx]
```

iii. Not discussed. The AI was acutely aware of file size when choosing the QC filter (step 100: *"354 sessions * 300 trials * 1300 clusters * 100 bins * 4 bytes ≈ 55 GB. Still too large"* and *"Let me use a more compact dtype (float32 instead of float64) for neural data, since these are spike counts (integers)"*) but never actually applied the dtype change it proposed — it solved the size problem with the QC filter alone and left the arrays at float64.
