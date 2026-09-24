# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates the dataset from the release freeze table shipped with the methods-paper repository, `/app/code/code_zhang2025/data/bwm_release.csv` (699 probe insertions, 459 sessions, 139 subjects, with columns `pid`, `eid`, `probe_name`, `subject`, `lab`). This is exactly the file that the reference script `src/0_data_caching.py` reads. Everything else is loaded through the ONE API pointed at the local cache `/app/data/one_cache`: one `SpikeSortingLoader` per `pid` for spikes/clusters/channels, and one `SessionLoader` per `eid` for the trials table, the wheel, and the camera motion energy. There is no pre-flight check that a session actually has all required datasets on disk; instead every session is wrapped in `try/except` and sessions that fail (or that have no whisker motion energy) are skipped. 444 of 459 sessions were converted, 15 skipped.

ii.
```python
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international', silent=True,
    cache_dir=args.cache_dir            # /app/data/one_cache
)

bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
print(f'BWM release: {len(bwm_df)} probes, {bwm_df.eid.nunique()} sessions')
```

```python
eids = bwm_df['eid'].unique()
...
for eid_idx, eid in enumerate(eids):
    try:
        rows = bwm_df[bwm_df['eid'] == eid]
        lab = rows.iloc[0]['lab']
        subject = rows.iloc[0]['subject']
        ...
        pids = rows['pid'].values
        probe_names = rows['probe_name'].values
        for pid, pname in zip(pids, probe_names):
            spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
```

```python
def load_spiking_data(one, pid, eid='', pname=''):
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

```python
def load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2.0):
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
```

iii. From the agent's notes and trajectory: the conversion is meant to reproduce the methods paper's pipeline, and the reference script `0_data_caching.py` starts from `freeze_file = 'data/bwm_release.csv'` and then calls `prepare_data(one, eid, bwm_df, params)`, which loops over `one.eid2pid(eid)` and merges probes. The agent mirrored that structure (`load_spiking_data`, `merge_probes`, `load_trials_and_mask` are near-verbatim copies of the reference functions), but takes `pid`/`probe_name` straight from the freeze table instead of calling `eid2pid`, which avoids a database round-trip. A sub-agent exploration (step 25) confirmed the CSV lists 699 probes / 459 sessions / 139 subjects and that the ONE cache holds the matching ALF files.

## 1-b. How are the data split into subjects (mice)?

i. No parsing is needed: the `subject` column of `bwm_release.csv` gives the mouse for every `eid`. Subjects are accumulated in first-appearance order into `subjects_list`, and each session stores the index of its subject. This yields 139 subjects, mean 3.3 sessions per subject (min 1, max 13).

ii.
```python
rows = bwm_df[bwm_df['eid'] == eid]
lab = rows.iloc[0]['lab']
subject = rows.iloc[0]['subject']
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

```python
'subjects': subjects_list,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The freeze table already carries a unique subject id per session, so nothing has to be derived. The agent used the count of unique subjects (139) as a sanity check against the data paper's "We trained 139 mice" (step 158, CONVERSION_NOTES sanity check 2).

## 1-c. How are the data split into sessions?

i. A session is the unit the release table is organised by. The AI iterates over `bwm_df['eid'].unique()` — one iteration per session — and all probes of a session (the rows sharing that `eid`) are merged into a single population, so a session is never split by probe.

ii.
```python
eids = bwm_df['eid'].unique()
if sample_mode:
    eids = eids[:5]
elif max_sessions is not None:
    eids = eids[:max_sessions]
...
for eid_idx, eid in enumerate(eids):
```

```python
if len(spikes_list) > 1:
    spikes, clusters = merge_probes(spikes_list, clusters_list)
else:
    spikes = spikes_list[0]
    clusters = clusters_list[0]
```

iii. No decision to make for the split itself; the merge of probes within a session is justified in CONVERSION_NOTES §5 by the reference code (`prepare_data` calls `merge_probes`) and the data paper ("Although a session may have included multiple probe insertions, we did not perform decoding on these probes separately because they are not independent").

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by the data. A trial is materialised as the 2 s window `stimOn_times + (-0.5, 1.5)`.

ii.
```python
masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values
```

```python
for trial_idx in range(n_trials):
    t_start = align_times[trial_idx] + window[0]
    t_end = align_times[trial_idx] + window[1]
```

iii. Nothing to decide — the trials table is already one row per trial; the window and alignment event follow the reference config `params = {'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` (CONVERSION_NOTES §2).

## 1-e. How are trials filtered based on quality controls?

i. Two filters combined.
(1) A trial mask that is a re-implementation of the reference `load_trials_and_mask` defaults: reaction time (`firstMovement_times - stimOn_times`) must be in [0.08 s, 2.0 s]; no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; and no-response trials (`choice == 0`) are dropped.
(2) A behaviour-coverage mask produced while interpolating the wheel and the whisker trace: a trial is dropped if fewer than two samples fall in its window, if any sample is NaN, or if the stream starts more than one bin after the window start / ends more than one bin before the window end. Wheel and whisker masks are ANDed.
Session-level QC: a session is skipped if fewer than 10 trials survive either mask, if no probe loaded, or if no whisker motion energy exists.
The reference code's `max_trial_len=10.0` (passed in `prepare_data`) was *not* reproduced.

ii.
```python
    # RT filter
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    if min_rt is not None:
        mask &= rt >= min_rt
    if max_rt is not None:
        mask &= rt <= max_rt

    # NaN exclusion
    for event in nan_exclude:
        mask &= ~trials[event].isnull()

    # Exclude no-choice trials (choice == 0)
    mask &= trials['choice'] != 0
```

```python
        if len(bt) < 2:
            good_mask[trial_idx] = False
            continue
        if np.any(np.isnan(bv)):
            good_mask[trial_idx] = False
            continue
        # Check coverage
        if np.abs(t_start - bt[0]) > binsize:
            good_mask[trial_idx] = False
            continue
        if np.abs(t_end - bt[-1]) > binsize:
            good_mask[trial_idx] = False
            continue
```

```python
combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    print(f'  Skipping: only {combined_mask.sum()} trials with valid behavior')
    skipped_sessions += 1
    continue
```

iii. CONVERSION_NOTES §4: "Matches `load_trials_and_mask` defaults in reference code. BWM paper states: 'trials were excluded if…the first wheel-movement time were outside the range of 0.08–2.00 s'". The coverage checks are copied from the reference `get_behavior_per_interval`, whose skip reasons are 'target data starts too late' / 'target data ends too early' / 'nans in target data'.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()` per probe. The merged cluster table (`merge_clusters(...).to_df()`) is used only for its row count (how many units the population has) and for `acronym` (brain region); its `label` QC column is deliberately not used.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

```python
cluster_ids = np.arange(len(clusters))
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. These are the two arrays the reference `bin_spiking_data` consumes (`neural_dict['spike_times']`, `neural_dict['spike_clusters']`). The agent kept the reference's structure but dropped the `raw_electrophysiology(...).fs` call of `load_spiking_data`, which is only needed for the sampling frequency and would stream raw data.

## 2-b. How is the `neural` data processed?

i. Probes of a session are merged into one population (cluster ids of the second probe offset by `clusters.index.max() + 1` of the first, spikes concatenated and re-sorted by time — a copy of the reference `merge_probes`, made non-mutating). Spikes are then counted into 20 ms bins over the 2 s window of each trial, giving `(n_trials, n_clusters, 100)`. Values are raw **spike counts** (not divided by the bin width, i.e. not converted to Hz), stored as `float32`. No smoothing, no normalisation, no z-scoring. Every row of the cluster table becomes a "neuron", including clusters that emitted no spike at all in the session; total 599,865 units, mean 1,351 per session.

ii.
```python
def merge_probes(spikes_list, clusters_list):
    ...
    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes_copy = {k: v.copy() for k, v in spikes.items()}
        spikes_copy['clusters'] = spikes_copy['clusters'] + cluster_max
        cluster_max = clusters.index.max() + 1
        ...
    sort_idx = np.argsort(merged_spikes_dict['times'], kind='stable')
    merged_spikes_dict = {k: v[sort_idx] for k, v in merged_spikes_dict.items()}
```

```python
        times_in_window = sorted_times[i_start:i_end]
        clusters_in_window = sorted_clusters[i_start:i_end]

        # Vectorized binning
        bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
        np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
        ...
        np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
```

iii. CONVERSION_NOTES §3 and §5: the reference uses `binsize=0.02` and the methods paper states "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps… we bin spike counts using all neurons". Probe merging follows `merge_probes` and the data paper's statement that probes in one session are not independent. The agent rewrote the binning loop for speed mid-run (step 110) after measuring ~40 s/session, and verified the optimised version reproduced the original output exactly (steps 112–114).

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every spike-sorted cluster is kept: no `label >= 1` ("good unit") cut, no firing-rate or presence-ratio cut, no removal of units whose Beryl acronym is `void` (histologically outside the brain) or `root`, and no removal of clusters with zero spikes. 599,865 units over 444 sessions are retained (the released set is 621,733), 280 Beryl regions appear in `brain_regions` (including `void`/`root`), and the resulting pickle is ~100 GB.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Load spike sorting data for a probe insertion. Uses all clusters (no QC filter)
    matching the reference code's default behavior in prepare_data."""
```

```python
# Get brain regions (Beryl mapping)
beryl_reg = brainreg.acronym2acronym(clusters['acronym'].values, mapping='Beryl')
...
cluster_ids = np.arange(len(clusters))     # every row of the cluster table is a "neuron"
```

```python
'neuron_selection': 'All spike-sorted clusters (matching reference code default)',
```

iii. CONVERSION_NOTES §1: "The reference code's `prepare_data` function calls `load_spiking_data` without the `qc` parameter, defaulting to `qc=None` which returns all clusters… Using all clusters means we include multi-unit activity, which is appropriate for population decoding." This is factually true of the reference code (`load_spiking_data(one, pid, compute_metrics=False, qc=None, ...)`, and `prepare_data` never passes `qc`), and it is also what the methods paper text says: "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session". The agent explicitly noted the counter-evidence — "The BWM paper describes 75,708 well-isolated neurons out of 621,733 units" — and chose the reference code's behaviour anyway.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. All IBL streams share one session clock, so alignment is a subtraction: the window for trial *k* is `[stimOn_k - 0.5, stimOn_k + 1.5]`, spikes inside it are located by `searchsorted` on the time-sorted spike array, and the bin index is `floor((t - t_start) / 0.02)` clipped to [0, 99]. Trials with NaN alignment times are skipped (they are already removed by the trial mask).

ii.
```python
    for trial_idx in range(n_trials):
        t_start = align_times[trial_idx] + window[0]
        t_end = align_times[trial_idx] + window[1]

        if np.isnan(t_start) or np.isnan(t_end):
            continue

        # Use searchsorted for fast spike selection
        i_start = np.searchsorted(sorted_times, t_start, side='left')
        i_end = np.searchsorted(sorted_times, t_end, side='left')
        ...
        bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```

```python
'temporal_alignment_event': 'stimulus onset (stimOn_times)',
'off_start': WINDOW[0],  # -0.5s before stimulus onset
'off_end': WINDOW[1],  # 1.5s after stimulus onset
```

iii. CONVERSION_NOTES §2: the task instructions say "Temporally align based on stimulus onset"; the methods paper says "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset"; and the reference config is `{'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, non-overlapping; `time_bin_size` is recorded as 20.0 ms in the metadata. Spikes are binned once directly at 20 ms — there is no finer binning followed by rebinning, and no resampling/smoothing of the neural data.

ii.
```python
WINDOW = (-0.5, 1.5)  # seconds relative to stimOn_times
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
```

```python
'time_bin_size': BINSIZE * 1000,  # 20ms
```

iii. CONVERSION_NOTES §3: "Reference code uses `binsize=0.02`. Methods paper states: 'Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps.'" The agent also noted that the methods paper mentions 50 ms bins for choice/prior specifically, and decided to use 20 ms universally "since we decode all variables simultaneously", which is also what the reference config does.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not from a raw variable as such: it is defined by the analysis window itself, i.e. by `stimOn_times` (the alignment event, which becomes t = 0) together with the constants `WINDOW = (-0.5, 1.5)` and `BINSIZE = 0.02`. The same 100-value vector is used for every trial of every session.

ii.
```python
# Time since stimulus onset (same for all trials: relative time in window)
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

```python
'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
```

iii. The decoder task lists "Time since stimulus onset, continuous, time-varying"; since trials are aligned to `stimOn_times`, the time axis relative to the onset is a deterministic function of the window and the bin size (CONVERSION_NOTES §8).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A single `np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS)`, i.e. the 100 **right edges** of the bins, from −0.48 s to +1.50 s (the verification log confirms the range [−0.480, 1.500]). It is stored as a continuous ramp (not a binary onset indicator), tiled unchanged into every trial's `(2, 100)` input array as row 0.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

```python
    # Input: (2, T) - time since stimulus onset + trial number in block
    time_input = sm['time_since_stim'].copy()  # (T,)
    trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
    inp = np.stack([time_input, trial_num], axis=0)  # (2, T)
    session_input.append(inp.astype(np.float32))
```

iii. CONVERSION_NOTES §8/§9: the grid is the one the reference code uses for its behavioural interpolation, `x_interp = np.linspace(interval_begs + binsize, interval_ends, n_bins)`, so the time input and the behavioural outputs sit on exactly the same sample points. (The notes call these "bin centers", which is a mislabel — the formula gives right edges.) The decoder task calls for a continuous, time-varying input, so a ramp is used rather than a binary onset flag.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: bin *k* of the neural array counts spikes in `[stimOn − 0.5 + 0.02k, stimOn − 0.48 + 0.02k)` and element *k* of the time input is `−0.48 + 0.02k`, i.e. the right edge of that same bin. The two arrays are index-for-index the same bins (the label is offset by half a bin from the bin centre, matching the reference code's convention rather than a bin-centre convention).

ii.
```python
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
```
```python
time_since_stim = np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS).astype(np.float32)
```

iii. Both the neural binning and the time vector are generated from the same `WINDOW`/`BINSIZE`/`N_BINS` constants and the same alignment event, so no further alignment step is required.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table: it is constant within a block, so any change of value marks the start of a new block. It is computed on the **full, unfiltered** trials table, then subset to the surviving trials.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    """Compute trial number within each block of constant probabilityLeft.

    A block change occurs when probabilityLeft changes from one trial to the next.
    Trial number is 1-indexed within each block.
    """
    pLeft = trials_df['probabilityLeft'].values
```

iii. The trials table has no block id column, so the blocks have to be recovered from the prior. The agent's note (CONVERSION_NOTES §8) says the variable is the "Count of trial within its block of constant probabilityLeft, starting from 1".

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop over all trials: the counter resets to 1 whenever `probabilityLeft` differs from the previous trial, otherwise it increments. Because the count is taken before trial filtering, discarded trials still advance the counter, so the value is the animal's true position in the block. The resulting per-trial scalar is then indexed by the trial mask and by the behaviour mask, and finally broadcast across the 100 time bins as row 1 of the input. Observed range in the full dataset: 1 to 99.

ii.
```python
    block_counter = 1
    for i in range(len(pLeft)):
        if i == 0 or pLeft[i] != pLeft[i-1]:
            block_counter = 1
        trial_num[i] = block_counter
        block_counter += 1

    # Return only masked trials
    return trial_num[mask]
```

```python
trial_num_in_block = compute_trial_number_in_block(trials, mask.values)
trial_num_in_block = trial_num_in_block[combined_mask]
```

```python
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
```

iii. The decoder task asks for "Trial number in block, continuous, per-trial"; the instructions require per-trial inputs to be given as `(d_input, n_timepoints)`, hence the broadcast across bins (CONVERSION_NOTES §8, §10).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table (values +1, −1, 0). No-response trials (`choice == 0`) have already been removed by the trial mask, so only ±1 reach the recoding.

ii.
```python
# Choice: left=-1 -> 0, right=1 -> 1 (as per decoder spec)
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

```python
'output_names': ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy'],
'output_values': [
    ['left', 'right'],  # choice: 0=left, 1=right
```

iii. CONVERSION_NOTES §7: "Decoder task specifies 'left = 0, right = 1'. In IBL data, choice=-1 is left, choice=1 is right." The agent asserted this convention without checking it against ibllib or the papers; nothing in the trajectory shows it being verified.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding `+1 → 1`, everything else `→ 0`, cast to int64, then broadcast across the 100 time bins so that the output array is `(4, 100)` per trial. Resulting distribution over the full dataset: 0 → 49.2 %, 1 → 50.8 %.

ii.
```python
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
...
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. The instructions require categorical outputs and say "If at all possible, make it time-varying"; the agent's note (CONVERSION_NOTES §10) explains that per-trial variables are replicated across time bins so that all four outputs share one `(4, T)` layout.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 and 0.8. Trials with NaN `probabilityLeft` are already excluded by the NaN mask.

ii.
```python
# Prior: probabilityLeft -> 0.2->0, 0.5->1, 0.8->2
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
```

iii. CONVERSION_NOTES §7: the decoder task specifies the mapping "0.2 -> 0, 0.5 -> 1, 0.8 -> 2"; `probabilityLeft` is the block prior that the task holds constant within a block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. An array of zeros is allocated and the three values are assigned by float equality (`== 0.2`, `== 0.5`, `== 0.8`); the result is broadcast across the 100 bins. Note that any value outside those three would silently stay 0 rather than raise — in practice the observed distribution (0.418 / 0.140 / 0.442) matches the expert's (0.418 / 0.141 / 0.442), so no trial was mis-assigned.

ii.
```python
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
...
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. Direct implementation of the mapping given in the decoder task; the 0.5 class is rare because the unbiased block is only the first 90 trials of a session (the agent used this as a sanity check, CONVERSION_NOTES sanity check 6).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. the `_ibl_wheel.position` / `_ibl_wheel.timestamps` datasets as processed by the loader; speed is the absolute value of the loader's `velocity` column, on the loader's `times` grid.

ii.
```python
# Load wheel speed
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. CONVERSION_NOTES §7: "Reference code's `load_target_behavior` for 'wheel-speed' computes `np.abs(sess_loader.wheel['velocity'].to_numpy())`" — the AI's two lines are that function inlined.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) Inside `SessionLoader.load_wheel`, the irregularly sampled wheel position is interpolated onto a uniform 1 kHz grid and differentiated with a low-pass filter to give velocity (loader default); the AI takes `|velocity|`. (2) Per trial, the samples within `[t_start − binsize, t_end + binsize]` are selected and linearly interpolated (`interp1d`, `fill_value='extrapolate'`) onto the 100 bin right-edges of that trial; trials failing the coverage/NaN checks are dropped. (3) After all sessions are processed, the values are discretised into three classes (see 7-c).

ii.
```python
        # Get behavior data in window (with small buffer)
        beh_mask = (beh_times >= t_start - binsize) & (beh_times <= t_end + binsize)
        bt = beh_times[beh_mask]
        bv = beh_values[beh_mask]
        ...
        # Interpolate to bin centers (matching reference code)
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
        try:
            f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
            binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

iii. `interpolate_behavior_to_bins` is a re-implementation of the reference `get_behavior_per_interval`: same linear `interp1d` with extrapolation, same `x_interp` formula, same skip conditions. The velocity filtering is whatever `SessionLoader` does by default, i.e. identical to the reference (CONVERSION_NOTES §9).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes at the 33.3rd and 66.7th percentile — but the percentiles are computed **globally**, pooling every time bin of every trial of all 444 sessions, and the same two edges are then applied to every session. The full run produced edges [0.0150, 0.4013] and a globally perfectly balanced 33.3 / 33.3 / 33.3 split; per-session splits deviate from balance (in the 5-session sample: 0.363/0.287/0.350, 0.339/0.278/0.384, 0.263/0.391/0.346, …), but only mildly, because wheel speed is in physical units and thus comparable across sessions.

ii.
```python
# Compute global discretization thresholds for wheel speed and whisker ME
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
...
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
```

```python
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

```python
'output_values': [ ..., ['low', 'medium', 'high'], ...]
'wheel_speed_tercile_edges': wheel_edges.tolist(),
```

iii. Neither paper nor the reference code discretises these variables (they are decoded as continuous regression targets), so the AI followed the instruction "Wheel speed discretized into 3 bins" and chose terciles so the classes are balanced: CONVERSION_NOTES §7, "Discretization: 3 bins using global tercile thresholds", and sanity check 9, "Tercile-discretized, producing balanced 33.3 %/33.3 %/33.3 % distributions as expected." The choice of *global* rather than *per-session* thresholds is stated but never argued for in the notes or the trajectory.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at the same 100 sample points as the neural bins of the same trial — `np.linspace(stimOn − 0.5 + 0.02, stimOn + 1.5, 100)`, the right edge of each neural bin — measured from the same `stimOn_times`. Trials whose wheel trace does not span the window (within one bin at each edge) are dropped from the session for all data streams, so the neural, input and output arrays keep the same trial ordering.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

```python
combined_mask = wheel_mask & whisker_mask
...
binned_spikes = binned_spikes[combined_mask]
wheel_binned = wheel_binned[combined_mask]
whisker_binned = whisker_binned[combined_mask]
```

iii. Wheel timestamps are on the same session clock as the spikes, so evaluating the interpolated trace on the neural bin grid is all the alignment needed; the grid formula is taken from the reference `get_behavior_per_interval` (CONVERSION_NOTES §9).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times, via `SessionLoader.load_motion_energy(views=[...])`, taking the `whiskerMotionEnergy` column. The left camera is tried first and the right camera is used as a fallback; if neither loads, the session is skipped (15 sessions were skipped for this reason).

ii.
```python
try:
    sl.load_motion_energy(views=['left'])
    me_times = sl.motion_energy['leftCamera']['times'].values
    me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
    whisker_binned, whisker_mask = interpolate_behavior_to_bins(...)
except Exception:
    try:
        sl.load_motion_energy(views=['right'])
        me_times = sl.motion_energy['rightCamera']['times'].values
        me_values = sl.motion_energy['rightCamera']['whiskerMotionEnergy'].values
        ...
    except Exception as e:
        print(f'  Warning: Could not load motion energy: {e}')
```

iii. CONVERSION_NOTES §7: "Reference code tries left first: `load_target_behavior(one, eid, 'left-whisker-motion-energy')`, falls back to right if unavailable" — this mirrors the reference `bin_behaviors`, which retries with the right camera when the left load returns `skip`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering, no normalisation, no correction for the differing frame rate/resolution of the left and right cameras). It goes through the same `interpolate_behavior_to_bins` as the wheel — window selection with a one-bin buffer, NaN and coverage checks, linear interpolation onto the 100 bin edges — and is then discretised (see 8-c).

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

```python
all_whisker_me_raw.append(whisker_binned)
```

iii. The reference uses the released `whiskerMotionEnergy` directly, so no extra processing is applied; the resampling routine is shared with the wheel and is a re-implementation of `get_behavior_per_interval` (CONVERSION_NOTES §9).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly as the wheel: 3 classes split at the 33.3rd / 66.7th percentile of **all sessions pooled**, one pair of edges ([2.730, 7.837] in the full run) applied to all 444 sessions. Globally this gives 33.3 / 33.3 / 33.3, but because motion-energy units are not comparable across sessions (different cameras — left at 60 Hz full resolution vs right at 150 Hz half resolution — lighting, ROI placement), the per-session distributions are strongly degenerate. In the AI's own 5-session sample file the per-session class fractions are 0.083/0.235/0.683, 0.076/0.205/0.719, 0.308/0.678/0.014, 0.804/0.196/0.000 and 0.500/0.490/0.010 — i.e. some sessions have an essentially empty class.

ii.
```python
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
print(f'Whisker ME tercile edges: {whisker_edges}')
```

```python
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. Same justification as the wheel (CONVERSION_NOTES §7 and sanity check 9): the instructions require three bins, terciles make the classes balanced. The agent checked balance only at the aggregate level ("Behavioral discretization produces perfectly balanced bins (terciles)") and never examined per-session distributions; the trajectory contains no discussion of whether motion energy is comparable across sessions or cameras.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: interpolated onto the same 100 bin edges of each trial relative to `stimOn_times`, and restricted to the trials that survive `combined_mask`, so whisker, wheel, neural and input arrays index the same trials and the same bins.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
...
whisker_arr = whisker_disc[trial].astype(np.int64)
out = np.stack([choice_arr, prior_arr, wheel_arr, whisker_arr], axis=0)
```

iii. Camera frame times are on the same session clock as the spikes, so sampling the trace at the neural bin grid is sufficient (CONVERSION_NOTES §9).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered skipping, never imputation.
- Probe level: a probe whose spike sorting fails to load prints a warning and is dropped; the session continues with the remaining probes. If no probe loads, the session is skipped.
- Trial level: NaN in any of six key trial events, or reaction times out of range, or no-response → trial dropped. Trials whose wheel/whisker trace contains NaN, has fewer than two samples in the window, or does not cover the window within one bin → trial dropped.
- Session level: fewer than 10 trials after the trial mask, fewer than 10 trials after the behaviour mask, or no whisker motion energy at all → session skipped (15 sessions skipped, mostly for missing motion energy).
- Anything else: a catch-all `except Exception` per session prints the traceback and skips that session.
- Residual oddities are left in place: 4 trials whose neural matrix is entirely zero were reported as warnings by the verifier and kept.

ii.
```python
                except Exception as e:
                    print(f'  Warning: Failed to load probe {pid}: {e}')

            if len(spikes_list) == 0:
                print(f'  Skipping: no probes loaded')
                skipped_sessions += 1
                continue
```

```python
            if mask.sum() < 10:
                print(f'  Skipping: only {mask.sum()} valid trials')
                skipped_sessions += 1
                continue
```

```python
            if whisker_binned is None:
                print(f'  Skipping: no whisker motion energy')
                skipped_sessions += 1
                continue
```

```python
        except Exception as e:
            print(f'  ERROR processing session {eid}: {e}')
            traceback.print_exc()
            skipped_sessions += 1
            continue
```

iii. The NaN/coverage rules are the reference code's own skip conditions (`get_behavior_per_interval`, `load_trials_and_mask`). The minimum of 10 trials per session is the AI's own addition (the instructions only require at least two trials per session for evaluation). The agent monitored the skip messages during the run (steps 147, 156) and reported "15 sessions skipped (missing whisker motion energy data)".

## 10-a. What are the most time-consuming steps of the code?

i. Measured by the agent during the run: initially ~40 s/session, of which the per-spike Python binning loop was the largest share; after rewriting the binning with `searchsorted` + `np.add.at` it dropped to ~16 s/session, and the agent then attributed the remaining cost to reading spike sorting from disk ("the data loading (from disk via ONE) is the main bottleneck, not the binning. Each session loads ~1–2 GB of spike data", step 119). The other significant costs are the per-trial behaviour interpolation (a boolean mask over the full ~1 kHz wheel array for every trial), the `np.argsort` of the whole session's spike times inside `bin_spikes_in_window`, and, at the end, pickling ~100 GB to disk. The whole conversion is single-process and took roughly 2 hours.

ii.
```python
    # Sort spike times for fast searchsorted
    sort_idx = np.argsort(spike_times)
    sorted_times = spike_times[sort_idx]
    sorted_clusters = spike_clusters[sort_idx]

    for trial_idx in range(n_trials):
        ...
        np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

```python
        beh_mask = (beh_times >= t_start - binsize) & (beh_times <= t_end + binsize)
```

```python
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent profiled by watching progress (steps 106, 119–121): "46 out of 459 sessions after ~30 minutes… The spike binning is the bottleneck — it's doing a Python-level loop over every spike", then optimised it and verified equality of results before restarting the full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain.
(1) `interpolate_behavior_to_bins` loops over trials and builds a boolean mask over the *entire* session-long behaviour array on each iteration (O(n_trials × n_samples); the wheel is 1 kHz over a full session), and constructs a fresh `interp1d` object per trial. The reference code instead slices with `np.searchsorted` once for all trials, which is what the human solution also does; a single `np.interp` over a concatenated query vector would remove the loop entirely.
(2) `bin_spikes_in_window` still loops over trials; the whole session could be binned with one `np.bincount` over a flattened (trial, unit, bin) index.
(3) `compute_trial_number_in_block` is a pure-Python loop over trials that is one `groupby(...).cumcount()` (or `cumsum` of a change indicator) in vectorised form.
None of these are correctness problems, but (1) is the one with a real cost at this data scale.

ii.
```python
    for trial_idx in range(n_trials):
        ...
        beh_mask = (beh_times >= t_start - binsize) & (beh_times <= t_end + binsize)
        bt = beh_times[beh_mask]
        bv = beh_values[beh_mask]
        ...
        f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
        binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

```python
    block_counter = 1
    for i in range(len(pLeft)):
        if i == 0 or pLeft[i] != pLeft[i-1]:
            block_counter = 1
        trial_num[i] = block_counter
        block_counter += 1
```

iii. The agent did vectorise the inner spike loop when it became the measured bottleneck (step 106–114) but never revisited the behaviour interpolation, presumably because it mirrors the reference implementation's per-interval structure and its cost is masked by the spike I/O.

## 10-c. What processing does the code repeat multiple times?

i. Several small repetitions.
- `np.argsort(spike_times)` is executed inside `bin_spikes_in_window` even though `merge_probes` has already sorted the merged spikes by time (and single-probe spikes come out of the loader sorted) — a full re-sort of tens of millions of spikes per session.
- The cluster-id → row-index map is built twice: once in a Python `for` loop over all clusters, and then again as a redundant `np.where`/fallback block inside the trial loop that computes `cluster_indices` two or three times per trial before overwriting it.
- `np.linspace(t_start + binsize, t_end, n_bins)` and a new `interp1d` object are rebuilt per trial and per behaviour stream, and `time_since_stim` — identical for every trial and every session — is stored per session and `.copy()`-ed per trial.
- Wheel and whisker traces are held in memory twice (in `session_meta` and in `all_wheel_speed_raw`/`all_whisker_me_raw`) so that the global percentiles can be computed in a second pass.

ii.
```python
    cluster_to_idx = np.full(int(cluster_ids.max()) + 1, -1, dtype=np.int32)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_idx[cid] = idx
```

```python
        # Map cluster ids to indices (vectorized)
        valid_mask = clusters_in_window < len(cluster_to_idx)
        cluster_indices = np.where(valid_mask,
                                    cluster_to_idx[clusters_in_window[valid_mask] if valid_mask.all() else clusters_in_window],
                                    -1)
        if not valid_mask.all():
            ci = np.full(len(clusters_in_window), -1, dtype=np.int32)
            ci[valid_mask] = cluster_to_idx[clusters_in_window[valid_mask]]
            cluster_indices = ci
        else:
            cluster_indices = cluster_to_idx[clusters_in_window]
```

```python
        all_wheel_speed_raw.append(wheel_binned)
        all_whisker_me_raw.append(whisker_binned)
        session_meta.append({... 'wheel_binned': wheel_binned, 'whisker_binned': whisker_binned, ...})
```

iii. Not discussed in the notes or the trajectory. The duplicated cluster-index block appears to be leftover scaffolding from the optimisation pass at step 110; since `cluster_ids = np.arange(len(clusters))` the map is the identity anyway.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dominant one is a consequence of the no-QC decision: every row of the cluster table becomes a neuron, including clusters that fire no spike at all inside any trial window and units whose Beryl acronym is `void` (histologically outside the brain) or `root`. About 600k units are binned, stored and written, giving a ~100 GB pickle whose rows are largely all-zero; the expert solution keeps 72k units (mean 164 per session vs 1,351 here) in a far smaller file. `void`/`root` also enter `brain_regions` (280 entries vs the paper's 270), which makes the region index less usable downstream.
Smaller items: `discretize_to_bins()` is defined and never called (the code inlines `np.percentile`/`np.digitize` instead); `ismember`, `Path` and `sys` are imported but unused; `lab` is carried through the pipeline but never written to the output; choice and prior are materialised as 100-long constant vectors per trial (required by the chosen `(4, T)` layout, but 100× redundant); `print_sanity_checks` walks every trial of every session in Python to build full value lists, which is an extra multi-minute pass over 190k trials.

ii.
```python
cluster_ids = np.arange(len(clusters))          # includes clusters with no spikes
beryl_reg = brainreg.acronym2acronym(clusters['acronym'].values, mapping='Beryl')   # keeps 'void'/'root'
```

```python
def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins categories using quantile-based binning.
    ...
    """
```
(never called)

```python
        choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
        prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

```python
    for out_idx, out_name in enumerate(data['output_names']):
        all_vals = []
        for s in range(n_sessions):
            for t in range(len(data['output'][s])):
                ...
                all_vals.extend(trial_out[out_idx].tolist())
```

iii. The agent was aware of the size ("The file is 100GB", step 156) but treated it as expected given the decision to keep all clusters, and did not prune never-spiking or out-of-brain units. The replication of per-trial variables across bins is deliberate and documented (CONVERSION_NOTES §10: "Making all time-varying enables the decoder to handle them uniformly").
