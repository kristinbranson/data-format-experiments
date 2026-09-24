# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It reads every file directly off the local ONE cache
(`/app/data/one_cache`), reconstructing each session directory from the reference repository's release
freeze file `/app/code/code_zhang2025/data/bwm_release.csv` (the same file `0_data_caching.py` uses):
`{lab}/Subjects/{subject}/{date}/{session_number:03d}/alf/`. Inside a session it opens the trials table,
the wheel position/timestamps, the camera motion-energy + camera times, and, per probe listed in the csv,
`alf/{probe}/pykilosort/{revision}/spikes.times.npy` and `spikes.clusters.npy` (plus `clusters.channels.npy`,
`clusters.depths.npy`, `clusters.metrics.pqt`, and the CCF brain-location ids). ONE's "use the latest
revision" rule is re-implemented by hand in `find_file()`/`load_spike_data()`: revision folders (`#...#`)
are listed and sorted in reverse so the most recent ISO-dated revision wins. All 459 sessions in the csv
are iterated in one single-process loop; 438 produced usable data.

ii.
```python
DATA_ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

def find_file(base_path, filename, revisions=None):
    """Find a file in the base path, checking revision directories."""
    direct = os.path.join(base_path, filename)
    if os.path.exists(direct):
        return direct
    if revisions is None:
        revisions = []
        for d in sorted(os.listdir(base_path), reverse=True):
            if d.startswith('#') and d.endswith('#'):
                revisions.append(d)
    for rev in revisions:
        candidate = os.path.join(base_path, rev, filename)
        if os.path.exists(candidate):
            return candidate
    return None
```
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
session_list = []
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((eid, row['lab'], row['subject'], row['date'],
                         row['session_number'], probe_names))
```
```python
session_num_str = str(int(session_number)).zfill(3)
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
```

iii. CONVERSION_NOTES Step 6: "Direct disk loading (bypasses ONE API since cache is read-only)". The
trajectory shows the agent first tried `ONE(mode='local')`, hit write/authentication problems against the
read-only cache, and then mapped the ALF directory layout itself. It kept the reference code's session
list (`bwm_release.csv`) so the set of sessions/probes is exactly the reference's.

## 1-b. How are the data split into subjects (mice)?

i. The subject is taken from the `subject` column of `bwm_release.csv` for each eid — no path parsing,
no inference. `subjects` is built in first-encounter order as sessions are processed, and
`subject_idx[session]` is that subject's position in the list. 135 subjects survive (139 in the release;
4 lost because every session of theirs was skipped).

ii.
```python
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```
```python
'subjects': subject_set,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 treats `bwm_release.csv` as the authoritative session/subject/probe index
("Subjects 139 (from bwm_release.csv)"), so the subject label is read straight from it and nothing is
derived.

## 1-c. How are the data split into sessions?

i. A session is a unique `eid` in `bwm_release.csv`; the csv has one row per probe insertion, so it is
grouped by `eid` and the probe names of a session are collected into a list. That gives 459 sessions,
each processed independently by `process_session()`, and each session becomes one element of
`data['neural']`/`['input']`/`['output']`. Sessions whose data cannot be loaded or that end with <2 usable
trials are dropped (21 dropped, 438 kept).

ii.
```python
session_groups = bwm_df.groupby('eid')
...
    probe_names = list(group['probe_name'])
```
```python
n_good_trials = np.sum(combined_mask)
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
```

iii. CONVERSION_NOTES Step 9: sessions come from the BWM freeze file; "21 skipped (0 valid trials)"; the
agent noted this leaves 438 vs the 433 sessions reported by the methods paper and judged the ~1%
difference acceptable.

## 1-d. How are the data split into trials?

i. Trials are the rows of `_ibl_trials.table.pqt` (latest revision). No re-segmentation is done: each row
supplies `stimOn_times`, and a trial is defined as the fixed window `stimOn_times + (-0.5, 1.5)` s.
Everything (spikes, wheel, whisker) is cut on those intervals, so the trial count per session is the
number of surviving rows of that table.

ii.
```python
def load_trials(session_path):
    """Load trials table from session."""
    alf_path = os.path.join(session_path, 'alf')
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    return pd.read_parquet(trials_file)
```
```python
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. CONVERSION_NOTES Steps 1/4: the reference `bin_spiking_data()` builds intervals the same way
(`trials_df[align_time] + time_window[0/1]`), with `align_time='stimOn_times'` and
`time_window=(-.5, 1.5)` taken verbatim from `0_data_caching.py`.

## 1-e. How are trials filtered based on quality controls?

i. Two masks are ANDed. (1) A re-implementation of the reference `load_trials_and_mask()` with the same
defaults the reference `prepare_data()` uses: drop trials with NaN in any of `stimOn_times`, `choice`,
`feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; require reaction time
(`firstMovement_times - stimOn_times`) in [0.08, 2.0] s; require trial length
(`feedback_times - goCue_times`) ≤ 10 s; drop no-response trials (`choice == 0`). (2) Behavioural
coverage masks from `interpolate_behavior()`: a trial is dropped unless both the wheel and the camera
stream have samples spanning the whole window to within one bin (the reference's "target data starts too
late / ends too early" checks). Sessions left with <2 trials are dropped. 186,261 trials survive across
438 sessions (mean 425/session).

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN, EXCLUDE_NOCHOICE = 0.08, 2.0, 10.0, True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trials_mask(trials_df):
    mask = np.ones(len(trials_df), dtype=bool)
    for event in NAN_EXCLUDE:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT); mask &= (rt <= MAX_RT)
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    mask &= (trials_df['choice'] != 0)
    return mask
```
```python
if np.abs(t_beg - trial_times[0]) > BINSIZE:      # target data starts too late
    continue
if np.abs(t_end - trial_times[-1]) > BINSIZE:     # target data ends too early
    continue
...
combined_mask = mask & wheel_mask & me_mask
```

iii. CONVERSION_NOTES Step 3/Step 4: the rules are copied from `load_trials_and_mask()` (reference
defaults `min_rt=0.08, max_rt=2., exclude_nochoice=True`, and `max_trial_len=10.0` as passed by
`prepare_data`), cross-checked against the data paper's reaction-time truncation at 80 ms and 2 s. The
coverage masks are the reference's `align_spike_behavior`/`get_behavior_per_interval` behaviour
("Ensure neural and behavior data match for each trial").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` of every probe insertion of the session (pykilosort,
latest revision). The cluster tables are read as well, but only `clusters.channels.npy` plus
`electrodeSites.brainLocationIds_ccf_2017.npy` (falling back to `channels.brainLocationIds_ccf_2017.npy`)
are used — for the Beryl region label of each unit, not for the activity. `clusters.metrics.pqt` and
`clusters.depths.npy` are loaded but never used.

ii.
```python
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```
```python
for candidate_dir in [probe_path, spike_dir]:
    for name in ['electrodeSites.brainLocationIds_ccf_2017.npy',
                  'channels.brainLocationIds_ccf_2017.npy']:
        ...
```

iii. CONVERSION_NOTES Step 1/Step 5 maps "spikes.times + spikes.clusters → neural", following the
reference `neural_dict = {'spike_times', 'spike_clusters', 'cluster_regions'}` built in `prepare_data()`.

## 2-b. How is the `neural` data processed?

i. Probes of a session are merged into one population (second probe's cluster ids offset by the first
probe's cluster count, then all spikes re-sorted by time), following the reference `merge_probes()`.
Spikes are then counted into 100 non-overlapping 20 ms bins per trial with a single `np.bincount` over the
flat `cluster * n_bins + bin` index. The stored value is the **raw spike count**, cast to `uint8` — not a
firing rate and not normalised, smoothed, or z-scored. Neurons are all clusters with id `0 … max(id)`,
i.e. the full sorted population of the session (mean 1,360/session, 595,576 total).

ii.
```python
for clusters, spikes in zip(clusters_list, spikes_list):   # merge_probes
    s = {'times': spikes['times'].copy(),
         'clusters': spikes['clusters'].copy() + cluster_max}
    n_clusters = int(spikes['clusters'].max()) + 1 if len(spikes['clusters']) > 0 else 0
    ...
    cluster_max += n_clusters
sort_idx = np.argsort(all_times, kind='stable')
```
```python
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
```
```python
neural_trial = neural_data[trial_idx].astype(np.uint8)  # already (n_clusters, N_BINS)
```

iii. CONVERSION_NOTES Step 10 ("Spike binning: Matches `bin_spiking_data()` — bincount-based approach,
20 ms bins, stimOn alignment, (-0.5, 1.5)s window") and Step 9 ("Using uint8 for neural arrays (spike
counts in 20 ms bins are small integers 0-255)") — the dtype was chosen to fit the 64 GB memory limit.
The reference code also stores counts, so no Hz conversion was done.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality control at all.** Cluster `label`/metrics are never consulted; units labelled
`root` (85,333) and `void` (12,562 units the atlas places outside the brain) and units with no spikes in
any trial are all kept. The metadata records this explicitly:
`'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)'`. This is why the converted
file holds 595,576 units (96% of the 621,733 sorted clusters) instead of the ~75,708 "well-isolated"
units the data paper curates, and why the pickle is 26.7 GB.

ii.
```python
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)   # loaded, never used for filtering
```
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
```
```python
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. CONVERSION_NOTES Step 4 lists this as an explicit discrepancy: "Neuron filtering | qc=None (all
neurons) | Cluster labels: 0.0-1.0 | Well-isolated neurons (75,708) | **Follow code: use ALL neurons**".
Step 3 records the label distribution it measured (label 1.0 = 7% of clusters) and Step 1 notes
`load_spiking_data()` is called with the default `qc=None` in `prepare_data()`, so the agent decided the
reference code outranks the data paper's curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial window is `stimOn_times + (-0.5, +1.5)` s on the session clock, which IBL has already
synchronised across ephys/behaviour streams, so alignment is a subtraction. Spikes are located with
`searchsorted` on the sorted merged spike times and their bin index is
`floor((t - interval_beg) / 0.02)`, clipped into [0, 99]; bin 0 therefore starts exactly at 500 ms before
stimulus onset for every trial.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```
```python
starts = np.searchsorted(spike_times, interval_begs[valid_idx], side='left')
ends = np.searchsorted(spike_times, interval_ends[valid_idx], side='right')
...
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. CONVERSION_NOTES Step 4: the methods paper describes different alignment events per decoder
(firstMovement for dynamic behaviours), but `0_data_caching.py` uses one unified `stimOn_times`
alignment, and the task instructions say "Temporally align based on stimulus onset", so the agent used
stimulus onset for all streams. The `--show-processing` PSTH plot was used to confirm the
post-stimulus response sits at t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `N_BINS = ceil(2.0 / 0.02) = 100` bins per trial, identical for every trial and session;
`metadata['time_bin_size'] = 20.0` ms. There is no rebinning, smoothing or resampling of the neural data —
spikes are counted once directly into the final grid.

ii.
```python
BINSIZE = 0.02          # 20ms bins
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to alignment event
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```
```python
'time_bin_size': BINSIZE * 1000,  # in ms
'n_time_bins': N_BINS,
```

iii. CONVERSION_NOTES Step 3: "divided into 20-ms bins, producing T = 100" (methods paper) and
`params = {'interval_len': 2, 'binsize': 0.02, ...}` in `0_data_caching.py`; the verification output
confirms T is exactly 100 everywhere.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Nothing beyond the alignment event itself: it is the analytic time axis of the trial window defined by
`stimOn_times` and the (-0.5, 1.5) s / 20 ms grid. The same 100-value vector is written into every trial of
every session, so it carries no session-specific raw variable.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 maps "stimOn_times → input[0] 'time_since_stim_onset',
np.linspace(-0.48, 1.5, 100), computed from bin edges", i.e. it is defined by the binning grid taken from
the reference parameters, not measured from the data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The values are the **right edge** of each 20 ms bin,
`linspace(-0.48, 1.5, 100)` — this is exactly the interpolation grid the reference
`get_behavior_per_interval()` builds (`np.linspace(interval_beg + binsize, interval_end, n_bins)`),
expressed relative to stimulus onset. It is stored as row 0 of a (2, 100) float32 array per trial.

ii.
```python
input_trial = np.zeros((2, N_BINS), dtype=np.float32)
input_trial[0, :] = time_since_onset
input_trial[1, :] = trial_num_in_block[trial_idx]  # broadcast scalar
```

iii. CONVERSION_NOTES Step 1: "Behavior interpolation: linear interpolation to bin centres at
`interval_beg + binsize, ..., interval_end`" — the agent adopted the reference's grid convention and
reused it for the time input so that the input axis and the behavioural samples coincide.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same grid the spikes are binned on. Neural bin *i* covers
`[stimOn - 0.5 + 0.02i, stimOn - 0.5 + 0.02(i+1))`, and `time_since_onset[i] = -0.5 + 0.02(i+1)`, i.e. the
closing edge of bin *i*. Column *i* of `input`, `output` and `neural` therefore all refer to the same
20 ms interval; the labelling convention is the bin's end rather than its centre (a 10 ms offset relative
to a centre convention), consistently for all streams.

ii.
```python
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)   # neural
```
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)                                    # behaviour
```
```python
time_since_onset = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)          # input[0]
```

iii. CONVERSION_NOTES Step 7: the `--show-processing` plots include a "Temporal alignment check" panel
overlaying mean neural activity and wheel speed on this axis; the agent reported "No temporal
misalignment observed".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `probabilityLeft` from the trials table. The IBL trials table has no block id, so a block boundary is
detected as a change in `probabilityLeft` between consecutive rows.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. CONVERSION_NOTES Step 5: "Trial index within block → input[1], compute from probabilityLeft
transitions, custom computation"; Step 3 records that blocks are runs of constant pLeft (0.2/0.5/0.8) with
the first 90 trials unbiased at 0.5.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter: for each trial, `i - index_of_first_trial_of_current_block`. Crucially it is computed
on the **unfiltered** trials table and only afterwards subset with `good_indices`, so a trial that is later
dropped by the QC mask still advances the counter and the number reflects the animal's true position in
the block. The scalar is then broadcast across all 100 time bins as row 1 of `input`. Observed range in the
full dataset: 0–98.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_val:
            current_block_start = i
            current_val = prob_left[i]
        trial_numbers[i] = i - current_block_start
    return trial_numbers
```
```python
# Trial number in block (use full trials_df for correct block computation)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. The in-code comment states the reason ("use full trials_df for correct block computation");
CONVERSION_NOTES Step 7 reports the sample range [0, 94] as a sanity check against the ~20–100 trial block
lengths described in the data paper.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which takes values {-1, +1, 0}. Trials with `choice == 0`
(no response) have already been removed by the trial mask, so only ±1 reaches the mapping.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]       # -1 or 1
```

iii. CONVERSION_NOTES Step 5: "trials.choice → output[0] 'choice'", matching the reference
`bin_behaviors()` which takes `choice = trials_df['choice'].to_numpy()`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A linear recode `(choice + 1) / 2`, i.e. **-1 → 0 and +1 → 1**, documented as "-1 (left) → 0,
1 (right) → 1" with `output_values[0] = ['left', 'right']`. The per-trial value is broadcast across all
100 time bins. This assignment is the opposite of the IBL convention: in the released trials tables,
correct trials with the stimulus on the left have `choice == +1` and correct trials with the stimulus on
the right have `choice == -1` (verified directly on three sessions of the provided cache: `feedbackType==1
& contrastLeft>0 → choice=+1` in 100% of trials). So class 0 in the converted file is labelled "left" but
actually contains rightward choices, and vice versa.

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1
```
```python
output_trial[0, :] = choice[trial_idx]
```
```python
'output_values': [
    ['left', 'right'],          # choice: 0=left, 1=right
```

iii. CONVERSION_NOTES Step 5 Key Decision 8 states "Choice mapping: IBL choice -1 (left) → 0, choice 1
(right) → 1". No justification is given and no check against `contrastLeft`/`contrastRight`/`feedbackType`
appears anywhere in the notes or trajectory — the convention was assumed. The reported sanity check was
only that the marginal distribution is ~50/50 (0.491/0.509), which cannot detect a label swap.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table (the block prior, constant within a block, values
0.2/0.5/0.8). Trials with NaN `probabilityLeft` are already excluded by the NaN mask.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]  # 0.2, 0.5, 0.8
```

iii. CONVERSION_NOTES Step 4: the reference code calls this variable "block"; the agent equated
"prior probability of left" with `probabilityLeft` as the task specification prescribes.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Value-by-value recode 0.2 → 0, 0.5 → 1, 0.8 → 2 exactly as the task specifies, broadcast across the 100
time bins, with `output_values[1] = ['0.2','0.5','0.8']`. Note the array is pre-filled with zeros and only
the three expected values are assigned, so any other value would silently be labelled 0; in practice
`probabilityLeft` only ever takes these three values in the release (verified by the agent in Step 2).
The resulting full-dataset distribution is 0.419 / 0.141 / 0.441, consistent with 90 unbiased trials at the
start of each session.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
```
```python
output_trial[1, :] = prior[trial_idx]
```

iii. CONVERSION_NOTES Step 5: "trials.probabilityLeft → output[1], Map: 0.2→0, 0.5→1, 0.8→2" — taken
directly from the Decoder Task specification; Step 5 planned the check "Block distribution: ~0.25 for 0.2,
~0.5 for 0.5 ... ~0.25 for 0.8", reported in Step 9.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` (the raw rotary-encoder samples). Speed is
the absolute value of the velocity derived from them, matching the reference's
`'wheel-speed' → np.abs(sess_loader.wheel['velocity'])`.

ii.
```python
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
...
speed = np.abs(vel).astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: "Wheel speed = `abs(wheel.velocity)` (absolute velocity)", from
`load_target_behavior()`; Step 5 maps "abs(wheel.velocity) → output[2]".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The brainbox `SessionLoader.load_wheel()` pipeline is re-implemented from scratch (because ONE is
bypassed): (1) linear interpolation of wheel position onto a uniform 1000 Hz grid
(`interpolate_position`); (2) an 8th-order Butterworth low-pass at 20 Hz applied zero-phase with
`sosfiltfilt`, then `diff * fs` with a leading 0 inserted (`velocity_filtered`); (3) `abs()` to get speed
in rad/s. That trace is then linearly interpolated (with `fill_value='extrapolate'`) onto the 100 bin
times of each trial, and finally discretized (7-c). No smoothing or normalisation is applied beyond the
20 Hz filter.

ii.
```python
fs = 1000
t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)

corner_frequency, order = 20, 8
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                           btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```
```python
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. CONVERSION_NOTES Step 6/Step 10: "Wheel velocity computed matching brainbox: interpolate to 1000 Hz
uniform, Butterworth LP filter (order=8, corner=20 Hz), diff * fs" — the agent read the brainbox source in
`/app/code/ibllib` and copied the parameters so the trace equals what `SessionLoader` would return.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three equal-frequency (tercile) classes computed **per session**: the 33.3rd and 66.7th percentiles of
all binned speed values pooled over every kept trial and every time bin of that session, then
`np.digitize`. Labels are `['low','medium','high']`. The full dataset shows 0.333/0.333/0.333, i.e. exactly
balanced classes within each session.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(valid, quantiles)
    result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
    return np.clip(result, 0, n_bins - 1)
```
```python
wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Discretization of wheel speed and whisker ME: Use 3
equal-frequency (quantile) bins across all trials in a session". The instructions only require "discretized
into 3 bins"; equal-frequency bins were chosen so chance level is 1/3 and no class is degenerate, and
per-session thresholds were chosen because absolute wheel gain/ME units are not comparable across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Per trial the 1 kHz speed trace is sliced to the window with `searchsorted` and interpolated onto
`linspace(stimOn - 0.48, stimOn + 1.5, 100)` — the same 100 timestamps used as `input[0]` and the closing
edges of the neural bins. Both streams live on the same session clock, so subtracting the stimulus onset
is the only alignment needed. A trial is dropped entirely if the wheel stream does not reach within one
bin of either window edge.

ii.
```python
idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
idxs_end = np.searchsorted(beh_times, interval_ends, side='left')
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. CONVERSION_NOTES Step 10: "Behavior interpolation: Matches `get_behavior_per_interval()` — linear
interp1d with gap checks"; the alignment plots in Step 7 were used to confirm the wheel trace lines up
with the neural PSTH.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, falling back to the right camera
(`rightCamera.ROIMotionEnergy.npy` / `_ibl_rightCamera.times.npy`) when the left camera is unavailable —
the same left-preferred fallback as the reference `bin_behaviors()`. The ROI motion energy released by IBL
(the whisker-pad ROI) is used as-is; it is the same array `SessionLoader.load_motion_energy()` exposes as
`whiskerMotionEnergy`. If the values and timestamps have different lengths the camera is treated as
missing.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
if me_file is not None and times_file is not None:
    me = np.load(me_file).flatten(); times = np.load(times_file).flatten()
    if len(me) == len(times):
        return times, me
# Fall back to right camera
me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. CONVERSION_NOTES Step 1/Step 5: "Whisker ME: tries left camera first, falls back to right (matching
reference code)" — mirroring `bin_behaviors()`'s
`load_target_behavior(..., 'left-whisker-motion-energy')` with a right-camera fallback on failure.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the signal itself: the released trace is used unfiltered and unnormalised. It is linearly
interpolated onto the trial's 100 bin timestamps by the same `interpolate_behavior()` used for the wheel
(with the same coverage checks and `fill_value='extrapolate'`), then discretized into 3 classes (8-c). Note
the left and right cameras run at different frame rates (60 vs 150 Hz), which the interpolation absorbs.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

iii. CONVERSION_NOTES Step 5 maps "whiskerMotionEnergy → output[3], Interpolate to bins, discretize into 3
equal-frequency bins", following `get_behavior_per_interval()`; the reference likewise applies no
transformation to the motion-energy trace.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session terciles of all binned ME values pooled over kept trials and time
bins, `np.digitize`, labels `['low','medium','high']`. Full-dataset distribution 0.333/0.333/0.335.

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```
```python
output_trial[3, :] = me_discrete_2d[trial_idx]
```

iii. Same rationale as 7-c (CONVERSION_NOTES Step 5 Key Decision 6): equal-frequency bins give a
well-defined 1/3 chance level, and per-session thresholds are required because motion-energy units depend
on camera, ROI and lighting and are not comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera frame times are on the same session clock as the spikes, the
trace is sliced to `stimOn + (-0.5, 1.5)` and interpolated onto the same 100 timestamps
(`linspace(t_beg + 0.02, t_end, 100)`) that define the neural bins and `input[0]`. Trials where the camera
stream starts more than one bin late or ends more than one bin early are dropped (this is what removes the
21 sessions with no camera data).

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```
```python
combined_mask = mask & wheel_mask & me_mask
```

iii. CONVERSION_NOTES Step 10 Check 6: the interpolation is the reference's
`get_behavior_per_interval()`, whose "starts too late / ends too early" checks the agent reproduced
verbatim so that trial-level coverage decisions match the reference.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent data is handled by skipping at the smallest possible scope, with the whole
session wrapped in a try/except so a single bad session cannot kill the run:
- probe with no `pykilosort` directory or no spike files → that probe is skipped; if no probe loads, the
  session is skipped;
- missing trials table → session skipped;
- wheel position/timestamps of unequal length or fewer than 2 samples → wheel treated as missing, so the
  wheel mask is all-False and the session ends up skipped;
- camera ME/times of unequal length → camera treated as missing (then the right-camera fallback);
- trial windows with NaN edges (NaN `stimOn_times`) are skipped in both binning and interpolation;
- trials not fully covered by a behavioural stream are dropped;
- sessions with <2 usable trials are dropped (21 sessions, all with 0 valid trials);
- histology missing → all units of the probe are labelled `'void'` rather than failing;
- cluster channel indices are clipped into the valid range of the brain-location array.
The file revision ambiguity (several `#date#` folders per dataset) is resolved by always taking the
highest-sorting (most recent) revision.

ii.
```python
try:
    result = process_session(session_info, show_processing=show)
except Exception as e:
    print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
    n_skipped += 1
    continue
```
```python
if len(re_pos) != len(re_ts) or len(re_pos) < 2:
    return None, None
```
```python
if np.isnan(t_beg) or np.isnan(t_end):
    continue
```
```python
chan_ids = np.clip(chan_ids, 0, len(clusters['brain_ids']) - 1)
...
cluster_beryl = np.array(['void'] * n_clusters)
```

iii. CONVERSION_NOTES Step 9 documents the 21 skipped sessions ("All 21 skipped sessions had 'Too few
valid trials (0)' after applying the trial filtering mask ... AND behavior interpolation mask") and the 4
subjects lost with them, and Step 9 also notes the 3 remaining all-zero-neural trials in session 252 as an
accepted warning. Caveat: the final `converted_data.pkl` produces 186,264 format warnings in
`train_decoder_full_out.txt` (one "neural dtype is uint8, expected float32" per trial, plus the 3 all-zero
trials), whereas `verification_full_out.txt` — produced before the uint8 change — reports only 3; that
discrepancy is never reconciled in the notes.

## 10-a. What are the most time-consuming steps of the code?

i. The conversion is single-process and took 1,980 s (33 min) for 438 sessions, ~4.5 s per session. Per
the script's own timing prints the cost is dominated by file I/O and per-session signal preparation:
loading `spikes.times`/`spikes.clusters` off disk (hundreds of MB per probe, ~0.9 s), the 1 kHz wheel
interpolation + zero-phase Butterworth filter over the whole session plus per-trial behaviour
interpolation (`beh` = 0.5–1.0 s), and spike binning (0.3–0.5 s). On top of the loop, every session's
arrays are pickled to a temp file and read back twice, and the final 26.7 GB pickle is written (38.6 s) —
tens of GB of extra I/O created purely by the memory-management strategy.

ii.
```python
print(f"  {eid}: {n_trials} trials, {n_clusters} neurons | "
      f"spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, "
      f"total={t_format_done-t0:.1f}s")
```
```python
tmp_file = os.path.join(tmp_dir, f'session_{n_processed:04d}.pkl')
with open(tmp_file, 'wb') as f:
    pickle.dump({...}, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. CONVERSION_NOTES Step 7 gives the per-step estimate ("Load spikes 0.9 s, Bin spikes 0.4 s, Load wheel
0.5 s, Other 0.1 s → ~15 min total") and Step 6 records the one optimisation made ("np.bincount linear
indexing for spike binning: 1.6 s → 0.4 s per session"). The realised 33 min is ~2x that estimate, which
the notes attribute to the larger sessions and the memory-driven temp-file round trip.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops remain:
- `bin_spikes_fast()` loops over trials; all trials could be binned with a single `np.bincount` by adding a
  `trial * n_clusters * N_BINS` offset to the flat index;
- `interpolate_behavior()` loops over trials; one `np.interp` over the concatenated query grid would
  replace it (and `scipy.interp1d` is constructed once per trial);
- `compute_trial_number_in_block()` is a pure Python loop over trials, replaceable by
  `groupby(cumsum(diff != 0)).cumcount()`;
- the final per-trial loop allocates three small arrays per trial (`np.zeros((2,100))`, `np.zeros((4,100))`)
  rather than slicing pre-built (n_trials, …) arrays;
- in `main()`, the brain-region index is built with a Python loop over **every neuron** that calls
  `list.index()` on a growing list — O(n_neurons × n_regions) linear scans over 595,576 neurons, where a
  dict lookup or `np.unique(..., return_inverse=True)` is O(n);
- `bin_spikes_vectorized()` is dead code containing a per-spike Python loop (never called).
Beyond loops, sessions are independent but are processed sequentially — no multiprocessing — whereas the
reference pipeline and the human solution use a worker pool.

ii.
```python
for i, trial_idx in enumerate(valid_idx):       # bin_spikes_fast
    ...
for trial_idx in range(n_trials):               # interpolate_behavior
    ...
for i in range(len(prob_left)):                 # compute_trial_number_in_block
    ...
```
```python
session_region_idx = np.zeros(result['n_neurons'], dtype=np.int32)
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
    if region not in region_set:
        region_set.append(region)
    session_region_idx[neuron_idx] = region_set.index(region)
```

iii. CONVERSION_NOTES Step 6 only claims the binning speedup ("np.bincount linear indexing ... 4x faster
than np.add.at") and "Vectorized searchsorted for trial boundaries"; the remaining loops and the absence of
parallelism are not discussed, and the instruction to parallelise/optimise if the full run exceeds 15
minutes was not revisited when the run took 33 minutes.

## 10-c. What processing does the code repeat multiple times?

i. - `BrainRegions()` (the Allen/Beryl atlas tables) is constructed **inside** `process_session()`, i.e.
  re-read from disk once per session (438 times) instead of once per process.
- Every session's arrays are pickled to a temp file, then the temp files are read **twice**: pass 1 loads
  each one only to read `eid`, `subject_idx`, `n_trials`, `n_neurons`, and pass 2 loads them all again to
  build the final dictionary — ~26 GB read twice and written once, purely for metadata that was already in
  memory in `result`.
- `find_file()` re-lists revision directories on each call (trials, wheel position, wheel timestamps, ME,
  camera times — five separate scans of the same `alf` directory).
- Spike binning and behaviour interpolation are run over **all** trials of a session and the QC mask is
  applied afterwards, so the ~30–40% of trials that will be discarded are fully processed first.

ii.
```python
# 4. Get brain regions using Beryl mapping
br = BrainRegions()
```
```python
# Pass 1: collect lightweight metadata
for meta in session_meta:
    with open(meta['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
    all_eids.append(sess['eid']); ...
...
for idx in range(batch_start, batch_end):        # Pass 2: same files again
    with open(session_meta[idx]['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
```
```python
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters)     # all trials, mask applied later
...
neural_data = binned_spikes[good_indices]
```

iii. The notes do not identify any repeated processing. The temp-file/two-pass design is justified in
Step 9 only as a memory measure ("Saving each session to a temporary pickle file during processing;
Reassembling from temp files at the end in batches of 50") after the first full runs were killed by the
64 GB cgroup limit.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. - **Neurons that are never used**: with no QC, 595,576 units are binned and stored, including 85,333
  `root`, 12,562 `void` (localised outside the brain), 344 `x`, all label-0 clusters (~12% of clusters) and
  clusters that emit no spike in any trial (three trials in session 252 are entirely zero). This is ~8x more
  data than the data paper's curated population and is what drove the 26.7 GB file; it then forced the
  agent to subsample 200 of 438 sessions to train at all, so more than half the converted dataset was
  discarded downstream.
- **Trials that are then dropped**: spikes are binned and behaviour interpolated for every row of the
  trials table before the QC mask is applied.
- **`clusters.metrics.pqt` and `clusters.depths.npy`** are read for every probe and never used (metrics
  would only matter if QC were applied); `merge_probes()` builds `merged_channels`/`merged_depths` lists
  that are never returned.
- **The uint8 cast** saves file size but the decoder converts the arrays straight back to float32 at
  training time (186,261 "expected float32" warnings), so the memory problem it was meant to solve
  reappears during training.
- The temp-file write/read round trip (10-c) and, in `--show-processing` mode, a 12-panel figure per
  session.

ii.
```python
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)     # never used
```
```python
merged_channels = []
merged_depths = []                                          # never returned
```
```python
neural_trial = neural_data[trial_idx].astype(np.uint8)
```
```python
# Subset per-session data fields  (train_decoder_lowmem.py)
selected_idx = np.sort(rng.choice(nsessions_total, size=max_sessions, replace=False))
data['neural'] = [data['neural'][i] for i in selected_idx]
```

iii. The notes do not flag any of this as unnecessary; the neuron decision is defended in Step 4 as
following the reference code's `qc=None`, and Step 11 presents the 200-session subsample as a memory
workaround ("The full dataset (438 sessions, 25 GB pickle) could not be trained in 64 GB ... Subsample
200 of 438 sessions"), asserting without test that "random selection ... does not introduce systematic
bias".
