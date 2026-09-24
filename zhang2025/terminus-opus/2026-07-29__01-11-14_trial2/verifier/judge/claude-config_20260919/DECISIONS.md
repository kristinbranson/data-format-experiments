# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It reads the session list from the reference repository's freeze file `code/code_zhang2025/data/bwm_release.csv` (699 PIDs / 459 sessions / 139 subjects), groups the rows by `eid` to get one entry per session with its list of probes, and then resolves each session to a directory on disk by concatenating `data/one_cache/<lab>/Subjects/<subject>/<date>/<00N>/alf`, trying only session numbers `001`, `002`, `003`. Sessions whose folder is not found are dropped (454 of 459 remain; the 5 lost are sessions recorded with `number >= 4`, which exist in the release). Every file is then read with `np.load`/`pd.read_parquet` directly. A helper `find_versioned_file` handles ALF revision folders (`alf/#2024-05-06#/...`) and is used for the trials table and the camera files, but **is not used for the wheel**, which is looked up only at the flat path. Loading is single-process, session by session, in a plain `for` loop.

ii.
```python
BASE_PATH = Path('data/one_cache')
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

def find_session_path(lab, subject, date):
    """Find the session path in the ONE cache."""
    for sess_num in ['001', '002', '003']:
        p = BASE_PATH / lab / 'Subjects' / subject / date / sess_num / 'alf'
        if p.exists():
            return p
    return None

def find_versioned_file(base_dir, filename):
    direct = base_dir / filename
    if direct.exists():
        return direct
    versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
    for vd in versioned_dirs:
        f = vd / filename
        if f.exists():
            return f
    return None
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    sessions.setdefault(eid, []).append({'pid': row.pid, 'probe_name': row.probe_name,
                                         'subject': row.subject, 'lab': row.lab, 'date': row.date})
available_sessions = {}
for eid, probes in sessions.items():
    sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
    if sess_path is not None:
        available_sessions[eid] = probes
```

```python
for i, (eid, probes) in enumerate(available_sessions.items()):
    result = process_session(eid, probes, show_processing=args.show_processing)
```

iii. CONVERSION_NOTES.md Step 6: "Direct file loading (no SpikeSortingLoader dependency)", and Step 10 Check 3 asserts "Data loading: Direct file loading matches SpikeSortingLoader output". The trajectory (step 25) shows the AI first tried to initialise ONE against the local cache and then decided to read the ALF files itself. The consequence was not investigated: 63 sessions were reported as "no wheel data" and the AI's own analysis (trajectory steps 88–89) concluded "this is likely a lab or data collection batch that doesn't have wheel data in the local cache". That conclusion is wrong — checking `one_cache/Brainwidemap/datasets.pqt` shows all 459 sessions have `_ibl_wheel.position.npy`, and the 63 failing sessions are exactly the 63 whose default-revision wheel file lives in an `alf/#date#/` folder that `compute_wheel_speed` never looks in. Net result: 378 of 459 sessions converted, versus 444 for the human reference.

## 1-b. How are the data split into subjects?

i. The subject name is taken from the `subject` column of `bwm_release.csv` (carried on each probe record and read off the session's first probe). At assembly the unique subject names are sorted and `subject_idx` holds each session's index into that list. 125 subjects survive (139 in the release; 14 lose all their sessions to the failures described in 1-a).

ii.
```python
result = {'eid': eid, 'subject': first_probe['subject'], 'lab': first_probe['lab'], ...}
```

```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_list.append(subject_to_idx[sess['subject']])
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. Not explicitly justified; the release file already carries a unique subject id per session so nothing has to be derived. CONVERSION_NOTES Step 9 records "Subjects | 139 | 125 | ~Yes (14 had no valid sessions)".

## 1-c. How are the data split into sessions?

i. A session is the unit of the release: rows of `bwm_release.csv` are grouped by `eid`, so each `eid` becomes one session and its multiple rows become the probe list for that session. The two probes of a dual-probe session are merged into a single population rather than being treated as separate sessions (see 2-b).

ii.
```python
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})
```

iii. CONVERSION_NOTES Step 2: "Session index: `bwm_release.csv` with 699 PIDs, 459 sessions, 139 subjects". The grouping by `eid` follows the reference code's `prepare_data`, which also works per `eid` and merges probes ("when merging probes we are interested in eids, not pids").

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` has one row per trial, so the split is given by the data. Trial intervals are then built from `stimOn_times` of the surviving rows.

ii.
```python
trials_df = pd.read_parquet(trials_file)
...
valid_trials = trials_df[trials_mask].copy()
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. No justification needed/given; the trials table is already one row per trial, as in the reference code (`load_trials_and_mask`).

## 1-e. How are trials filtered based on quality controls?

i. Two stages. **Stage 1** reproduces the reference `load_trials_and_mask` with the arguments `prepare_data` passes: reaction time (`firstMovement_times - stimOn_times`) must be between 0.08 s and 2.0 s; trial length (`feedback_times - goCue_times`) must be <= 10 s; no NaN in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`; no-choice trials (`choice == 0`) are dropped. **Stage 2** drops trials whose wheel trace or whisker-ME trace does not cover the 2-s window (first sample later than `t_start + 20 ms` or last sample earlier than `t_end - 20 ms`), which is the reference's `get_behavior_per_interval` skip rule. Sessions left with fewer than 2 trials are dropped. There is no explicit restriction of `probabilityLeft` to {0.2, 0.5, 0.8}.

ii.
```python
MIN_RT = 0.08; MAX_RT = 2.0; MAX_TRIAL_LEN = 10.0; EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT); mask &= (rt <= MAX_RT)
if 'goCue_times' in trials_df.columns:
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN)
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
```

```python
# in interpolate_behavior(): coverage test, per trial
if np.abs(t_start - seg_times[0]) > binsize:
    valid_mask[trial_idx] = False; continue
if np.abs(t_end - seg_times[-1]) > binsize:
    valid_mask[trial_idx] = False; continue
...
combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None
```

iii. CONVERSION_NOTES Step 1: "Trial filtering: min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True" taken from `load_trials_and_mask` defaults plus the `prepare_data` call; Step 3 repeats it as the trial curation rule. The coverage rule is documented in the code as "Check that behavior data covers the interval (matching reference code)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's `alf/<probe>/pykilosort/#rev#/` folder. `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` are loaded only to give each cluster an anatomical label (peak channel -> CCF id -> Beryl acronym); they do not enter the neural array. `clusters.metrics`/`label` is **not** loaded.

ii.
```python
spike_times = np.load(spike_dir / 'spikes.times.npy').flatten()
spike_clusters = np.load(spike_dir / 'spikes.clusters.npy').flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
n_clusters = len(cluster_channels)
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

```python
br = BrainRegions()
cluster_acronyms_raw = br.id2acronym(cluster_brain_ids)
cluster_acronyms_beryl = br.acronym2acronym(cluster_acronyms_raw, mapping='Beryl')
```

iii. CONVERSION_NOTES Step 5 maps "Spike times + clusters -> neural"; Step 1 notes the Beryl mapping is what the reference uses (`BrainRegions().acronym2acronym(acronyms, mapping='Beryl')`).

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the [-0.5, +1.5] s window around `stimOn_times`, giving a (n_trials, n_clusters, 100) array of **spike counts** (not divided by the bin width, so the units are counts/bin rather than Hz). No smoothing, no normalisation. When a session has more than one probe, the probes are pooled into one population: the second probe's cluster ids are offset by the first probe's cluster count and the merged spike train is re-sorted by time. Each trial is finally stored as a `float32` (n_neurons, 100) matrix.

ii.
```python
for trial_idx in range(n_trials):
    idx_start = np.searchsorted(spike_times, t_start, side='left')
    idx_end = np.searchsorted(spike_times, t_end, side='left')
    trial_times = spike_times[idx_start:idx_end]
    trial_clusters = spike_clusters[idx_start:idx_end]
    bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
for st, sc, nc, cbi in probe_data:
    all_times.append(st); all_clusters.append(sc + cluster_offset); all_brain_ids.append(cbi)
    cluster_offset += nc
spike_times = np.concatenate(all_times); spike_clusters = np.concatenate(all_clusters)
sort_idx = np.argsort(spike_times, kind='stable')
```

iii. CONVERSION_NOTES Step 3/Step 10: "Spikes binned at 20ms, aligned to stimOn_times, window (-0.5, 1.5)s = 100 bins", "Binning: 20ms bins, 100 bins per trial matches reference". The probe merge mirrors the reference `merge_probes` (which offsets `spikes['clusters']` by `clusters.index.max() + 1` and re-sorts by time).

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron filtering at all.** Every sorted cluster of every probe is kept, including clusters that fail IBL's spike-sorting QC and including clusters whose Beryl acronym is `void` (histology places them outside the brain) or `root`. The result is 539,857 "neurons" over 378 sessions (mean 1,428 per session, max 3,140), of which 71,790 are `root` and 12,524 are `void`. The `label` column that IBL's QC produces was seen by the AI (trajectory step 30: "Cluster metrics has a 'label' column with values like 0.333333, 0.666667, 1.0") and then not used; `clusters.metrics.pqt` is never read by the conversion script.

ii. There is no filtering code. The only cluster-level operation is the region labelling:
```python
cluster_brain_ids = chan_brain_ids[cluster_channels]
...
cluster_acronyms_beryl = br.acronym2acronym(cluster_acronyms_raw, mapping='Beryl')
```
and every cluster index is carried through to the output:
```python
region_idx = np.array([region_to_idx[r] for r in sess['cluster_acronyms_beryl']], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 1: "**No QC filtering**: `load_spiking_data` called with `qc=None` in `prepare_data`"; Step 3: "**Neuron curation**: No QC filtering (all clusters used)"; Step 4 resolves the question as "QC filtering | Use qc=None as in reference code"; Step 10 Check 3 repeats "Neuron filtering: No QC filtering (qc=None) matches reference". This reading of the Zhang et al. code is factually right — `prepare_data` does call `load_spiking_data` without a `qc` argument — but the data paper's headline curation (75,708 well-isolated units out of 621,733, "stringent quality control") is never compared against, and the instructions' explicit prompt ("For electrophysiology neural data, do cells need to be filtered based on quality?") is answered only by reference to the method code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one session clock, so alignment is a subtraction. Per trial the window is `[stimOn_times - 0.5, stimOn_times + 1.5]`; the spikes inside it are found with `searchsorted` and their bin index is `floor((t - t_start) / 0.02)`, i.e. time is measured from `stimOn_times - 0.5` and therefore from stimulus onset. `stimOn_times` is guaranteed non-NaN by the trial mask; a `np.isnan` guard is nevertheless present.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
```

iii. CONVERSION_NOTES Step 3: "Align time | stimOn_times | 0_data_caching.py"; Step 10: "Temporal alignment: stimOn_times + (-0.5, 1.5) matches reference". The decoder-task specification also asks for stimulus-onset alignment. Metadata records `temporal_alignment_event: 'stimulus onset (stimOn_times)'`, `off_start: -0.5`, `off_end: 1.5`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2-s trial, identical for every trial and session. No rebinning, resampling or smoothing of the neural data — spikes are counted once, directly on the final grid. `time_bin_size` is written to metadata as 20.0 ms. (The behavioural traces are resampled onto this same grid, see 7-b/8-b.)

ii.
```python
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)  # relative to stimOn_times
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
...
'time_bin_size': BINSIZE * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 1: "Parameters: binsize=0.02s, align_time='stimOn_times', time_window=(-0.5, 1.5)", taken from the `params` dict of the reference `0_data_caching.py`; Step 3 also cites 20 ms from methods.txt. `int(np.ceil(...))` reproduces the reference comment "np.ceil because we want to make sure our bins contain all data".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From nothing in the raw data beyond the alignment event itself: the input is the fixed time grid of the window around `stimOn_times`, so one vector of 100 values serves every trial of every session.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Time since stimOn | input[0] | linspace(-0.48, 1.5, 100)". The value is defined by the window and bin size, which come from the reference code's `params`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The chosen values are the **right edges** of the 100 neural bins: -0.48, -0.46, ..., 1.50, which is exactly the grid the reference code interpolates behaviour onto (`x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)`). The human reference instead uses bin centres (-0.49 ... 1.49); the two differ by a fixed 10 ms label offset. The same vector object is copied into every trial of every session (a (2, 100) float32 array per trial).

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
...
inp = np.stack([
    sess['time_since_stim'],                                                     # (n_bins,)
    np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),    # (n_bins,)
], axis=0)  # (2, n_bins)
```

iii. The code comment says "Matches the bin centers used in reference code" (the values are in fact the bin ends used by the reference `get_behavior_per_interval`, which the AI had read in trajectory step 15). README: "`input[0]`: Time since stimulus onset (seconds), ranges from -0.48 to 1.5".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same grid the spikes are binned on: bin *k* of the neural matrix covers `[stimOn - 0.5 + 0.02k, stimOn - 0.48 + 0.02k)` and the input value at column *k* is `-0.48 + 0.02k`, the closing edge of that bin. It is also the grid the wheel and whisker traces are interpolated onto, so all four streams share a column index.

ii.
```python
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)   # neural
x_interp = np.linspace(t_start + binsize, t_end, n_bins)                                     # behaviour + input grid
```

iii. Not separately justified; it follows from using one window and one bin size for all streams. CONVERSION_NOTES Step 10 Check 2 reports "Input data: time_since_stim ranges [-0.48, 1.5] as expected".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table: it is constant inside a block, so a change of value marks a new block. No block id exists in the data.

ii.
```python
trial_num_in_block = compute_trial_number_in_block(valid_trials_final['probabilityLeft'])
```

```python
def compute_trial_number_in_block(prob_left):
    """A block is a sequence of consecutive trials with the same probabilityLeft."""
```

iii. CONVERSION_NOTES Step 5: "Trial number in block | input[1] | Count within block, broadcast to time". No further justification given.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop walks the `probabilityLeft` column, resetting a counter whenever the value changes, and numbers trials **from 1**. Crucially the counter is run on `valid_trials_final`, i.e. **after** all trial filtering (reaction-time, no-choice, NaN, and wheel/camera coverage), so dropped trials do not advance the count and the value is the trial's position among *surviving* trials in the block, not its true position in the block the animal experienced. The per-trial scalar is then broadcast across all 100 time bins. Observed range across the dataset: [1, 91].

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
    count = 0
    for i in range(len(prob_left)):
        val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
        if val == current_block:
            count += 1
        else:
            current_block = val
            count = 1
        trial_nums[i] = count
    return trial_nums
```

```python
valid_trials_final = valid_trials[combined_valid].copy()
valid_trials_final.reset_index(drop=True, inplace=True)
...
trial_num_in_block = compute_trial_number_in_block(valid_trials_final['probabilityLeft'])
```

iii. No justification is given for counting after filtering; CONVERSION_NOTES and README describe it only as "Trial number within the current block (constant across time bins)". Because the blocks alternate 0.2/0.8 (with an initial 0.5 block) the block *boundaries* are still detected correctly after filtering; only the counts are compressed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1, -1 or 0. No-choice (0) trials have already been removed by the mask. The AI maps **-1 -> 0 labelled "left"** and **+1 -> 1 labelled "right"**.

ii.
```python
# Output 0: Choice (binary, per-trial) left(-1)->0, right(1)->1
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

```python
'output_values': [
    ['left', 'right'],  # choice: 0=left, 1=right
    ...
```

iii. CONVERSION_NOTES Step 5: "Choice | output[0] | left(-1)->0, right(1)->1"; the spot check in Step 13 only verifies that a raw value of 1.0 becomes 1, not what 1.0 means. The reference code (`bin_behaviors`) passes `choice` through unmapped, so the sign convention had to be supplied by the AI; it was not checked against `contrastLeft`/`contrastRight`/`feedbackType`. IBL's own convention is the opposite: in `brainbox.behavior.training`, "choice == -1 means contrast on right hand side" and `rightward = trials.choice == -1`, so +1 is a leftward choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond that recoding. The per-trial scalar is cast to int and broadcast over the 100 time bins so the output is time-varying in shape but constant within a trial.

ii.
```python
out = np.stack([
    np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64),
    np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64),
    wheel_disc[trial_idx].astype(np.int64),
    me_disc[trial_idx].astype(np.int64),
], axis=0)  # (4, n_bins)
```

iii. Broadcasting per-trial variables across time follows the target-format instruction "If at all possible, make it time-varying". The mapping itself is inverted relative to the IBL convention (see 5-a).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, recoded 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 exactly as the decoder-task specification requires.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. CONVERSION_NOTES Step 5: "Prior prob left | output[1] | 0.2->0, 0.5->1, 0.8->2" — taken straight from the Decoder Task section of the instructions. Step 10 Check 4 reports the resulting distribution (~0.42 / 0.14 / 0.44), consistent with the task design (a 90-trial unbiased 0.5 block at the start of each session, then alternating 0.2/0.8 blocks).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; the scalar is broadcast over the 100 bins like choice. Note the implementation is a sequence of equality assignments over an array pre-filled with zeros, so any value that is not exactly 0.2/0.5/0.8 would silently be labelled "0.2" rather than rejected (NaNs are already excluded by the trial mask, and BWM only uses these three values, so no trial is actually affected).

ii.
```python
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
...
np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64)
```

iii. As in 6-a: the mapping is dictated by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, from which velocity and then speed (`|velocity|`) are computed by re-implementing `SessionLoader.load_wheel`. The two files are looked for **only** at the flat path `alf/_ibl_wheel.position.npy`; the revision-aware `find_versioned_file` helper is not used here, so the 63 sessions whose default-revision wheel files sit in `alf/#date#/` are reported as "no wheel data" and dropped entirely.

ii.
```python
def compute_wheel_speed(sess_path, fs=1000, corner_frequency=20, order=8):
    pos_file = sess_path / '_ibl_wheel.position.npy'
    ts_file = sess_path / '_ibl_wheel.timestamps.npy'
    if not pos_file.exists() or not ts_file.exists():
        return None, None
```

```python
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
if wheel_times is None:
    print(f'  Session {eid}: no wheel data')
    return None
```

iii. CONVERSION_NOTES Step 1: "Wheel processing: interpolate_position(1000Hz) -> velocity_filtered(corner=20Hz, order=8) -> abs()", matching the reference `load_target_behavior('wheel-speed')`. The 63 dropped sessions are recorded in Step 9 as a data-availability fact ("Failures: 63 no wheel data") rather than as a bug.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps, the first three a faithful re-implementation of `brainbox.behavior.wheel`: (1) the irregularly sampled wheel position is linearly interpolated onto a uniform 1000 Hz grid (`interpolate_position`, no gap filling); (2) an 8th-order Butterworth low-pass at 20 Hz is applied with `sosfiltfilt` and differentiated, `*fs`, with a 0 prepended (`velocity_filtered`); (3) speed = `|velocity|` in rad/s; (4) the trace is cut per trial and linearly interpolated (`interp1d`, `fill_value='extrapolate'`) onto the 100-point grid of that trial. The AI arrived at this after discovering (trajectory steps 62–66) that a naive `diff(position)/diff(timestamps)` left most trials without coverage because raw wheel timestamps have gaps of up to 80 s.

ii.
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
if len(t_uniform) > 0 and t_uniform[-1] > timestamps[-1]:
    t_uniform = t_uniform[:-1]
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. CONVERSION_NOTES Step 6: "Wheel processing matching SessionLoader.load_wheel() (interpolate_position + velocity_filtered)"; Step 10 Check 3 confirms the same. Comparing against the installed `brainbox.behavior.wheel`, the re-implementation is line-for-line equivalent.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 equal-frequency classes using **global** tercile thresholds computed once over every time bin of every trial of all 378 sessions pooled (0.0145 and 0.4220 rad/s), then applied to all sessions. Globally the classes are exactly balanced (0.333/0.333/0.333); within a session they are not (the fraction of class "low" ranges from about 0.16 to 0.43 across sessions in the verification output). The thresholds are stored in metadata.

ii.
```python
all_wheel = np.concatenate([sess['wheel_binned'].flatten() for sess in session_results])
all_wheel = all_wheel[~np.isnan(all_wheel)]
wheel_quantiles = np.array([-np.inf, np.quantile(all_wheel, 1/3), np.quantile(all_wheel, 2/3), np.inf])
...
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
wheel_disc = np.clip(wheel_disc, 0, 2)
```

iii. CONVERSION_NOTES Step 6: "Global quantile-based discretization for wheel speed and whisker ME"; README: "discretized into 3 equal-frequency bins using global quantiles". No explicit argument for global over per-session is given. For wheel speed the global choice is defensible: speed is in physical units (rad/s) that are comparable across sessions, so a common threshold keeps the class labels physically meaningful.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated at exactly the same 100 time points as the neural bins of the same trial, taken from the same `stimOn_times`, so column *k* of the wheel output and column *k* of the neural matrix describe the same 20 ms bin. Trials whose wheel samples do not span the window (by more than one bin at either edge) are dropped rather than extrapolated.

ii.
```python
idx_start = np.searchsorted(beh_times, t_start, side='right')
idx_end = np.searchsorted(beh_times, t_end, side='left')
...
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. "Interpolate to bin centers (matching reference code)" (code comment); CONVERSION_NOTES Step 3: "Behavioral signals interpolated to same time bins". The wheel is on the same session clock as the spikes, so no further alignment is needed.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`; if either is missing the right camera (`rightCamera.ROIMotionEnergy.npy` + `_ibl_rightCamera.times.npy`) is used instead. Both are found through `find_versioned_file`, so revision folders are handled here. Sessions with neither camera (12 of them) are dropped — a check against the ONE cache index confirms those sessions genuinely have no motion-energy dataset.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
if me_file is None or times_file is None:
    return None, None
```

iii. Code docstring: "Matches reference code: tries left camera first, then right", which is the reference `bin_behaviors` logic (`load_target_behavior(..., 'left-whisker-motion-energy')`, falling back to right on 'skip').

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering, no normalisation). Two clean-ups are applied: the value and time arrays are truncated to their common length, and samples where either is NaN are removed. The trace is then interpolated onto the same 100-point grid per trial as the wheel, with the same coverage test.

ii.
```python
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]; me_times = me_times[:min_len]
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]; me_times = me_times[valid]
...
me_binned, me_valid = interpolate_behavior(me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
```

iii. CONVERSION_NOTES Step 3/README: "Whisker motion energy: Loaded from left camera (fallback to right), interpolated to match neural bins". The length/NaN clean-up is not discussed in the notes; it is a defensive measure against the known left/right camera frame-count mismatches.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly as for wheel speed: 3 classes cut at **global** terciles of the pooled motion energy of all sessions (2.543 and 7.875 in camera units). Because motion energy is an arbitrary-unit quantity that depends on camera (left ~60 Hz high-res vs right ~150 Hz), ROI placement, lighting and mouse position, its absolute scale is not comparable across sessions, and the global cut produces strongly degenerate per-session distributions: in the full verification output the per-session fraction of class "low" ranges from 0.035 to 0.994, and several sessions contain only classes 0 and 1 (output range `[0.0, 1.0]`), i.e. one of the three classes never occurs in those sessions.

ii.
```python
all_me = np.concatenate([sess['me_binned'].flatten() for sess in session_results])
all_me = all_me[~np.isnan(all_me)]
me_quantiles = np.array([-np.inf, np.quantile(all_me, 1/3), np.quantile(all_me, 2/3), np.inf])
...
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
me_disc = np.clip(me_disc, 0, 2)
```

iii. Same justification as 7-c ("Global quantile-based discretization for wheel speed and whisker ME"); the fact that the left and right cameras are pooled on one scale, and the resulting per-session imbalance, are never examined. Step 12 of CONVERSION_NOTES treats the resulting whisker accuracy (0.711) as the best result in the set rather than as a sign that the classes have become a session-identity signal.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: `interpolate_behavior` is called with the same `interval_starts`/`interval_ends` derived from `stimOn_times`, and evaluates the camera trace at the same 100 points, so the columns line up bin for bin with the neural matrix. Trials whose camera frames do not cover the window are dropped (`me_valid`), and the wheel and camera masks are ANDed so both streams are complete for every retained trial.

ii.
```python
me_binned, me_valid = interpolate_behavior(me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
combined_valid = wheel_valid & me_valid
binned_spikes = binned_spikes[combined_valid]
wheel_binned = wheel_binned[combined_valid]
me_binned = me_binned[combined_valid]
```

iii. Camera frame times are on the same session clock as the spikes; no extra alignment is claimed or needed. The coverage test is documented in the code as "matching reference code".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing things are dropped, at three granularities. **Session**: no session folder, no trials table, no probe with spikes, no wheel, no camera motion energy, or fewer than 2 usable trials -> the session is skipped with a printed message (63 + 12 + 1 = 76 of 454 skipped). **Trial**: NaN in any of six key trial-event fields, no choice, out-of-range reaction time or trial length, or wheel/camera not spanning the window -> the trial is dropped. **Sample**: NaN motion-energy samples are removed and ME value/time arrays are truncated to a common length; a `np.isnan` guard skips trials with NaN window bounds; spike bin indices are clipped into range so a spike exactly on the window edge cannot overflow. Any unexpected exception inside `process_session` is caught in `main`, printed with a traceback, and the session is skipped. Trials in which no neuron fired at all are kept (3 such trials in session 326 are reported as warnings by the verifier and left in place). The weak point is that "missing" is decided by a flat-path existence test for the wheel, which mislabels 63 sessions whose wheel files are present but stored under an ALF revision folder.

ii.
```python
if not probe_data:
    print(f'  Session {eid}: no probe data loaded'); return None
if len(valid_trials) < 2:
    print(f'  Session {eid}: too few valid trials ({len(valid_trials)})'); return None
if wheel_times is None:
    print(f'  Session {eid}: no wheel data'); return None
if me_times is None:
    print(f'  Session {eid}: no whisker ME data'); return None
if np.sum(combined_valid) < 2:
    print(f'  Session {eid}: too few valid trials after behavior filtering'); return None
```

```python
try:
    result = process_session(eid, probes, show_processing=args.show_processing)
    ...
except Exception as e:
    print(f'  Session {eid}: FAILED with error: {e}')
    traceback.print_exc()
    n_failed += 1
```

iii. CONVERSION_NOTES Step 9/Step 10 Check 5: "Sessions with missing wheel data handled (skipped) / Sessions with missing whisker ME handled (skipped) / Sessions with too few valid trials handled (skipped) / Zero neural data trials flagged as warnings". The 63 wheel failures were noticed during the run and explained away in the trajectory (steps 88–89) as "likely a lab or data collection batch that doesn't have wheel data in the local cache"; no check against the cache index was made.

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments itself and prints per-session timings for spike binning, wheel processing and motion energy. Across the full run (2,132 s = 35.5 min for 454 sessions, single process) the dominant costs are: (1) reading the two large spike arrays per probe plus the per-trial binning loop with `np.add.at` (0.5–3.2 s per session, the largest single line item on most sessions); (2) the 1000 Hz wheel interpolation and `sosfiltfilt` over the whole session, which is spiky and occasionally dominates (0.3–29 s on long sessions); (3) at the end, pickling a 90.4 GB dictionary to disk, which is pure I/O and by itself takes a large share of the wall clock. Motion energy is negligible (~0.0–0.1 s). No parallelism is used anywhere.

ii.
```python
t1 = time.time()
binned_spikes = bin_spikes_fast(spike_times, spike_clusters, interval_starts, interval_ends,
                                n_clusters, BINSIZE, N_BINS)
t_spike = time.time() - t1
...
print(f'  Session {eid}: {n_trials_final} trials, {n_clusters} neurons, ... '
      f'(spike:{t_spike:.1f}s, wheel:{t_wheel:.1f}s, ME:{t_me:.1f}s, total:{t_total:.1f}s)')
```

iii. CONVERSION_NOTES Step 7: "~4s per session average; Full conversion estimate: ~30 min (actual: 35.5 min)". The instructions asked for optimisation if the estimate exceeded 15 minutes; the AI estimated ~30 min and proceeded without adding parallelism, and the trajectory (steps 91, 97) notes the slow wheel sessions without acting on them.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) `compute_trial_number_in_block` is a pure-Python per-trial loop with `.iloc` indexing; it is a two-line vectorised operation (`(p != p.shift()).cumsum()` then `groupby(...).cumcount()`). (2) The per-trial loop in `bin_spikes_fast` uses `np.add.at`, which is the slowest available accumulator; a single `np.bincount` on a flat `(unit * n_bins + bin)` index, as the human reference does, is far faster, and the whole loop could be done in one pass by additionally offsetting the flat index by the trial. (3) `interpolate_behavior` builds a fresh `scipy.interpolate.interp1d` object per trial; `np.interp` on a single concatenated query vector would avoid the per-trial object construction. (4) The per-trial list comprehensions in `build_final_dataset` re-`np.stack` constant vectors for every trial. Above all, sessions are independent and are processed in a serial `for` loop; a process pool (as in the human reference) would have cut the 35 min by roughly the number of workers.

ii.
```python
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
```

```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
```

```python
for i, (eid, probes) in enumerate(available_sessions.items()):
    result = process_session(eid, probes, ...)
```

iii. CONVERSION_NOTES Step 6 lists "Code inefficiencies identified / Code speedups added" only implicitly; the only speed-up actually documented is replacing a boolean-mask spike selection (`bin_spikes_vectorized`) with `searchsorted` (`bin_spikes_fast`). The remaining loops are not discussed.

## 10-c. What processing does the code repeat multiple times?

i. (1) `BrainRegions()` is instantiated inside `process_session`, so the Allen/Beryl atlas tables are re-read once per session (378 times) although the object is stateless and could be built once. (2) `find_versioned_file` re-scans the session directory for every file it looks up. (3) Spikes are binned for every trial that passed the trials-table mask, including the trials that are subsequently thrown away by the wheel/camera coverage test — that work is done and then discarded. (4) The identical 100-value `time_since_stim` vector is re-stacked and stored separately for each of the 164,322 trials. (5) The behavioural traces are walked twice at assembly time: once to concatenate every value of every session for the global quantiles, once again to `digitize` them.

ii.
```python
# inside process_session, once per session
br = BrainRegions()
cluster_acronyms_raw = br.id2acronym(cluster_brain_ids)
```

```python
binned_spikes = bin_spikes_fast(...)          # all mask-passing trials
...
binned_spikes = binned_spikes[combined_valid]  # some of that work discarded
```

```python
all_wheel = [] ; all_me = []
for sess in session_results:
    all_wheel.append(sess['wheel_binned'].flatten())
    all_me.append(sess['me_binned'].flatten())
```

iii. Not discussed in CONVERSION_NOTES. The repeats are individually small relative to file I/O, except the double pass over all behavioural data, which materialises two ~16 M-element arrays in memory at assembly time.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `_ibl_trials.stimOnTrigger_times.npy` and `_ibl_trials.goCueTrigger_times.npy` are loaded and attached to the trials frame on every session, and neither is ever used (the mask uses `goCue_times` from the parquet table). (2) Three functions are dead code: `bin_spikes_vectorized` (superseded by `bin_spikes_fast`), `discretize_to_bins` (the global quantile code in `build_final_dataset` is used instead), and `merge_probes`, which additionally contains a broken no-op loop left in the file (`for ... in zip(*[iter(x) for x in [spikes_list]]): pass  # This won't work, let me fix`) — the probe merge is duplicated inline in `process_session`. (3) `interpolate_behavior` allocates and returns NaN rows for trials that fail the coverage test, which are then masked away. (4) The biggest item is not code but data: because no quality control is applied (2-c), 539,857 clusters are binned, stored and written, including 12,524 `void` units that the atlas places outside the brain and the large majority of clusters that fail IBL's spike-sorting QC. This is what makes the output 90.4 GB, and none of it is usable signal for the decoder.

ii.
```python
stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
if stim_trig_file is not None:
    trials_df['stimOnTrigger_times'] = np.load(stim_trig_file)   # never read afterwards
```

```python
def merge_probes(spikes_list, clusters_info_list):
    ...
    for spike_times, spike_clusters, n_clusters, cluster_brain_ids in zip(*[iter(x) for x in [spikes_list]]):
        pass  # This won't work, let me fix
```

```python
def discretize_to_bins(values, n_bins=3, quantiles=None):   # never called
```

iii. Not discussed in CONVERSION_NOTES; Step 13 claims the directory was cleaned up, but the dead code and unused loads remain in `convert_data.py`. The file size (90.38 GB) is reported in Step 9 without comment.
