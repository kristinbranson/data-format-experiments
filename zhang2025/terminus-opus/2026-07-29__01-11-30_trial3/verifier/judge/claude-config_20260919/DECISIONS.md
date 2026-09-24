# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API to resolve datasets. It found that `ONE` works offline against the shipped cache but that `eid2pid` requires a network connection, so it built its own file-system resolver instead. The session list comes from the reference code's release table `code/code_zhang2025/data/bwm_release.csv`, grouped by `eid`; each group gives the subject, date, lab and the list of `probe_name`s. A directory walk of `data/one_cache/<lab>/Subjects/<subject>/<date>/<session>/` builds a `(subject, date) -> session path` map, and the two are joined on `(subject, date)`. Every file is then read with `np.load` / `pd.read_parquet` through a `find_file()` helper that looks first in `alf/` and then in the ONE revision folders (`alf/#YYYY-MM-DD#/`), taking `sorted(matches)[-1]` (the latest revision). Spikes come from `alf/<probe>/pykilosort/<latest>/`. All 459 `eid`s in the release table matched a directory; 444 survived processing.

ii.
```python
DATA_DIR = 'data/one_cache'
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

def find_file(base_dir, pattern):
    """Find a file matching pattern in base_dir, handling versioned directories."""
    matches = glob.glob(os.path.join(base_dir, pattern))
    if matches:
        return sorted(matches)[-1]  # Return latest version
    matches = glob.glob(os.path.join(base_dir, '#*#', pattern))
    if matches:
        return sorted(matches)[-1]
    return None

def build_session_map(data_dir):
    """Build mapping from (subject, date) to session directory path."""
    ...
                    if os.path.isdir(sess_path):
                        key = (subject, date)
                        session_map[key] = sess_path
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]; date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({'eid': eid, 'subject': subject, 'date': date,
                              'lab': lab, 'probe_names': probe_names,
                              'path': session_map[key]})
```

iii. From the trajectory (step 55): *"ONE works in offline mode but `eid2pid` requires a remote connection. I'll need to use the `bwm_release.csv` to map eids to pids and probe names, and load data directly from the file paths."* CONVERSION_NOTES Step 2 records the cache layout, and Step 4 records "459 eids in bwm_df / 461 dirs in cache → use 459 from bwm_df".

## 1-b. How are the data split into subjects?

i. Subject identity is taken verbatim from the `subject` column of `bwm_release.csv`; nothing is parsed out of paths. `subjects` is built in order of first appearance while sessions are processed (not sorted), and `subject_idx` is the index into that list for each kept session. Result: 136 subjects over 444 sessions (3 of the paper's 139 mice disappear because all of their sessions were dropped for missing whisker motion energy).

ii.
```python
subject = sess_info['subject']
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
...
all_subject_idx.append(subject_to_idx[subject])
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 9: "Subjects 139 (papers) vs 136 (converted) — 3 subjects lost with skipped sessions". The release table already carries a unique subject id, so no derivation was needed.

## 1-c. How are the data split into sessions?

i. A session is one `eid` in `bwm_release.csv`, i.e. one `<subject>/<date>/<number>/` directory. The release table is grouped by `eid` so that a session with two insertions produces one entry with two probe names rather than two sessions. Each session becomes one element of `neural` / `input` / `output`.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    ...
    probe_names = list(group['probe_name'].unique())
```

iii. Not discussed explicitly; the release table is already one row per (session, probe), so grouping by `eid` is the natural session split. The AI verified "461 session dirs in cache, 459 unique eids in bwm_df, 459/459 matched" (CONVERSION_NOTES Step 2/Step 4).

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` has one row per trial, so no splitting is done; the row index is the trial index. Trial windows are then cut from the continuous streams as `stimOn_times + (-0.5, 1.5)`.

ii.
```python
def load_trials(alf_dir):
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    trials_df = pd.read_parquet(trials_file)
    return trials_df
...
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. No decision needed — the parquet trials table is already one row per trial (CONVERSION_NOTES Step 2).

## 1-e. How are trials filtered based on quality controls?

i. The AI re-implemented `ibllib`'s `load_trials_and_mask` as a pandas `eval` query, with the defaults it verified in the library source (trajectory step 58): drop trials with reaction time (`firstMovement_times - stimOn_times`) `< 0.08 s` or `> 2.0 s`; drop trials with `feedback_times - goCue_times > 10 s`; drop trials with NaN in any of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`; drop no-response trials (`choice == 0`). It deliberately set `EXCLUDE_UNBIASED = False`, keeping the `probabilityLeft == 0.5` trials because the decoder task requires 0.5 as a third prior class. A second, much weaker filter is applied later: a trial is kept only if the wheel and camera streams have ≥ 2 samples within ±1 s of the trial window (otherwise the trace is linearly *extrapolated*, see 9). Sessions with fewer than 2 surviving trials are dropped. Net result: 188,985 trials over 444 sessions (425.6 per session).

ii.
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False  # Keep unbiased trials for prior decoding
EXCLUDE_NOCHOICE = True   # Exclude no-choice trials
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trials_mask(trials_df):
    query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    for event in NAN_EXCLUDE:
        query_parts.append(f'{event}.isnull()')
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    query = ' | '.join(query_parts)
    mask = ~trials_df.eval(query)
    return mask
```

```python
combined_valid = wheel_valid & me_valid
if n_final < 2:
    print(f'  Too few valid trials after behavior filtering')
    return None
```

iii. CONVERSION_NOTES Step 4: *"the reference code default is exclude_unbiased=True … For our decoder task, we need prior with 3 classes including 0.5. So we should set exclude_unbiased=False. But we still need to exclude no-choice trials since choice is an output."* Trajectory step 58 records the correction that the library defaults are actually `min_rt=0.08, max_rt=2.0, exclude_unbiased=False, exclude_nochoice=True`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from `alf/<probe>/pykilosort/<latest>/`. The anatomical label of each unit comes from `clusters.channels.npy` (peak channel of each cluster) indexed into `channels.brainLocationIds_ccf_2017.npy`, then `BrainRegions.id2acronym` → `acronym2acronym(mapping='Beryl')`. `clusters.metrics.pqt` (which carries the `label` QC score) is read during exploration but **not** used by the conversion.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir,
                            'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

```python
acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(acronyms, mapping='Beryl')
```

iii. CONVERSION_NOTES Step 2 lists these four files as the per-probe contents; Step 1 notes the reference's `list_brain_regions` uses the Beryl mapping, which the AI copied.

## 2-b. How is the `neural` data processed?

i. Per session, all probes listed in the release table are loaded and merged into one population: each probe's `spikes.clusters` is offset by the cumulative number of clusters of the previous probes, the concatenated spike times are sorted, and the per-cluster Beryl acronyms are concatenated in the same order. Spikes are then counted into 20 ms bins over each trial's `[stimOn − 0.5 s, stimOn + 1.5 s)` window using `searchsorted` to slice the trial, integer division for the bin index and a flat-index `np.add.at` accumulation. **The values stored are raw spike counts, not firing rates** (there is no division by the bin width), stored as `float32`. No smoothing, no z-scoring, no normalisation.

ii.
```python
def merge_probes_data(probes_data):
    for spike_times, spike_clusters, cluster_channels, channel_brain_ids in probes_data:
        n_clusters = len(cluster_channels)
        all_spike_times.append(spike_times)
        all_spike_clusters.append(spike_clusters + cluster_offset)
        cluster_brain_ids = channel_brain_ids[cluster_channels]
        all_cluster_regions.append(cluster_brain_ids)
        cluster_offset += n_clusters
    ...
    sort_idx = np.argsort(merged_times)
```

```python
def bin_spikes_fast(...):
    i_start = np.searchsorted(spike_times, t_beg, side='left')
    i_end = np.searchsorted(spike_times, t_end, side='left')
    trial_times = spike_times[i_start:i_end]
    trial_clusters = spike_clusters[i_start:i_end]
    bin_idx = np.minimum(((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
    linear_idx = trial_clusters * n_bins + bin_idx
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
```

iii. CONVERSION_NOTES Step 1 identifies `merge_probes` and `bin_spiking_data` as the reference functions and Step 10 Check 3 claims "Binning: 20 ms bins, matching reference". The reference code's `bin_spiking_data` likewise returns counts, so counts (rather than Hz) were kept.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every sorted cluster of every released probe is kept, including clusters whose Beryl acronym is `root` (85,656 units) or `void` (12,827 units, i.e. channels the histology placed outside the brain). The AI explicitly inspected `clusters.metrics.pqt`, saw the IBL `label` column with values 0, 1/3, 2/3, 1, and decided not to use it because the reference code calls `load_spiking_data` without a `qc` argument (`qc=None` → all clusters). The converted dataset therefore contains 599,865 units (≈ the data paper's 621,733 *total* clusters, not its 75,708 QC-passing neurons), a mean of 1,351 units per session, and the pickle is 99.3 GB.

ii. There is no filtering code to quote — every cluster in the table is used:
```python
spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
n_clusters = len(cluster_brain_ids)
...
binned_spikes = bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                                interval_begs, interval_ends, BINSIZE, N_BINS)
```
```python
acronyms = br.id2acronym(cluster_brain_ids)
beryl_regions = br.acronym2acronym(acronyms, mapping='Beryl')   # 'void' and 'root' kept
```

iii. CONVERSION_NOTES Step 1: *"**No QC filtering**: `load_spiking_data` called without qc parameter (defaults to None = all clusters)"*; Step 4: *"QC filtering | qc=None (all clusters) | label values: 0, 0.33, 0.67, 1.0 | 'all neurons sorted by Kilosort 2.5' | No filtering, consistent"*; Step 5 Key Decision 1: *"No QC filtering: Use all clusters as in reference code"*. Trajectory step 44 shows the AI saw the `label` column and chose `qc=None` anyway; step 905 of the notes adds *"ALL clusters are included (including root, void, x regions). This matches our conversion."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. All IBL streams are already on one session clock, so the trial window is simply `interval_begs = stimOn_times - 0.5`, `interval_ends = stimOn_times + 1.5`, and a spike's bin is `floor((t - interval_beg) / 0.02)`, clipped to the last bin. So bin 0 starts exactly at 0.5 s before stimulus onset and bin 25 starts exactly at onset. No resampling or clock correction is applied.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # relative to stimOn_times
...
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
bin_idx = np.minimum(((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
```

iii. CONVERSION_NOTES Step 3: *"Alignment | stimOn_times | 'align trials to the stimulus onset' (methodpaper)"* and Step 1 records the reference parameters `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`. Metadata records `temporal_alignment_event = 'stimulus onset (stimOn_times)'`, `off_start = -0.5`, `off_end = 1.5`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and session (`Min T = Max T = 100` in the verification log). No rebinning, downsampling or smoothing is applied to the neural data — spikes are counted once, directly on the final grid. `metadata['time_bin_size'] = 20.0` ms.

ii.
```python
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```
```python
'time_bin_size': BINSIZE * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 3: *"Neural data time bin | 20 ms | 'divided into 20-ms bins, producing T = 100 time steps' (methodpaper)"*, and Step 1 records `binsize: 0.02` from the reference `0_data_caching.py`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable: it is the analysis grid itself, defined by the alignment event `stimOn_times` and the chosen window/bin size. Because every trial uses the same window, one 100-element vector of bin centres is computed once and copied into every trial.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 variable mapping: *"Time since stim onset | input[0] | np.linspace(-0.5, 1.48, 100) | Computed | Continuous, time-varying"*. The window and bin size come from the reference code parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre vector, which runs from −0.49 s to +1.49 s in 0.02 s steps (verified directly in `sample_data.pkl`: `input[0][0][0, :3] = [-0.49, -0.47, -0.45]`, `[..., -3:] = [1.45, 1.47, 1.49]`). It is stored as `float32` as row 0 of each trial's `(2, 100)` input array. It is kept continuous rather than being turned into a binary onset indicator.

ii.
```python
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)  # (2, n_bins)
input_list.append(inp)
```

iii. The Decoder Task specifies "Time since stimulus onset, continuous, time-varying", so the AI used the continuous bin-centre time rather than a one-hot onset (CONVERSION_NOTES Step 5).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the centre of the same bins the spikes were counted into: neural bin *k* spans `[stimOn − 0.5 + 0.02k, stimOn − 0.5 + 0.02(k+1))` and `input[0][k] = −0.5 + 0.02k + 0.01`. So the two share a time axis bin for bin, with the input marking the middle of each neural bin.

ii.
```python
# neural bins
bin_idx = np.minimum(((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
# input, bin centres of the same grid
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE/2,
                              TIME_WINDOW[1] - BINSIZE/2, N_BINS)
```

iii. Not separately justified; the AI verified in CONVERSION_NOTES Step 10 Check 2 that "Time since stimulus onset ranges from −0.5 to 1.5" and the verification log reports the input-0 range as `[-0.5, 1.5]` for every session.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The IBL trials table carries no block id, so a block boundary is detected as a change in the value of `probabilityLeft` between consecutive rows.

ii.
```python
def compute_trial_number_in_block(prob_left):
    """A block change occurs when probabilityLeft changes value."""
    ...
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
```

iii. CONVERSION_NOTES Step 5: *"Trial number in block | input[1] | Count from block start | Computed from probabilityLeft"*. Step 3 records the task structure: "first 90 trials … equal probability", then 20:80/80:20 blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The position of the trial inside its block, counted **from 1**, computed over the **unfiltered** trials table and only afterwards subsetted by the trial mask — so a dropped trial still advances the counter and the number reflects the animal's true position in the block. The scalar is broadcast across all 100 time bins as row 1 of the input array. Observed range per session is `[1, 85]`–`[1, 99]`, consistent with the 90-trial unbiased opening block plus 20–100-trial biased blocks.

ii.
```python
trial_nums[i] = i - current_block_start + 1
```
```python
# Trial number in block - compute from ALL trials, then select valid ones
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```
```python
np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
```

iii. The code comment states the intent ("compute from ALL trials, then select valid ones"); CONVERSION_NOTES Step 10 Check 2 reports "trial number in block starts at 1".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which takes the values +1, −1 and 0. The AI recoded **−1 → 0 ("left") and +1 → 1 ("right")**, with `choice == 0` already removed by the trial mask. This mapping is the reverse of the IBL convention: checking the raw data (`_ibl_trials.table.pqt` for NYU-46/2021-06-22), every *correct* trial with a left-side stimulus has `choice == +1` (177/177) and every correct trial with a right-side stimulus has `choice == −1` (159/159), i.e. +1 is a leftward report and −1 a rightward one. The stored `output[0]` labels are therefore swapped relative to `output_values = ['left', 'right']` and to the instruction "left = 0, right = 1".

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```
```python
'output_values': [
    ['left', 'right'],           # choice: 0=left, 1=right
```

iii. CONVERSION_NOTES Step 4 asserts the convention without checking it against the data: *"Choice encoding | choice in trials_df | −1 (left), 0 (no choice), 1 (right) | left=0, right=1 | Remap: −1→0 (left), 1→1 (right), exclude 0"*. Step 10 Check 2 only verifies that the stored values are 0/1, not that they carry the right meaning.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the recoding above. The per-trial scalar is broadcast to all 100 bins and stored as `int64` row 0 of the output array.

ii.
```python
out = np.stack([
    np.full(N_BINS, int(choice[t]), dtype=np.int64),
    ...
], axis=0).astype(np.int64)
```

iii. The Decoder Task asks for choice as a per-trial binary variable; the format spec prefers time-varying arrays, so the AI broadcast it over the bins.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes exactly the three values 0.2, 0.5 and 0.8, recoded to 0, 1, 2 with `np.isclose` (float-safe comparison). Trials with NaN `probabilityLeft` were already removed by the mask. Unbiased (0.5) trials are deliberately retained so that class 1 exists.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. CONVERSION_NOTES Step 4: *"Prior variable | block = probabilityLeft | values: 0.2, 0.5, 0.8 | 'prior probability of left' | Map: 0.2→0, 0.5→1, 0.8→2"* — the mapping given in the Decoder Task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the three-way recoding; the scalar is broadcast across the 100 bins as `int64` row 1. Full-dataset distribution is 20.4 % / 60.4 % / 19.2 % for 0.2 / 0.5 / 0.8 — the 0.5 class dominates because the unbiased opening block was kept and the reaction-time mask preferentially keeps early, well-engaged trials.

ii.
```python
np.full(N_BINS, int(prior[t]), dtype=np.int64),
```

iii. CONVERSION_NOTES Step 9 lists the resulting distribution as "20.4/60.4/19.2 — Reasonable". No further processing was considered necessary.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. Position is converted to a velocity with `brainbox.behavior.wheel.velocity_filtered` (the 20 Hz, 8th-order Butterworth low-pass that `SessionLoader.load_wheel` uses), and speed is `np.abs(velocity)`.

ii.
```python
from brainbox.behavior.wheel import velocity_filtered
...
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)
```

iii. CONVERSION_NOTES Step 1: *"Wheel speed: absolute value of wheel velocity"*; the function docstring in `convert_data.py` states "Uses Butterworth-filtered velocity matching the reference code (`SessionLoader.load_wheel`)".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) a sampling rate is estimated as `fs = 1 / median(diff(timestamps))`; (2) `velocity_filtered(position, fs)` is applied **directly to the raw wheel position samples**, and the absolute value taken; (3) the resulting speed trace is linearly interpolated (via `scipy.interpolate.interp1d`, `fill_value='extrapolate'`) onto each trial's 100 sample points.

   The step the reference pipeline performs first — `interpolate_position(timestamps, position, freq=1000)`, which puts the position on a uniform 1 kHz grid — is **omitted**. IBL wheel data is encoder-tick data and is strongly non-uniform: in a representative session the median inter-sample interval is 1.3 ms but 25 % of intervals are more than twice that and the largest gap is 18 s. `velocity_filtered` documents its input as "Vector of uniformly sampled wheel positions", so feeding it raw ticks time-warps the trace. Recomputing that session both ways: the AI's speed has mean 1.06 / median 1.18 rad s⁻¹ and is below 0.01 rad s⁻¹ for only 0.5 % of samples, whereas the reference procedure gives mean 0.35 / median 0.07 rad s⁻¹ and is below 0.01 rad s⁻¹ for 32 % of samples (the mouse is simply still). The two traces correlate at r = 0.51. The quiescent periods and the peak speeds are both lost.

ii.
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
try:
    velocity, _ = velocity_filtered(position, fs)
except Exception:
    ...
speed = np.abs(velocity)
```
```python
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The AI's stated intent (code docstring, CONVERSION_NOTES Step 10 Check 3, Step 12) is "Butterworth-filtered wheel velocity … matching reference code". Trajectory step 192 shows it printed the `velocity_filtered` help text — including "Compute wheel velocity from **uniformly sampled** wheel data" — and then in step 193 applied it to the raw non-uniform positions with `fs = 1/median(diff(timestamps))` without comment.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equal-frequency bins whose edges are the 33.33rd and 66.67th percentiles computed **globally, over the pooled interpolated speed values of all 444 sessions at once**, then applied to every session with `np.digitize`. The edges are stored in the metadata (for the 2-session sample: 0.041 and 0.494 rad s⁻¹). Because the split is global, class fractions vary across sessions (class-0 fraction ranges roughly 0.12–0.58), but no session degenerates to a single class.

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: *"Wheel speed discretization: 3 equal-frequency bins across all data"*. The Decoder Task only says "discretized into 3 bins"; the AI chose equal-frequency so the classes are balanced overall.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated at 100 points per trial given by `np.linspace(t_beg + binsize, t_end, 100)`, i.e. at the **right edge** of each of the 100 neural bins, measured from the same `stimOn_times`. This is exactly the grid the reference code's `get_behavior_per_interval` uses, so it is bin-for-bin consistent with the reference pipeline's binned behaviour; relative to the bin *centres* used for `input[0]` it is a constant +10 ms (half-bin) offset.

ii.
```python
# Match reference code interpolation points
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```
(reference `ibl_data_utils.get_behavior_per_interval`: `x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)`)

iii. The docstring of `interpolate_behavior_to_bins` says "Matches reference code", and the trajectory confirms the reference line was read verbatim. CONVERSION_NOTES Step 10 Check 3: "Temporal alignment: stimOn_times with (−0.5, 1.5) window, matching reference".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with frame times `_ibl_leftCamera.times.npy`; if either is absent the right camera (`rightCamera.ROIMotionEnergy.npy` / `_ibl_rightCamera.times.npy`) is used. 14 sessions had neither and were dropped.

ii.
```python
def load_whisker_me(alf_dir):
    # Try left camera first (matching reference code)
    me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
    if me_file is None or times_file is None:
        # Fall back to right camera
        me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. CONVERSION_NOTES Step 1: *"Whisker motion energy: tries left camera first, falls back to right"*, copied from the reference `load_target_behavior`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI motion-energy trace is used as-is, with no filtering, smoothing or normalisation. Two robustness steps are added: the value and time arrays are truncated to their common length, and samples with NaN in either array are dropped. The trace is then interpolated onto the same 100 per-trial sample points as the wheel, and discretized (8-c).

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]; me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]; me_times = me_times[valid]
```

iii. CONVERSION_NOTES Step 10 Check 5 lists "Handled length mismatches between camera times and ME values" as an edge case it deliberately covered; no additional processing is described because the reference uses the released trace directly.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The same rule as the wheel: 33.33 / 66.67 percentiles computed **globally over all sessions pooled**, applied with `np.digitize`. Because ROI motion energy is in arbitrary camera- and session-dependent units (different rigs, lighting, ROI placement, and left vs right camera), a global threshold does not track the per-session distribution: the verification log shows per-session class-0 fractions from 0.021 to 0.999, i.e. several sessions are assigned essentially a single class for all trials and all time bins (e.g. 0.999, 0.981, 0.968, 0.950). Those sessions carry no within-session information for this output, and the high reported whisker accuracy (0.726) is partly inflated by the decoder being able to infer the session's overall level.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
...
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: *"Whisker ME discretization: 3 equal-frequency bins across all data"*, and Step 9 reports "Whisker ME bins | 3 (equal frequency) | Yes". The per-session degeneracy visible in its own verification output is not mentioned anywhere in the notes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: `np.interp`-style linear interpolation of the camera trace onto `np.linspace(stimOn − 0.5 + 0.02, stimOn + 1.5, 100)`, i.e. the right edge of each neural bin, on the shared session clock. Camera frame times are already synchronised to the ephys clock by the IBL pipeline, so no further correction is applied.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. Same justification as 7-d: the interpolation grid is copied from the reference `get_behavior_per_interval`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled, one is handled questionably and one is mis-diagnosed.
   - Missing files: a probe with no `pykilosort` output or unreadable spike files is skipped (cluster offsets stay consistent because only successfully loaded probes enter the merge); a session with no trials table, no spike data, no wheel or no motion energy returns `None` and is dropped — 14 sessions were dropped for missing whisker ME, 1 for too few trials, giving 444 of 459.
   - Revisions: `find_file` falls back to the `#date#` revision folders and takes the latest.
   - NaNs: NaN in any of six trial columns removes the trial; NaN camera samples and camera/time length mismatches are stripped; NaN trial-window endpoints are skipped in both the binning and the interpolation loops; `velocity_filtered` is wrapped in a `try/except` that falls back to a plain finite difference.
   - **Behavioural coverage is not enforced.** A trial is kept if there are ≥ 2 wheel/camera samples anywhere within ±1 s of its window, and `interp1d(..., fill_value='extrapolate')` then *linearly extrapolates* wheel speed and motion energy for any part of the window not actually covered by the stream, rather than dropping the trial.
   - **Neural coverage is not checked.** The verification log reports session 254, trials 242–244 as all-zero neural data; those are the last three trials of a 245-trial session with 658 units, so the spike stream ends before their windows. CONVERSION_NOTES Step 10 explains them as trials that "have genuinely no neural activity in the window" and marks them unfixable, rather than as uncovered trials that should be dropped.

ii.
```python
if len(probes_data) == 0:
    print(f'  No spike data found'); return None
if me_times is None:
    print(f'  No whisker motion energy data found'); return None
if n_final < 2:
    print(f'  Too few valid trials after behavior filtering'); return None
```
```python
if i_end - i_start < 2:
    valid_mask[trial_idx] = False
    continue
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
```

iii. CONVERSION_NOTES Step 10 Check 5: *"Handled NaN values in trial data through mask; Handled missing whisker ME data by skipping sessions; Handled length mismatches between camera times and ME values; Handled multiple probes per session through merging"*, and Check 1: *"3 warnings for all-zero neural data in session 254 (trials 242–244) — these are edge cases where no spikes occurred in the time window. Cannot fix."*

## 10-a. What are the most time-consuming steps of the code?

i. The conversion ran single-process in 2,031 s (~34 min) for 459 sessions, of which ~1,739 s is inside `process_session`. The dominant costs are (1) reading the two spike arrays per probe — 25–50 M spikes per session, i.e. hundreds of MB of `np.load` per session, which is not instrumented at all; (2) `bin_spikes_fast`, the only step that is timed, at 0.5–6.5 s per session (≈ 400 s total); (3) the per-trial behaviour interpolation, which builds a fresh `scipy.interp1d` object for each of the 188,985 trials; and (4) accumulating the whole 99.3 GB dataset in RAM and pickling it at the end (~290 s of the total, plus the global percentile pass which concatenates every behavioural sample twice). The 99 GB output size — and therefore the pickling cost and the memory footprint — is a direct consequence of keeping all 599,865 unfiltered units (2-c). The AI estimated "~2.5 s/session, ~21 min for 459 sessions" from the 2-session sample, above the 15-minute budget the instructions set, and did not optimise further; actual runtime was 1.6× its estimate.

ii.
```python
t0 = time.time()
binned_spikes = bin_spikes_fast(...)
print(f'  Spike binning: {time.time()-t0:.1f}s, shape={binned_spikes.shape}')
```
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f)
```

iii. CONVERSION_NOTES Step 7 "Run Time Estimates" contains only one row — "Full pipeline | ~2.5 s | ~21 min for 459 sessions" — and Step 6 ("Code inefficiencies identified" / "Code speedups added") was left empty. No bottleneck analysis is documented.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain:
   - `bin_spikes_fast`: one iteration per trial. It is already efficient inside the loop (`searchsorted` slice + one `np.add.at`), and could be made a single `np.bincount` over all trials by offsetting each spike's flat index by its trial, but the gain would be small.
   - `interpolate_behavior_to_bins`: one iteration per trial, and it constructs a `scipy.interpolate.interp1d` object *inside* the loop. Replacing it with a single `np.interp` call (or one vectorised query vector over all trials) would be markedly faster and is the clearest win.
   - `compute_trial_number_in_block`: a pure-Python scalar loop over every trial of the session, which is a two-line vectorised pandas expression (`(p != p.shift()).cumsum()` + `groupby(...).cumcount()`).
   - The per-trial assembly loops that `np.stack` the input and output arrays and append them to lists, run once per trial in two separate passes over the session.
   The AI did replace a slow O(n_spikes) Python loop (`bin_spikes_vectorized`) with the vectorised `bin_spikes_fast`, but left the original in the file.

ii.
```python
for trial_idx in range(n_trials):          # bin_spikes_fast
for trial_idx in range(n_trials):          # interpolate_behavior_to_bins
    interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
for i in range(len(prob_left)):            # compute_trial_number_in_block
for t in range(n_final_trials):            # input assembly
for t in range(n_trials):                  # output assembly
```

iii. Not documented — Step 6's "Code inefficiencies identified" and "Code speedups added" fields are blank. The only implicit statement is the retained, unused `bin_spikes_vectorized` versus the used `bin_spikes_fast` ("Fast vectorized spike binning using searchsorted and histogram").

## 10-c. What processing does the code repeat multiple times?

i. Repeated work is minor but real:
   - The identical 100-element `time_since_stim` vector and the constant `trial_number`, `choice` and `prior` values are materialised into a fresh `(2, 100)` / `(4, 100)` array for each of the 188,985 trials — 100× duplication of four per-trial scalars (the target format does allow per-trial vectors of shape `(n_input,)`).
   - Trials are masked twice and the session is looped over twice (once in `process_session` to build inputs, once in `main` to build outputs after the global percentiles are known), so the per-trial arrays are re-touched in a second pass.
   - The full behavioural arrays are concatenated and flattened once per stream to compute percentiles, and then digitized again per session.
   - `find_file` re-globs the `alf` directory once per dataset name.
   - Two complete spike-binning implementations exist (`bin_spikes_vectorized`, `bin_spikes_fast`) and `discretize_continuous` is defined but never called — dead code rather than repeated computation, but it means two versions of the same logic must be kept consistent.

ii.
```python
inp = np.stack([time_since_stim,
                np.full(N_BINS, trial_nums_final[t], dtype=np.float32)], axis=0)
```
```python
def bin_spikes_vectorized(...):   # never called
def discretize_continuous(values, n_bins=3):   # never called
```

iii. Not documented anywhere in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
   - By far the largest: 599,865 units are binned, stored and written, of which only ~75,708 would survive the data paper's quality control and 12,827 sit outside the brain entirely (`void`). Roughly 8× more spike-binning work, memory and disk (99.3 GB) than the analysis needs, and the extra units are noise for the decoder.
   - `discretize_continuous` computes quantile edges and a digitization that are never used (the equivalent logic was re-implemented inline in `main`), and `bin_spikes_vectorized` is never called.
   - `channel_brain_ids`/`cluster_channels` are loaded and mapped through `id2acronym` + `acronym2acronym` for every cluster including the ones that carry no anatomical information.
   - `interp1d` returns an object with spline machinery that is only used for a single linear evaluation per trial.
   - `wheel_valid`/`me_valid` masks are computed per trial although, because of the `extrapolate` fill, they are almost always all-`True`.
   - `np.random.seed(42)` is set in `--sample` mode but never used (sessions are taken as `sessions_info[:2]`).
   - The `--show-processing` plots are generated after the pickle is written and only for the first two sessions.

ii.
```python
np.random.seed(42)
# Pick sessions that are likely to have good data
sessions_info = sessions_info[:2]
```
```python
def discretize_continuous(values, n_bins=3):
    ...
    return discretized, edges        # never called
```

iii. Not documented. CONVERSION_NOTES Step 9 records the 99.32 GB output and Step 10 Check 4 explains the 599,865 vs 621,733 neuron count as being "due to 15 skipped sessions", without noting that the paper's curated count is 75,708.
