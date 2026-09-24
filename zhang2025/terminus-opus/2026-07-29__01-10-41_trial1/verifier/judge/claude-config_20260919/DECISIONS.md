# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI deliberately bypassed the ONE API and read every file straight off disk with `glob`/`np.load`/`pd.read_parquet`. The session list comes from the reference repo's release table, `code/code_zhang2025/data/bwm_release.csv` (699 probes / 459 sessions / 139 subjects), grouped by `eid` so each row of the loop is one session with its list of probe names. The on-disk location of a session is *reconstructed* from the csv columns as `data/one_cache/{lab}/Subjects/{subject}/{date}/001` — the session number is hard-coded to `001`. Within a session, each stream is found with a different, hand-written path pattern: trials only under a revision sub-directory (`alf/*/_ibl_trials.table.pqt`), spikes under `alf/{probe}/pykilosort/*/`, the wheel only at the top level of `alf/`, and motion energy through a helper `find_file()` that tries both layouts. Sessions whose directory or spikes are not found are skipped. The result was 340 of 459 sessions (50 "directory not found", 69 sessions that reached the loop but ended with "only 0 valid trials" because wheel/motion-energy were not found at the expected path), 119 subjects, 149,339 trials.

ii.
```python
def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    """Find session directory in ONE cache."""
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')
    if os.path.exists(session_dir):
        return session_dir
    return None
```

```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()
```

```python
trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
...
spike_files = glob.glob(os.path.join(
    session_dir, 'alf', pname, 'pykilosort', '*', 'spikes.times.npy'))
...
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file  = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
```

iii. From the trajectory (step 48): "I can successfully load all the data directly from files", and CONVERSION_NOTES Step 6 records the design as "Direct file loading from ONE cache (no API dependency)" with "Flexible file finding for hash-dated subdirectories". The AI verified the layout on a single session (`angelakilab/NYU-11/2020-02-18/001`) and generalised from it. It noticed the shortfall afterwards (step 119): "459 total sessions ... 50 not found ... 68 missing wheel or ME data ... The paper says 433 sessions", and concluded in CONVERSION_NOTES Step 9 "These are data availability issues, not processing errors" — it never cross-checked the claim against the ONE dataset index that ships in the cache (`Brainwidemap/datasets.pqt`).

## 1-b. How are the data split into subjects?

i. The subject identity is taken verbatim from the `subject` column of `bwm_release.csv`, carried through `session_info['subject']` and attached to each session's result. At assembly the AI builds `subjects` in first-appearance order with a dict, and records one index per session in `subject_idx`. No parsing or de-duplication logic beyond the dict is needed because the release table already names each animal. 119 subjects survived.

ii.
```python
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

```python
'subjects': all_subjects,
'subject_idx': np.array(subject_idx_list),
```

iii. Never explicitly justified; the release table supplies a unique subject id per session, so nothing has to be derived. The AI's only related comment is the observation in CONVERSION_NOTES Step 9 that 119 of the expected 139 subjects remain, attributed to "missing sessions affect subject count".

## 1-c. How are the data split into sessions?

i. A session is one `eid` in `bwm_release.csv`; the AI groups the probe-level table by `eid` and aggregates the probe names into a list, so a two-probe session stays one session and its probes are merged later. Each grouped row is then processed independently and appended as one element of `neural`/`input`/`output`.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()
session_groups.rename(columns={'probe_name': 'probe_names'}, inplace=True)
```

```python
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    eid = row['eid']; probe_names = row['probe_names']
```

iii. CONVERSION_NOTES Step 2 records "Probes per session | 1 (219) or 2 (240)" and Step 1 lists `merge_probes` from the reference `ibl_data_utils.py` as the function that "Merges spikes from multiple probes in same session", so grouping the release table by `eid` reproduces the reference's unit of analysis.

## 1-d. How are the data split into trials?

i. No splitting is performed: the trials table (`_ibl_trials.table.pqt`) has one row per trial, and every downstream array is indexed by that row order. Trial windows are built by broadcasting the `stimOn_times` column, so trial *t* of the neural array corresponds to row *t* of the trials table until the quality mask is applied.

ii.
```python
trials_df = pd.read_parquet(trial_files[0])
...
stim_on = trials_df[ALIGN_TIME].values          # ALIGN_TIME = 'stimOn_times'
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. CONVERSION_NOTES Step 2 lists the trials table columns and notes it is already one row per trial; no decision was needed.

## 1-e. How are trials filtered based on quality controls?

i. Two masks are ANDed. The first reproduces the reference repo's `load_trials_and_mask`: reaction time (`firstMovement_times - stimOn_times`) in [0.08 s, 2.0 s]; trial length (`feedback_times - goCue_times`) ≤ 10 s; no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; and `choice != 0` (no-response trials dropped). The second is data availability: a trial is kept only if both the wheel trace and the camera motion-energy trace have at least two non-NaN samples strictly inside the trial window. Sessions left with fewer than 2 valid trials are dropped entirely.

ii.
```python
def create_trials_mask(trials_df):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times',
                   'probabilityLeft', 'firstMovement_times', 'feedbackType']
    mask = pd.Series(True, index=trials_df.index)
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)          # 0.08
    mask &= (rt <= MAX_RT)          # 2.0
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN)   # 10.0
    for event in nan_exclude:
        mask &= ~trials_df[event].isna()
    mask &= (trials_df['choice'] != 0)
    return mask
```

```python
combined_mask = mask.values & wheel_valid & me_valid
n_valid = combined_mask.sum()
if n_valid < 2:
    print(f"  Skipping {eid}: only {n_valid} valid trials")
    return None
```

iii. CONVERSION_NOTES Step 1 identifies `load_trials_and_mask` as the reference's curation function with exactly these parameters ("min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True"), and Step 3 repeats the rules under "Trial curation rules". The behaviour-availability half is justified by Step 1's note that the reference `align_spike_behavior` "Removes trials with missing behavior data, applies trials_mask".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe are the only two arrays that build the neural matrix. `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` are read only to give each unit a Beryl region label and to know how many clusters a probe has (used as the renumbering offset when probes are merged). `clusters.metrics.pqt` (which carries the `label` QC score) is *not* read by the conversion script.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
n_clusters = len(clusters_channels)
brain_ids = np.load(brain_id_files[0]).flatten()
cluster_brain_ids = brain_ids[clusters_channels]
acronyms = br.id2acronym(cluster_brain_ids)
beryl = br.acronym2acronym(acronyms, mapping='Beryl')
```

iii. Trajectory step 48: "Spikes: 20.7M spikes, 898 clusters (neurons) ... Brain location IDs for channels", and step 50 confirms the cluster→channel→`brainLocationIds_ccf_2017`→Beryl chain reproduces `list_brain_regions` from the reference `ibl_data_utils.py` (CONVERSION_NOTES Step 1).

## 2-b. How is the `neural` data processed?

i. Probes are merged first: each probe's `spikes.clusters` is offset by the running cluster count, the concatenated spike times are stably sorted, and the per-cluster Beryl labels are concatenated in the same order. Spikes are then counted into 20 ms bins over the 2 s window of every trial, with `searchsorted` to slice the spikes of a window and `np.add.at` to scatter them into a `(n_clusters, 100)` grid. The stored value is a raw **spike count per 20 ms bin**, not a firing rate — no division by the bin width, no smoothing, no normalisation. The arrays are written as `float32` by the script; the delivered pickle is `uint8` because a separate ad-hoc script re-cast the whole file afterwards.

ii.
```python
spike_clusters = spike_clusters + cluster_offset
cluster_offset += n_clusters
...
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
sort_idx = np.argsort(merged_times, kind='stable')
```

```python
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
times_curr = spike_times[i_start:i_end]
clust_curr = spike_clusters[i_start:i_end]
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. CONVERSION_NOTES Step 3: "Neural: Align to stimOn_times, window (-0.5, 1.5)s, bin at 20ms -> 100 time steps", derived from the reference's `bin_spiking_data` and the Zhang2025 quote "divided into 20-ms bins, producing T = 100 time steps". Keeping counts rather than rates was never discussed; trajectory step 117 only notes "max neural value is 68, so uint8 (0-255) works perfectly".

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control at all.** Every sorted cluster on every probe is kept, including clusters whose Beryl label is `root` (63,769 units in the delivered file) or `void` (11,898 units, i.e. channels the histology placed outside the brain). The delivered dataset therefore averages ~1,450 units per session (range 135–3,140) instead of the ~108 good units per probe reported by the data paper. The AI did read the `label` column during exploration and saw the effect (76 good of 898 on the example probe) but wrote no filter into `convert_data.py`.

ii. There is no filtering code; every cluster of every probe flows through:
```python
all_spike_times.append(spike_times)
all_spike_clusters.append(spike_clusters)
all_cluster_regions.extend(beryl)
```

```python
n_clusters = len(cluster_regions)   # all clusters, unfiltered
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
```

iii. CONVERSION_NOTES Step 1: "The code loads ALL clusters (not just good ones) - qc=None"; Step 3 "Neuron curation rules: Reference code uses ALL clusters (no QC filtering, qc=None) ... 'root' and 'void' regions excluded in some analyses but not in data caching". Step 4 records the conflict with the data paper explicitly and dismisses it: "Code uses ALL clusters, not just good ones. Paper stats about well-isolated are for different analysis". Trajectory step 49 confirms the AI had the numbers in hand: "76 clusters with label >= 1.0 (good), 230 with 0.667, 401 with 0.333, 191 with 0.0".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by subtraction on the shared session clock: the window of trial *t* is `[stimOn_times[t] - 0.5, stimOn_times[t] + 1.5]` in absolute session time, spikes inside it are selected with `searchsorted`, and the bin index is computed from the time elapsed since the window start, so bin 25 always begins exactly at stimulus onset. No resampling or clock correction is applied because spikes, trial events, wheel and camera timestamps are already on one IBL-synchronised clock.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. CONVERSION_NOTES Step 4, "Key Decision: Alignment": "The task description says 'Temporally align based on stimulus onset' for all variables. The reference code (0_data_caching.py) also uses stimOn_times with window (-0.5, 1.5). I will follow this approach." The AI explicitly rejected the paper's per-variable alignments (prior at −0.6→−0.1 s, wheel/whisker at first-movement onset) because the decoder spec requires a single alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`BINSIZE = 0.02`), 100 bins spanning the 2 s window, identical for every trial and every session; `metadata['time_bin_size']` is written as 20.0 ms. Spikes are binned once, directly at 20 ms — there is no finer intermediate binning and no rebinning/resampling step. The verification log confirms Mean/Min/Max T = 100 for all 340 sessions.

ii.
```python
BINSIZE = 0.02  # 20 ms
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to stimOn_times
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

```python
'time_bin_size': BINSIZE * 1000,  # in ms
```

iii. CONVERSION_NOTES Step 3 cites Zhang2025 — "split into 2-s trials", "divided into 20-ms bins, producing T = 100 time steps" — and Step 1 records the reference caching parameters "binsize: 0.02 (20 ms)", "interval_len: 2 seconds". Step 4 notes the paper's 50 ms bins for the prior decoder and overrides it: "Task spec says 20ms bins for all - will use 20ms".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all; it is the analysis grid the AI defined. It depends on `stimOn_times` only indirectly, through the fact that the grid is defined relative to it. The values are the centres of the 100 bins, −0.49 s … +1.49 s, identical for every trial of every session (the verification log reports input 0 range `[-0.5, 1.5]` for all sessions).

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table (recorded in the trajectory): "time since stimOn | input[0] | np.arange(-0.5, 1.5, 0.02) + 0.01 | N/A | Time-varying, same for all trials". The window and bin size are the reference code's decoding parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre vector once per session and copying it into every trial's input array. The AI chose bin centres rather than edges so the value labels the middle of the interval the spike count covers. It is stored as a real-valued time series rather than the binary onset indicator the format spec allows.

ii.
```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],                                   # (1, N_BINS)
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)     # (1, N_BINS)
])  # (2, N_BINS)
input_list.append(inp)
```

iii. No explicit justification; the decoder spec lists the variable as "continuous, time-varying", which is what the AI produced.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid. The neural bin index is `floor((t − (stimOn − 0.5)) / 0.02)`, so bin *k* covers `[−0.5 + 0.02k, −0.5 + 0.02(k+1))` relative to onset, and the input value for bin *k* is `−0.5 + 0.02k + 0.01`, the centre of exactly that interval. The two are aligned bin for bin by construction, with no interpolation or offset.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```
```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)   # t_beg = stimOn - 0.5
```

iii. The `--show-processing` plots draw the neural raster and every behavioural trace on this same `time_since_stim` axis with a dashed line at 0, which the AI used as its visual check that "There are no temporal misalignments".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table carries no block id, so a block boundary is inferred wherever `probabilityLeft` changes from one row to the next.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

```python
if i > 0 and prob_left[i] != prob_left[i-1]:
    counter = 1
```

iii. CONVERSION_NOTES Step 5 mapping: "trial number in block | input[1] | Count from block start | N/A | Per-trial, reset at each block change". Step 3 records the task structure from the data paper ("initial 90 unbiased trials", "block length truncated to lie between 20 and 100 trials") which the resulting range (1 … 85–99 per session in the verification log) is consistent with.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter over **all** trials of the session, starting at 1 and reset to 1 at every change of `probabilityLeft`. It is computed on the unfiltered trials table and only afterwards indexed by the surviving trials, so a trial that is later dropped still advances the count and the number reflects the animal's true position in the block. The per-trial scalar is broadcast across all 100 bins so the input array is always `(2, 100)`.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums
```

```python
# Need to compute on ALL trials first, then select valid ones
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The in-code comment ("Need to compute on ALL trials first, then select valid ones") is the whole justification given; the intent is that the counter measures the animal's real position in the block rather than a position within the filtered subset.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single `choice` column of the trials table, whose raw values are −1, 0 and +1. The 0 (no-response) trials have already been removed by the trial mask, so only ±1 reach the mapping. The result is a per-trial scalar broadcast over the 100 bins.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. Trajectory step 48: "Choice values are -1, 0, 1 (not 0, 1 as needed for decoder)"; CONVERSION_NOTES Step 1 records the reference's `bin_behaviors` as "choice = trials_df['choice'].to_numpy() (values: -1, 0, 1)".

## 5-b. What processing is involved in computing `output` *Choice*?

i. A single recoding, `−1 → 0` labelled **left** and `+1 → 1` labelled **right**, with `output_values[0] = ['left', 'right']`. This is the opposite of the IBL convention: in `brainbox.behavior.training` a rightward choice is `trials.choice == -1` ("# choice == -1 means contrast on right hand side"), so the delivered labels have left and right swapped. Implementation detail: `np.where(choice == -1, 0, 1)` sends *everything* that is not −1 to class 1, which is safe only because `choice == 0` trials were already dropped.

ii.
```python
# Choice: -1 -> 0 (left), 1 -> 1 (right)
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

```python
'output_values': [
    ['left', 'right'],  # choice
    ...
```

iii. Trajectory step 48: "Need to map: -1 -> left (0), 1 -> right (1)", repeated verbatim in the Step 5 mapping table. No source is cited for the direction and no check (e.g. against `contrastLeft`/`feedbackType`) was performed. The AI's only downstream validation was that the class split is ~50/50, which is insensitive to the sign.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three values 0.2, 0.5 and 0.8 (0.5 being the initial unbiased block). One value per trial, broadcast over the 100 bins.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. Trajectory step 48: "probabilityLeft: 0.2, 0.5, 0.8 -> need to map to 0, 1, 2"; CONVERSION_NOTES Step 3 "Prior: probabilityLeft mapped to categories (0.2->0, 0.5->1, 0.8->2)", which is the mapping dictated by the decoder spec.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A dictionary lookup only, `0.2→0, 0.5→1, 0.8→2`, with `output_values[1] = ['0.2','0.5','0.8']`. NaN values never reach it because `probabilityLeft` is in the NaN-exclusion list of the trial mask. Any value outside the three expected ones is silently coerced to class 1 (0.5) by the `dict.get(p, 1)` default rather than dropping the trial. The resulting distribution (~44 % / 12 % / 44 % on the sample) matches the block structure.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

```python
out = np.vstack([
    np.full((1, N_BINS), choice_mapped[i], dtype=np.int64),
    np.full((1, N_BINS), prior_mapped[i], dtype=np.int64),
    ...
```

iii. Trajectory step 77: "Prior distribution: 0.2 (44.3%), 0.5 (11.8%), 0.8 (43.9%) - reasonable given block structure" — the AI's sanity check on the mapping. The mapping itself is prescribed by the decoder spec.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw rotary-encoder streams `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, read directly from `alf/`. Speed is the absolute value of the velocity derived from them. `SessionLoader.load_wheel` was not used (the AI worked without the ONE API), so the loader's internals were re-implemented by hand.

ii.
```python
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. CONVERSION_NOTES Step 1 lists `load_target_behavior` ("Loads wheel speed (abs velocity)"), `interpolate_position` and `velocity_filtered` as the reference functions; Step 1 Notes: "Wheel speed = abs(velocity) where velocity is computed from Butterworth-filtered position". Trajectory step 57: "Since we can't use brainbox's SessionLoader, I need to implement wheel velocity computation. Let me check the brainbox wheel code."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps, the first three a faithful re-implementation of `brainbox.behavior.wheel`: (1) the irregularly-sampled position is linearly interpolated onto an even 1000 Hz grid `np.arange(t0, t_end, 1/1000)` with the same trailing-sample guard as `interpolate_position`; (2) an 8th-order Butterworth low-pass at 20 Hz is applied with `sosfiltfilt` and differentiated, `np.insert(np.diff(...), 0, 0) * fs`, exactly as `velocity_filtered` does; (3) speed = `np.abs(velocity)`; (4) the speed trace is then resampled per trial onto the 100 bin centres by linear interpolation. Sessions whose position/timestamp lengths disagree, or with fewer than 10 samples, return `None` and are dropped.

ii.
```python
t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
if t_interp[-1] > wheel_ts[-1]:
    t_interp = t_interp[:-1]
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)

sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
```

```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean, kind='linear',
                                fill_value='extrapolate')(x_interp)
```

iii. CONVERSION_NOTES Step 1: "velocity_filtered | wheel.py | PROCESSING | Butterworth lowpass filter (order=8, corner=20Hz), then diff*fs" and "interpolate_position | wheel.py | PROCESSING | Interpolates wheel position to 1000 Hz"; the constants `WHEEL_FS/WHEEL_CORNER_FREQ/WHEEL_FILTER_ORDER` are commented "from brainbox wheel.py". Per-trial resampling is attributed to the reference `get_behavior_per_interval` ("Matches get_behavior_per_interval from reference code").

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equal-occupancy classes per session. The 33.3rd and 66.7th percentiles are computed over the pooled `(n_valid_trials × 100)` speed values of that session — i.e. one threshold pair per session, not per trial and not global — and `np.digitize` assigns 0/1/2 (`['low','medium','high']`). NaNs are excluded from the quantile computation. The verification log confirms the intended consequence: every session reports class fractions of exactly 0.333/0.333/0.333.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    if len(flat) == 0:
        return np.zeros_like(values, dtype=np.int64)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
    return result
```

```python
wheel_data = wheel_binned[valid_idx]      # session's surviving trials only
wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
```

iii. CONVERSION_NOTES Step 5 mapping: "wheel speed | output[2] | abs(velocity), interpolate, discretize into 3 bins". Trajectory step 77 records the check: "Wheel speed and whisker ME evenly distributed across 3 bins (by design)". The decoder spec's requirement is "Wheel speed discretized into 3 bins"; quantile boundaries were chosen so no class is starved.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The 1000 Hz speed trace is evaluated at exactly the same 100 bin centres, measured from the same `stimOn_times`, that define the neural bins, so wheel bin *k* and neural bin *k* describe the same 20 ms of session time. No lag or shift is introduced. Only samples strictly inside the window are used for the interpolation, with linear extrapolation covering the first and last bin centre if needed.

ii.
```python
idx_beg = np.searchsorted(beh_times, t_beg, side='right')
idx_end = np.searchsorted(beh_times, t_end, side='left')
beh_t = beh_times[idx_beg:idx_end]
beh_v = beh_values[idx_beg:idx_end]
...
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
```
(`t_beg = stimOn - 0.5`, `t_end = stimOn + 1.5`, the same bounds used for the spikes.)

iii. Both streams are on the one IBL-synchronised session clock, so evaluating the trace on the neural grid is all the alignment required. The `--show-processing` figure plots wheel speed and the neural raster on a shared `time_since_stim` axis with onset marked, which is the AI's stated visual verification of alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The released per-frame ROI motion energy of a side camera, `leftCamera.ROIMotionEnergy.npy` with frame times `_ibl_leftCamera.times.npy`; if the left camera's pair is missing or the two arrays disagree in length, the right camera pair is used instead. A helper searches both `alf/` and `alf/*/` for these files. If neither camera yields a usable pair the session contributes no valid trials and is dropped.

ii.
```python
def load_whisker_motion_energy(session_dir):
    me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
    if me_file and cam_file:
        me = np.load(me_file).flatten()
        cam_times = np.load(cam_file).flatten()
        if len(me) == len(cam_times) and len(me) > 0:
            return cam_times, me
    me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
    ...
    return None, None
```

iii. Docstring: "Try left camera first, then right camera (matching reference code)". CONVERSION_NOTES Step 1 lists `load_target_behavior` as the reference function that "Loads wheel speed (abs velocity), whisker motion energy", and Step 2 records the two file families found in the cache.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used exactly as it is — no filtering, smoothing, baseline subtraction or normalisation. The only operation is the same per-trial resampling used for the wheel: samples inside the trial window are taken, NaN samples are dropped, and the remainder is linearly interpolated onto the 100 bin centres. A trial with fewer than two usable samples in the window is marked invalid and dropped.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

```python
nan_mask = ~np.isnan(beh_v)
if nan_mask.sum() < 2:
    valid[trial_idx] = False
    continue
beh_t_clean = beh_t[nan_mask]
beh_v_clean = beh_v[nan_mask]
```

iii. CONVERSION_NOTES Step 5 mapping records the transform as "interpolate, discretize into 3 bins" with the reference function `get_behavior_per_interval`; the AI's Step 1 reading of the reference found that motion energy is consumed as released, so no extra processing was introduced.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `np.quantile` at 1/3 and 2/3 over all non-NaN values of that session's surviving trials, then `np.digitize` into 0/1/2 with names `['low','medium','high']`. Thresholds are per session, which absorbs between-session differences in camera gain and ROI size; the verification log again shows exactly 0.333/0.333/0.333 for every session.

ii.
```python
me_data = me_binned[valid_idx]
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

```python
quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. Same rationale as the wheel: the decoder spec asks for 3 bins, and equal-occupancy bins make chance exactly 1/3 for every session. Trajectory step 77 lists the resulting even distribution as a passed check.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is sampled at the same 100 bin centres relative to the same `stimOn_times` as the neural bins — the identical `interpolate_behavior_to_bins` call used for the wheel — so camera bin *k* and neural bin *k* cover the same 20 ms. The camera frame times are already on the session clock, so no further correction is applied.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
```

iii. As for the wheel: shared clock plus shared grid. The `--show-processing` plot overlays the continuous and discretized whisker traces on the `time_since_stim` axis with the onset line, as the AI's alignment check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive layers, all of the "drop it" kind. At session level: a missing directory, missing spikes or missing trials table → session skipped and counted; any exception inside `process_session` is caught, traced and the session skipped; a session left with <2 valid trials is skipped. At stream level: `load_wheel_speed` returns `None` if position/timestamp lengths disagree, if there are <10 samples, or if the interpolation grid is empty; `load_whisker_motion_energy` returns `None` unless motion energy and camera times have equal, non-zero length, and falls back from left to right camera. At trial level: NaN `stimOn_times` are detected before binning; NaNs in the six key trial fields are masked out; a trial with <2 non-NaN behavioural samples in its window is dropped; `np.clip` guards a spike landing on the window edge; the per-trial interpolation is wrapped in `try/except`. Two weaknesses: `fill_value='extrapolate'` means a trial whose wheel or camera data covers only part of the window is *extrapolated* rather than dropped, and 3 trials with all-zero neural data (session 190) were reported by the verifier and left unresolved. `warnings.filterwarnings('ignore')` is set globally at import.

ii.
```python
try:
    result = process_session(session_info, session_dir, show_processing=args.show_processing)
except Exception as e:
    print(f"  Error processing session: {e}")
    traceback.print_exc()
    n_skipped += 1
    continue
```

```python
if len(wheel_pos) != len(wheel_ts):
    return None, None
if len(wheel_pos) < 10:
    return None, None
```

```python
if np.isnan(t_beg) or np.isnan(t_end):
    valid[trial_idx] = False
    continue
...
if len(beh_v) < 2:
    valid[trial_idx] = False
    continue
```

iii. CONVERSION_NOTES Step 9: "340 vs 433 sessions: 50 sessions missing data files, 68 missing wheel/ME data ... These are data availability issues, not processing errors". Trajectory step 119 adds the speculation "The Zhang2025 paper may have used sessions without all 4 behavioral variables - they may have used sessions with just choice and prior ... But our task requires all 4 outputs." The three all-zero-neural warnings were noted ("verification_full_out.txt shows only 3 warnings ... No errors reported") but not investigated.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented the script with per-session and per-stage timers and identified spike binning as the bottleneck: the first implementation (per-trial `bincount2D` calls) took ~104 s per session; after rewriting it with `searchsorted` + `np.add.at` it fell to ~1.6 s per session, and the full serial run over 459 sessions took 1,986 s (33 min). Two costs the AI did not call out but which dominate the wall clock of the whole exercise: serialising the 84.7 GB pickle at the end of the run, and the separate follow-up pass that loaded that entire pickle back into memory, re-cast it to `uint8`/`int8` and rewrote it. Sessions are processed strictly serially — no multiprocessing.

ii.
```python
t_bin = time.time()
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
print(f"{time.time()-t_bin:.1f}s")
```
```python
elapsed = time.time() - t0
print(f"  Session processed in {elapsed:.1f}s")
...
total_time = time.time() - t_total
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Initial spike binning used per-trial bincount2D calls (104s/session). Code speedups added: Replaced with searchsorted + np.add.at vectorized binning (1.6s/session)". Trajectory step 73: "I need to optimize the spike binning which is the main bottleneck (104s for one session)."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite the docstring claim "Optimized vectorized version", `bin_spikes_per_trial` is still a Python loop over trials, and inside it `np.add.at` is an unbuffered scatter that is far slower than the flat-index `np.bincount` the reference uses (`flat = unit * N_BINS + bin; np.bincount(flat, minlength=...)`), which would also let all trials be binned in one call by offsetting each spike's index by its trial. Also loop-bound: `interpolate_behavior_to_bins` (one `interp1d` object constructed per trial — a single `np.interp` over a concatenated query vector would do); `compute_trial_number_in_block`, a pure-Python per-trial loop that is a one-line `groupby(...).cumcount()` (or `np.arange - np.maximum.accumulate`) vectorisation; the two per-trial `np.vstack` loops that build `input_list` and `output_list`; and the per-neuron Python loop that maps region names to indices in `main`. Above all, the session loop itself is serial where the reference runs a 10-worker `ProcessPoolExecutor`.

ii.
```python
def bin_spikes_per_trial(...):
    """...Optimized vectorized version."""
    for trial_idx in valid_indices:
        ...
        np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```
```python
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        counter = 1
    trial_nums[i] = counter
```
```python
for i, r in enumerate(regions):
    if r not in region_map:
        region_map[r] = len(all_brain_regions)
        all_brain_regions.append(r)
    neuron_region_idx[i] = region_map[r]
```

iii. The AI documented only the one optimisation it made (CONVERSION_NOTES Step 6, "~60x speedup"). The remaining loops and the absence of parallelism were never identified; trajectory steps 82–89 show the AI simply waiting out the serial 33-minute run.

## 10-c. What processing does the code repeat multiple times?

i. Three real repetitions. (1) Spikes are binned for **every** trial in the session and only afterwards masked down to the valid ones — on a typical session ~400 of ~570 trials survive, so roughly 25–30 % of the binning work, and the peak `(n_trials, n_clusters, 100)` allocation, is thrown away. (2) The identical 100-element `time_since_stim` vector is re-`vstack`ed and stored independently for each of the 149,339 trials, and the per-trial constants choice/prior are materialised 100 times each. (3) The whole 84.7 GB pickle was written, then read back, re-cast and rewritten by an ad-hoc script outside `convert_data.py` — so the delivered `converted_data.pkl` (`uint8` neural, `int8` output) cannot be reproduced by running the shipped script, which emits `float32`/`int64`. Minor: `find_file` re-globs the session directory once per stream, and the sample-mode selector re-globs every candidate session before processing.

ii.
```python
# 7. Bin spikes per trial (for ALL trials first, then filter)
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
...
valid_idx = np.where(combined_mask)[0]
neural_data = binned_spikes[valid_idx]
```
```python
for i in range(n_valid_trials):
    inp = np.vstack([
        time_since_stim[np.newaxis, :],
        np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
    ])
    input_list.append(inp)
```
```python
neural_list = [neural_data[i].astype(np.float32) for i in range(n_valid_trials)]
```
(the delivered file is `uint8`, produced afterwards by a separate script: `data['neural'][i][j].astype(np.uint8)` … `pickle.dump(data, f)`)

iii. Not documented. The in-code comment "(for ALL trials first, then filter)" shows the choice was deliberate — binning before masking keeps trial indices aligned with the trials table — but no cost/benefit was recorded. The dtype re-pass is described in the trajectory (step 118) only as a file-size win: "The file size is now 20.7 GB (down from 84.7 GB) ... a 4x reduction by using uint8 instead of float32."

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item is carrying the entire unfiltered spike sorting: ~1,450 units per session including 63,769 `root` and 11,898 `void` units (units the histology placed outside the brain), which no downstream analysis can use, inflating the delivered file to 22 GB and diluting the decoder's input. Next, the ~25–30 % of trials that are binned and then dropped by the quality mask. Then the storage redundancy: 100-fold replication of the per-trial scalars (choice, prior, trial-number-in-block) and 149,339 copies of one constant time vector. Smaller items: `clusters.channels`/`brainLocationIds` are loaded and Beryl-mapped for every cluster including those in `void`; the whole trials table is read although only nine columns are used; `bincount2D` is imported and never called; `--show-processing` renders per-session figures that no downstream step consumes; and the neural array is built as `float32` only to be re-cast to `uint8` afterwards.

ii.
```python
from iblutil.numerical import bincount2D      # imported, never used
```
```python
n_clusters = len(cluster_regions)   # every sorted cluster, incl. root and void
binned_all = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
```
```python
out = np.vstack([
    np.full((1, N_BINS), choice_mapped[i], dtype=np.int64),   # one value, 100 copies
    np.full((1, N_BINS), prior_mapped[i], dtype=np.int64),    # one value, 100 copies
    wheel_disc[i:i+1, :],
    me_disc[i:i+1, :],
])
```

iii. The per-trial broadcasting of scalars is required by the target format ("If at all possible, make it time-varying"), so that redundancy is intentional. Keeping every cluster follows the AI's Step 1 reading that the reference caching script passes `qc=None`; the retention of `void` units was never discussed. The other items were not identified — CONVERSION_NOTES Steps 5 through 12 were left at "NOT STARTED" in the delivered file, so the critical-review steps that would have surfaced them are missing from the documentation even though the AI drafted some of that text in the trajectory.
