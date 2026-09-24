# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API at all. It takes the session list from the reference code's release table, `/app/code/code_zhang2025/data/bwm_release.csv` (459 unique `eid`s, 699 probes), and then reads every file straight off the local ONE cache by re-implementing ONE's path resolution itself: a session directory is built as `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf`, and a hand-written `find_latest_revision()` walks that directory for `#YYYY-MM-DD#` revision folders and picks the lexicographically newest one. Trials come from `_ibl_trials.table.pqt` (`pd.read_parquet`), spikes from `<probe>/pykilosort/spikes.{times,clusters}.npy`, wheel from `_ibl_wheel.{position,timestamps}.npy`, whisker from `<side>Camera.ROIMotionEnergy.npy` + `_ibl_<side>Camera.times.npy` (all `np.load`). Sessions are processed one at a time in a serial `for` loop (`ProcessPoolExecutor` is imported but never used).

Consequence: the session-number component of the path is hard-coded to `001` (`find_session_path(..., number=1)` → `f'{number:03d}'`). The cache actually contains 36 sessions under `002`, 9 under `003`, 3 under `004`, and one each under `007`/`008` — exactly the 50 sessions the log reports as `path not found, skipping`. A further 4 sessions died on a missing wheel file (`expected str, bytes or os.PathLike object, not NoneType`) and 17 on `<2 trials with all data`, leaving **392 of 459 sessions** converted (the human reference converts 441–444).

ii.
```python
DATA_DIR = Path('/app/data/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def find_session_path(lab, subject, date, number=1):
    """Find the session directory in the ONE cache."""
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None

def find_latest_revision(base_path, filename_pattern):
    """Find the latest revision of a file, checking revision directories first."""
    candidates = []
    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith('#') and item.name.endswith('#'):
                ...
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]
```
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)          # groupby('eid').first()
...
for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. From CONVERSION_NOTES.md Step 6: *"Direct file loading (no ONE API needed, works offline from cache)"*, and Step 5 decision 7: *"Session list: Use BWM release CSV (459 sessions) as ground truth"*. The AI justified the choice by the release CSV being what the Zhang 2025 reference code itself iterates over, and by wanting a loader that does not need a database connection. It never noticed or investigated the 50 `path not found` skips; Step 9 simply records *"392 (67 skipped) | 85% - OK, missing files"*.

## 1-b. How are the data split into subjects?

i. The subject name is read from the `subject` column of the BWM release CSV, carried on each session's result dict, and at assembly time the subjects are the sorted unique names over the successfully converted sessions, with `subject_idx` giving each session's index into that list. No path or filename parsing is used for identity. Result: 129 subjects over 392 sessions (data paper: 139 mice; human reference: 136).

ii.
```python
    return {..., 'subject': subject, ...}

all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
    subject_idx.append(subject_to_idx[r['subject']])
...
    'subjects': all_subjects,
    'subject_idx': np.array(subject_idx),
```

iii. No explicit justification is given beyond the mapping table in Step 5; the subject id is already a column of the release table, so nothing has to be derived. Step 9's consistency table accepts 129 vs 139 as *"93% - OK"*.

## 1-c. How are the data split into sessions?

i. A session is one `eid`. The release CSV has one row per probe insertion, so it is collapsed with `groupby('eid').first()` to one row per session (459). Each surviving session becomes one element of `data['neural']` / `data['input']` / `data['output']`, and the two probes of a dual-probe session are merged into that single session rather than kept separate.

ii.
```python
def get_session_list(bwm_df):
    """Get unique sessions from BWM release CSV."""
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```
```python
probe_dirs = sorted([d.name for d in alf_path.iterdir()
                     if d.is_dir() and d.name.startswith('probe')])
...
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. CONVERSION_NOTES Step 4 resolves the 459 (data paper / BWM CSV) vs 461 (directories on disk) vs 433 (methods paper) discrepancy with *"Use BWM CSV as ground truth for session list"*. Merging probes within a session follows the reference code's `merge_probes()` (Step 1 table).

## 1-d. How are the data split into trials?

i. No splitting is performed: the trials table has one row per trial, and that row set (after masking) defines the trials. Each trial becomes one `(n_neurons, 100)` neural array, one `(2, 100)` input array and one `(4, 100)` output array, built from a 2 s window around that row's `stimOn_times`.

ii.
```python
trials = pd.read_parquet(trials_file)
...
valid_trials = trials[mask].reset_index(drop=True)
align_times = valid_trials[ALIGN_TIME].values          # 'stimOn_times'
interval_starts = align_times + TIME_WINDOW[0]
interval_ends   = align_times + TIME_WINDOW[1]
```

iii. Implicit — the Step 5 mapping table treats the trials table as the trial index. The window `(-0.5, 1.5)` around `stimOn_times` is taken from the reference code's caching parameters (Step 1: *"binsize=0.02, align_time='stimOn_times', time_window=(-0.5, 1.5)"*).

## 1-e. How are trials filtered based on quality controls?

i. Two masks applied in series.

`create_trial_mask()` reproduces the reference code's `load_trials_and_mask(..., max_trial_len=10.0)`:
 - drop trials with NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType` (this is the reference's verbatim `nan_exclude='default'` list);
 - reaction time `firstMovement_times - stimOn_times` must lie in [0.08 s, 2.0 s];
 - no-response trials (`choice == 0`) dropped;
 - trial length `feedback_times - goCue_times` must be ≤ 10 s.

A second `combined_mask` then drops any trial whose 2 s window is not fully covered by the wheel **and** the camera. Coverage is enforced inside `interpolate_behavior_to_bins()` with the reference code's own edge test (first/last sample within one bin of the window edge); a trial that fails leaves an all-NaN row, and those rows are removed. Finally a session with fewer than 2 surviving trials is dropped entirely. Result: 167,487 trials, mean 427/session (human reference: 188,740).

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    mask &= (trials['choice'] != 0)
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    return mask
```
```python
        if np.abs(t_start - t_sel[0]) > binsize:
            continue
        if np.abs(t_end - t_sel[-1]) > binsize:
            continue
```
```python
combined_mask = np.ones(len(valid_trials), dtype=bool)
if wheel_speed_binned is not None:
    combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
else:
    combined_mask[:] = False
if me_binned is not None:
    combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
else:
    combined_mask[:] = False
```

iii. CONVERSION_NOTES Step 3: *"Trial curation rules (both papers & code): Exclude NaN in [6 events]; Exclude RT < 0.08 s or > 2.0 s; Exclude no-choice trials; Exclude trials with max_trial_len > 10.0 s"*, and Step 6: *"Trial mask matching reference code exactly"*. The AI read `prepare_data()` in `ibl_data_utils.py`, which calls `load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)`, and copied those defaults. The coverage checks are copied from `get_behavior_per_interval()`'s `'target data starts too late'` / `'ends too early'` guards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's `pykilosort` folder (latest revision). `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` supply the anatomical label for each unit (not the activity itself). `clusters.metrics.pqt` and `clusters.depths.npy` are also read but never used.

ii.
```python
    spikes = {
        'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
        'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
    }
    clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
    chan_brain_ids = np.load(rev_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```
```python
def get_brain_regions(clusters, br):
    chan_acronyms = br.id2acronym(clusters['chan_brain_ids'])
    chan_beryl = br.acronym2acronym(chan_acronyms, mapping='Beryl')
    cluster_regions = chan_beryl[clusters['channels']]
    return cluster_regions
```

iii. Step 1/Step 5 map `spikes.times + spikes.clusters → neural`, mirroring the reference's `load_spiking_data()` / `bin_spiking_data()`. Step 5 decision 4: *"Brain region mapping: Use Beryl atlas mapping (iblatlas BrainRegions)"*, matching the reference's `list_brain_regions()`.

## 2-b. How is the `neural` data processed?

i. Per session, the probes are merged into one population: cluster ids of probe *n* are offset by the cumulative cluster count, channel indices by the cumulative channel count, and the concatenated spike times are stable-sorted. Spikes are then counted into 100 × 20 ms bins spanning [stimOn − 0.5 s, stimOn + 1.5 s) using `searchsorted` to slice the trial window plus a flat `np.add.at` accumulation. **The counts are kept as raw counts, not converted to a firing rate**, and are stored as `uint8` (the human reference stores Hz as `float32`). No smoothing, no z-scoring, no baseline subtraction.

ii.
```python
def merge_probes(spikes_list, clusters_list):
    ...
        merged_spikes_clusters.append(spikes['clusters'] + cluster_offset)
        merged_channels.append(clusters['channels'] + chan_offset)
        cluster_offset += n_clusters
        chan_offset += len(clusters['chan_brain_ids'])
    sort_idx = np.argsort(all_times, kind='stable')
```
```python
def bin_spikes_vectorized(spike_times, spike_clusters, n_clusters,
                          interval_starts, interval_ends, binsize, n_bins):
    binned = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    for trial_idx in range(n_trials):
        idx_beg = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
        flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
    return binned
```
```python
    for t in range(len(final_spikes)):
        neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. Step 1 records the reference chain `bin_spiking_data → get_spike_data_per_interval` producing `(n_trials, n_clusters, n_bins)` spike counts, and Step 5 maps `spikes.times + spikes.clusters → neural, "Bin into 20 ms windows aligned to stimOn, shape (n_neurons, 100)"`. The `uint8` cast is justified in Step 9/Step 10 purely on memory grounds: *"neural stored as uint8 to fit in 64 GB container memory"*, with the check *"Max spike count 68 (fits uint8)"*.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every sorted cluster of every probe is kept: `n_clusters = len(clusters['channels'])`. `clusters.metrics.pqt` — which holds the IBL `label` column (0, 1/3, 2/3, 1) used to define a "well-isolated" unit — is loaded and merged across probes but never consulted. Units whose Beryl acronym is `void` (histology places them outside the brain) are also kept, as are `root`, `x` and `y`. The result is 534,911 neurons over 392 sessions (mean 1,365/session, max 3,140), against the data paper's 75,708 well-isolated units and the human reference's ~72,400 (mean 164/session). The verification output shows **12,004 `void` neurons and 75,867 `root` neurons** in the delivered dataset.

ii.
```python
    spikes, clusters = merge_probes(spikes_list, clusters_list)
    n_clusters = len(clusters['channels'])      # every sorted cluster
    cluster_regions = get_brain_regions(clusters, br)
```
```python
    metrics_file = rev_dir / 'clusters.metrics.pqt'
    if metrics_file.exists():
        metrics = pd.read_parquet(metrics_file)   # loaded, merged, never used
```

iii. This is the AI's most consequential deliberate choice and it is documented repeatedly. Step 1: *"qc=None: ALL clusters used (not filtered by quality label)"*. Step 3 notes the paper's curation (*"amplitude > 50 µV, noise cutoff < 20 µV, refractory period violation pass ... captured in clusters.metrics.pqt 'label' column (label >= 1 = good)"*) but adds *"**Reference code uses qc=None: ALL clusters, not just good ones**"*. Step 4 records the conflict explicitly — *"Code says qc=None (all units) | Data shows label>=1 for good | Papers say 75,708 well-isolated"* — and resolves it *"**Follow reference code: use all units (qc=None)**"*. Step 5 decision 1 repeats it. The AI verified in the trajectory that `prepare_data()` calls `load_spiking_data(one, pid, eid=eid, pname=probe_name)` with `qc` left at its `None` default, which is correct about the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. All IBL streams are already on one synchronised session clock, so alignment is a subtraction: the trial window is `[stimOn − 0.5, stimOn + 1.5]`, spikes inside it are located with `searchsorted`, and each spike's bin is `floor((t − t_start)/0.02)` clipped to the last bin, so bin 0 begins exactly at stimOn − 0.5 s. `off_start = -0.5` and `off_end = 1.5` are recorded in metadata together with `temporal_alignment_event: 'stimulus onset (stimOn_times)'`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends   = align_times + TIME_WINDOW[1]
```
```python
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
```
```python
        'temporal_alignment_event': 'stimulus onset (stimOn_times)',
        'off_start': TIME_WINDOW[0],
        'off_end': TIME_WINDOW[1],
```

iii. Step 4 flags that the methods paper aligns wheel speed and whisker motion energy to `firstMovement_times` rather than `stimOn_times`, and resolves it in favour of the task specification: *"**Task spec says stimOn for all; follow task spec**"* and *"ALL alignment to stimulus onset (not firstMovement for wheel/whisker)"*. The window itself is the reference code's `(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial over the 2 s window; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once at that resolution directly from spike times — there is no rebinning, resampling or interpolation of the neural data. The behavioural traces are sampled once onto the same 100-point grid. All sessions and trials have exactly T = 100 (confirmed in verification: `T: mean 100.00, min 100, max 100`).

ii.
```python
BINSIZE = 0.02  # 20 ms time bins
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```
```python
        'time_bin_size': BINSIZE * 1000,  # in ms
        'n_timepoints': N_BINS,
```

iii. Step 4: *"Bin size | 0.02 s (20 ms) [code] | 20 ms (dynamic), 50 ms (choice/prior) [papers] | **Follow reference code: 20 ms for all**"*. Step 3 records the methods paper's *"split into 2-s trials ... 20-ms bins ... T = 100"*.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not derived from a raw variable at all — it is the decoding grid itself, defined by `stimOn_times` (the alignment event) plus the constants `TIME_WINDOW = (-0.5, 1.5)` and `BINSIZE = 0.02`. One 100-element vector is computed once and reused for every trial of every session.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```
```python
    'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
```

iii. Step 5 mapping table: *"Time since stimOn → input[0] → np.linspace(-0.5, 1.48, 100), continuous time-varying | Manual construction | Same for all trials"*. (The note's stated endpoints are transposed relative to the code, which actually produces −0.48 … 1.50.)

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid. The AI uses the **right edge** of each 20 ms bin (−0.48, −0.46, …, 1.48, 1.50) rather than the bin centre (the human reference uses centres, −0.49 … 1.49), i.e. a constant 10 ms — half a bin — offset relative to the human solution. The choice of grid is not arbitrary: it is exactly the grid the reference code interpolates behaviour onto, `np.linspace(interval_beg + binsize, interval_end, n_bins)`, so the input axis and the behavioural outputs are built on one identical convention.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)

    for t in range(len(final_trials)):
        inp = np.array([
            time_since_stim,                                             # time-varying
            np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),  # per-trial
        ], dtype=np.float32)  # (2, n_bins)
        input_trials.append(inp)
```

iii. Step 3 records the reference's interpolation grid: *"Interpolates to uniform sampling: `np.linspace(interval_beg + binsize, interval_end, n_bins)`"*, and Step 10's self-check confirms *"Input shapes correct: (2, 100), time axis from -0.48 to 1.5"*.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: both are indexed on the same 100-bin grid anchored at the same `stimOn_times`. Neural bin *i* counts spikes in `[stimOn − 0.5 + 0.02i, stimOn − 0.5 + 0.02(i+1))`; the input value at index *i* is `−0.5 + 0.02(i+1)`, the closing edge of that same bin. There is therefore no lag between the two streams beyond the ≤1-bin edge convention, and the same grid is reused for the wheel and whisker outputs.

ii.
```python
interval_starts = align_times + TIME_WINDOW[0]
interval_ends   = align_times + TIME_WINDOW[1]
...
        bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
```
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The `--show-processing` plots were produced to check exactly this; Step 7 records *"Processing plots saved, show proper neural activity, wheel/ME alignment"*, and the plotting code uses `time_axis` as the shared x-axis for the neural raster, the wheel trace and the whisker trace.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. The trials table carries no block id, so block boundaries are recovered as the positions where `probabilityLeft` changes from the previous row.

ii.
```python
    prob_left_all = trials['probabilityLeft'].values
    trial_in_block_all = compute_trial_in_block(prob_left_all)
```
```python
def compute_trial_in_block(prob_left):
    """Compute trial number within each block.
    Block boundaries are where probabilityLeft changes."""
```

iii. Step 5 mapping table: *"Trial number in block → input[1] → Count trials within each block, per-trial scalar | Computed from probabilityLeft changes | Resets at block boundary"*, and Step 5 decision 6: *"Trial number in block: Computed by detecting block boundaries from changes in probabilityLeft"*.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter starting at **1** that resets whenever `probabilityLeft` differs from the previous trial, implemented as a plain Python loop. Crucially the counter is run over the **complete, unfiltered** trials table and only afterwards indexed by the trial mask and the coverage mask, so a trial that is later discarded still advances the count and the number reported is the animal's true position in the block. The per-trial scalar is then broadcast across all 100 bins so the input array is `(2, 100)`. Observed range in the full dataset: 1 … 99 (human reference: 0 … 98, i.e. the same quantity 0-indexed).

ii.
```python
def compute_trial_in_block(prob_left):
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block
```
```python
    trial_in_block_all = compute_trial_in_block(prob_left_all)   # over ALL trials
    trial_in_block_valid = trial_in_block_all[mask.values]
    trial_in_block_final = trial_in_block_valid[combined_mask]
```
```python
            np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
```

iii. Step 3 records the block structure from the data paper (*"Block length 20-100 trials (mean ~51)"*, *"Unbiased block: First 90 trials"*), which is what makes counting on the unfiltered table the meaningful definition. The AI did not comment on 1- vs 0-indexing.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which is +1 / −1 / 0. Trials with `choice == 0` (no response) are already removed by the trial mask. The remaining ±1 is recoded to {0, 1} with `(choice + 1) // 2`, **under the stated assumption that −1 means left and +1 means right**, and `output_values[0] = ['left', 'right']` labels index 0 as "left".

That assumption is backwards. Checking the raw trials tables directly: over 20 sessions, every correct trial with a left-side stimulus (`contrastLeft > 0`, `feedbackType == 1`) has `choice == +1` (4,691 of 4,691) and every correct right-stimulus trial has `choice == −1` (4,336 of 4,336). So **+1 is a leftward choice and −1 is rightward**, as the human reference states (`CHOICE = {1.0: 0, -1.0: 1}`). The AI's labels are inverted: what it stores as "left" is actually the animal's rightward choice. The class fractions confirm the mirror — AI: left 0.494 / right 0.506; reference: left 0.5075 / right 0.4925.

ii.
```python
    # Output 0: choice (binary, per-trial) - left=0, right=1
    choice = final_trials['choice'].values.copy()
    # IBL: -1=left, 1=right -> convert to 0=left, 1=right
    choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
```
```python
        'output_values': [
            ['left', 'right'],                  # choice: 0=left, 1=right
```

iii. Step 4 states the decision as *"Choice: left=0, right=1 (reference code: left=-1, right=1)"* and Step 5's mapping table says *"trials.choice → output[0] → left(-1)->0, right(1)->1"*. No justification or sanity check is offered for the sign convention anywhere in CONVERSION_NOTES or the trajectory — the comment `# IBL: -1=left, 1=right` is simply asserted, and the planned sanity check *"Choice distribution approximately balanced (~50/50)"* cannot detect a sign flip.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding described in 5-a: `(choice + 1) // 2`, then the per-trial scalar is broadcast over all 100 time bins so choice is constant within a trial. No other transformation. Because the mapping is inverted, every stored choice label is the opposite of the animal's actual side.

ii.
```python
    choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
...
        out = np.array([
            np.full(N_BINS, choice_encoded[t], dtype=int),   # per-trial, broadcast
            ...
        ], dtype=int)
```

iii. Step 5: broadcast per-trial variables across time because the format spec says *"If at all possible, make it time-varying"*. Step 10's check confirms *"Choice and prior constant within trials (per-trial variables)"*.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes exactly the three block values 0.2, 0.5 and 0.8, recoded to 0, 1, 2 as the Decoder Task section prescribes.

ii.
```python
    prob_left = final_trials['probabilityLeft'].values
    prior_encoded = np.zeros(len(prob_left), dtype=int)
    prior_encoded[prob_left == 0.2] = 0
    prior_encoded[prob_left == 0.5] = 1
    prior_encoded[prob_left == 0.8] = 2
```
```python
            ['0.2', '0.5', '0.8'],              # prior: 0=0.2, 1=0.5, 2=0.8
```

iii. Step 4: *"Prior encoding | block (probabilityLeft) | 0.2, 0.5, 0.8 | Prior = running estimate [paper] | Task spec: 0.2->0, 0.5->1, 0.8->2 (categorical)"* — the AI noted that the methods paper decodes a continuous running prior estimate and deliberately used the discrete block probability instead because the task specification names the three values explicitly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the three-way recoding, followed by broadcasting the per-trial value across the 100 bins. NaN `probabilityLeft` has already been removed by the trial mask. Note the array is initialised with `np.zeros`, so any value that is not exactly 0.2/0.5/0.8 would silently be stored as class 0 rather than raising — in practice IBL only emits those three values, and the resulting distribution (0.420 / 0.141 / 0.439) matches the human reference (0.418 / 0.141 / 0.442) almost exactly.

ii.
```python
    prior_encoded = np.zeros(len(prob_left), dtype=int)
    prior_encoded[prob_left == 0.2] = 0
    prior_encoded[prob_left == 0.5] = 1
    prior_encoded[prob_left == 0.8] = 2
...
            np.full(N_BINS, prior_encoded[t], dtype=int),  # per-trial, broadcast
```

iii. Step 5 planned sanity check: *"Prior distribution: 3 classes roughly proportional to block structure"*; Step 9 reports *"Prior dist | ~42/14/44 | 42.0/14.1/43.9 | Excellent"*. The ~14% at 0.5 is the unbiased block at the start of each session, as expected from the data paper.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From the raw rotary-encoder data, `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. Speed is the absolute value of the velocity derived from them (radians/s).

ii.
```python
        wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
        wheel_ts_raw  = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
        wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```
```python
    return t, np.abs(vel).astype(np.float32)  # wheel speed = |velocity|
```

iii. Step 1 lists the reference's `load_target_behavior()` as the loader for *"wheel, motion energy, pupil signals"* and Step 3 records that the reference's behaviours are *"choice, reward, block, wheel-speed, whisker-motion-energy"*; the reference likewise derives `'wheel-speed'` as `np.abs` of velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI re-implements `ibllib`'s wheel pipeline by hand rather than calling `SessionLoader.load_wheel()` (it never imports ONE):
 1. the irregularly sampled position is linearly interpolated onto an even 1000 Hz grid (`interpolate_position`);
 2. velocity is the 8th-order Butterworth low-pass (20 Hz corner) filtered position differentiated and scaled by `fs` (`velocity_filtered`);
 3. speed = `|velocity|`;
 4. the session-long speed trace is linearly interpolated onto each trial's 100-point grid, with `fill_value='extrapolate'` and the reference's coverage guards.

The re-implementation is numerically faithful to `ibllib`: `Wn = corner_freq / fs * 2` equals `ibllib`'s `corner_frequency / (fs/2)` = 0.04, and `np.insert(np.diff(sosfiltfilt(sos, pos)), 0, 0) * fs` is `velocity_filtered` verbatim, with `ibllib`'s default `order=8, corner_frequency=20`.

ii.
```python
def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ,
                      order=WHEEL_FILTER_ORDER):
    """Interpolate wheel position and compute velocity (matching ibllib)."""
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```
```python
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
        y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. Step 6: *"Wheel velocity via Butterworth filter matching ibllib (1 kHz interp, 20 Hz corner, order 8)"*. The interpolation onto the trial grid is copied from the reference's `get_behavior_per_interval()` (Step 3: *"Uses scipy.interpolate.interp1d() with kind='linear', fill_value='extrapolate'"*).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three equal-frequency (quantile) classes cut at the 33.3rd and 66.7th percentile **of that session's own pooled speed values**, i.e. the percentiles are computed over the flattened `(n_trials × 100)` matrix of the session, not per trial and not globally. Classes are labelled `['low', 'medium', 'high']`. The delivered dataset is 0.333 / 0.333 / 0.333, identical to the human reference.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    """Discretize continuous values into n_bins equal-frequency bins using quantiles.
    Computed across all valid (non-NaN) values."""
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(int), boundaries
```
```python
    all_wheel_flat = final_wheel.flatten()
    wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
    wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

iii. Step 5 decision 5: *"Discretization of wheel speed/whisker ME: Use equal-frequency (quantile) bins across all trials within a session, 3 bins (low/medium/high)"*. The Decoder Task only says "discretized into 3 bins"; equal-frequency was chosen so that no class is degenerate, and the `--show-processing` plots overlay the raw trace on the discretized one to verify the cut. Step 9 checks *"Wheel speed dist: 33.3/33.3/33.3% (quantile-based)"*.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The session-long speed trace is evaluated at `stimOn + linspace(-0.48, 1.50, 100)` — the same 100 points as the `time_since_stimulus_onset` input and the closing edges of the same neural bins — so bin index *i* of the wheel output and bin index *i* of the neural matrix describe the same 20 ms of the same trial. Trials where the wheel does not span the window (first/last sample more than one bin from the edge) are dropped rather than extrapolated.

ii.
```python
        wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
            wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
        )
```
```python
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```
```python
        if np.abs(t_start - t_sel[0]) > binsize:
            continue
        if np.abs(t_end - t_sel[-1]) > binsize:
            continue
```

iii. All IBL streams share one session clock, so evaluating the trace at the bin grid is all the alignment needed. The `--show-processing` figures plot the wheel trace against the same `time_axis` used for the neural raster (Step 7: *"Processing plots ... show proper neural activity, wheel/ME alignment"*).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the released side-camera ROI motion energy, `leftCamera.ROIMotionEnergy.npy` with frame times `_ibl_leftCamera.times.npy`, falling back to the right camera when the left is absent. One value per video frame, used exactly as released.

ii.
```python
def load_motion_energy(alf_path, side='left'):
    if side == 'left':
        me_file    = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
    else:
        me_file    = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
    ...
    min_len = min(len(me), len(times))
    return times[:min_len], me[:min_len].astype(np.float32)
```
```python
        me_times, me_values = load_motion_energy(alf_path, side='left')
        if me_times is None:
            me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. Step 6: *"Motion energy loading from leftCamera (fallback to rightCamera)"*, following the reference's `load_anytime_behaviors()`, which the AI's Step 1 notes loads *"whisker motion energy (left/right)"*.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the trace itself — no filtering, smoothing or normalisation. The only handling is truncating `values` and `times` to their common length when the released arrays disagree, then linearly interpolating onto each trial's 100-point grid with the same coverage guards used for the wheel, and finally the quantile split.

ii.
```python
    min_len = min(len(me), len(times))
    return times[:min_len], me[:min_len].astype(np.float32)
```
```python
            me_binned, me_good = interpolate_behavior_to_bins(
                me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
            )
```

iii. Step 5 mapping table: *"whisker motion energy → output[3] → Discretize into 3 equal-frequency bins, time-varying | load_target_behavior, get_behavior_per_interval"*. The length-truncation is justified in the code comment as *"Handle length mismatch (common in IBL data)"*.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: three equal-frequency classes cut at the 33.3rd/66.7th percentile of that session's own pooled motion-energy values, labelled `['low', 'medium', 'high']`. Delivered distribution 0.333 / 0.332 / 0.335 (human reference 0.333 / 0.333 / 0.335).

ii.
```python
    all_me_flat = final_me.flatten()
    me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
    me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. Same as 7-c — Step 5 decision 5, applied to both continuous outputs so that the three classes are balanced within each session; Step 10 check 7: *"Whisker ME dist: 33.3/33.2/33.5% (quantile-based)"*.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is evaluated at the same `stimOn + linspace(-0.48, 1.50, 100)` grid as the neural bins, the time input and the wheel output, using the same `interpolate_behavior_to_bins()` function. Camera frame times are on the session clock, so no resynchronisation is needed; trials whose window the camera does not span are dropped.

ii.
```python
            me_binned, me_good = interpolate_behavior_to_bins(
                me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
            )
```
```python
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
```

iii. Step 4/Step 5: everything is aligned to `stimOn_times` per the task specification, overriding the methods paper's `firstMovement_times` alignment for whisker/wheel. The processing plots overlay raw and discretized whisker traces on the shared time axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data is handled by dropping the affected trial, probe or session, wrapped in broad `try/except Exception` blocks:
 - missing session directory → session skipped (`path not found`);
 - a probe whose spike files cannot be read → that probe skipped, the session continues with the remaining probes; no probe readable → session skipped;
 - unreadable trials table → session skipped;
 - missing/unreadable wheel or motion energy → an exception is caught, the corresponding binned array stays `None`, `combined_mask` is forced all-`False`, and the session is dropped at the `<2 trials` check;
 - camera `ROIMotionEnergy` / `times` length mismatch → both truncated to the shorter length;
 - trials whose wheel or camera does not cover the window, or whose interpolation raises → left as NaN and removed;
 - NaN in any of the six key trial events → trial removed;
 - sessions with fewer than 2 usable trials → dropped (the format spec requires ≥2 trials per session).

Net effect on the full run: 50 sessions `path not found`, 4 sessions lost to a missing wheel file (`expected str, bytes or os.PathLike object, not NoneType` — `find_latest_revision` returned `None` and was passed to `np.load`), 13 more with `<2 trials`, for 392 of 459 sessions delivered. The AI did not diagnose the 50 `path not found` cases; they are all sessions stored under a directory number other than `001` (36 under `002`, 9 under `003`, 3 under `004`, 2 under `007`/`008`), which the hard-coded path template cannot reach. A module-level `warnings.filterwarnings('ignore')` suppresses all warnings for the whole run.

ii.
```python
warnings.filterwarnings('ignore')
```
```python
    try:
        trials = load_trials(alf_path)
    except Exception as e:
        print(f"  Session {eid}: error loading trials: {e}")
        return None
```
```python
        except Exception as e:
            print(f"  Session {eid}, {probe_name}: error loading spikes: {e}")
            continue
```
```python
    except Exception as e:
        print(f"  Session {eid}: error loading wheel: {e}")
...
    if wheel_speed_binned is not None:
        combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
    else:
        combined_mask[:] = False
```
```python
    if n_valid < 2:
        print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
        return None
```

iii. Step 9/10: *"67 sessions skipped: missing data paths, missing wheel, or <2 valid trials"*, and the Step 9 consistency table rates 392/459 as *"85% - OK, missing files"* and 129/139 subjects as *"93% - OK"* — i.e. the AI explicitly decided the shortfall was attributable to the dataset rather than to its own path construction, and did not investigate further.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented timing only at session granularity: `process_session` records wall time per session and `main` prints a running ETA every 10 sessions. It reported ~20 s/session on the two sample sessions and 1,741 s (29 min) for the full 459-session run, and concluded no optimisation was needed. It never profiled *within* a session, so it never named a bottleneck. In fact the dominant costs are (a) `np.load` of `spikes.times.npy` + `spikes.clusters.npy`, which for a 2-probe session is hundreds of MB of file I/O and is the same bottleneck the human reference identifies; (b) the per-trial binning loop, which uses `np.add.at` — an unbuffered scatter-add that is substantially slower than the `np.bincount` the human reference uses; (c) the 1 kHz interpolation plus 8th-order `sosfiltfilt` over the entire session's wheel trace; and (d) pickling 23 GB to disk at the end.

Also relevant: in Step 7 the AI estimated the full conversion at *"~9200 s (~2.5 hours)"* and wrote *"This is acceptable"*, even though the instructions direct that an estimate over 15 minutes be optimised; the actual 29 min still exceeded that threshold and no optimisation was attempted.

ii.
```python
def process_session(session_info, br, show_processing=False):
    t0 = time.time()
    ...
    elapsed = time.time() - t0
    print(f"  Session {eid} ({subject}/{date}): {n_valid} trials, {n_clusters} neurons, {elapsed:.1f}s")
```
```python
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (idx + 1)
            remaining = rate * (len(sessions) - idx - 1)
            print(f"\n  Progress: {idx+1}/{len(sessions)} sessions, "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
```

iii. CONVERSION_NOTES Step 7: *"Total | ~20 s / Session | ~2.5 hours ... Note: ~20 s/session, full run ~9200 s (~2.5 hours). This is acceptable."* Step 9 reports the realised *"Total: 1741 s (29 min) for 459 sessions"*. No per-step breakdown is documented anywhere.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain, and the whole conversion runs serially:
 - `bin_spikes_vectorized()` — despite the name, a Python `for` loop over trials whose inner accumulation is `np.add.at`, the slowest scatter-add in NumPy; the human reference does the same slicing but fills each trial with a single `np.bincount` on a flat index, and the whole thing could be done in one `bincount` over all trials by offsetting each spike's index by its trial.
 - `interpolate_behavior_to_bins()` — a loop over trials building a fresh `interp1d` object per trial; one `np.interp` over a single concatenated query vector would do.
 - `compute_trial_in_block()` — a scalar Python loop over every trial of the session; the human reference does it in one line with `(probabilityLeft != probabilityLeft.shift()).cumsum()` + `groupby(...).cumcount()`.
 - the three list-building loops at the end of `process_session` that call `np.full(N_BINS, ...)` per trial.
 - `main()` processes sessions one at a time; `ProcessPoolExecutor, as_completed` is imported at the top of the file and never used, so 10-way parallelism (what the human reference uses) was left on the table.

ii.
```python
from concurrent.futures import ProcessPoolExecutor, as_completed   # never used
```
```python
def bin_spikes_vectorized(...):
    for trial_idx in range(n_trials):
        ...
        np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```
```python
def compute_trial_in_block(prob_left):
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
```
```python
    for idx in range(len(sessions)):
        sess = sessions.iloc[idx]
        result = process_session(sess, br, show_processing=args.show_processing)
```

iii. Step 6 claims *"Spike binning via vectorized numpy operations"* and the function is named `bin_spikes_vectorized`, but neither the notes nor the trajectory identify any loop as a candidate for vectorisation or discuss parallelism. The implicit justification is Step 7's judgement that ~20 s/session was *"acceptable"*.

## 10-c. What processing does the code repeat multiple times?

i. Mostly minor, but there is some:
 - `find_latest_revision()` re-walks the session's `alf` directory (and every `#date#` subdirectory in it) from scratch on each call — once for the trials table, twice for the wheel, twice for each camera view tried, i.e. 5–7 full directory scans per session where one listing would serve.
 - The identical 100-element `time_since_stim` vector is materialised into every one of the 167,487 trials' input arrays rather than shared.
 - `np.full(N_BINS, ...)` is re-allocated per trial for `trial_in_block`, `choice` and `prior`, which are constant within a trial.
 - `BrainRegions()` is instantiated once in `main()` and reused — correctly not repeated.

None of this is on the critical path; the repeated directory scans are the only one with measurable cost, and it is small next to the spike file I/O.

ii.
```python
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
...
        wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
        wheel_ts_raw  = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
...
        me_file    = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
        times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
```
```python
    for t in range(len(final_trials)):
        inp = np.array([
            time_since_stim,
            np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
        ], dtype=np.float32)
```

iii. Not discussed in CONVERSION_NOTES or the trajectory — the AI made no decision about repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things, one of them significant:
 - **`clusters.metrics.pqt` is read per probe, deep-copied, `cluster_id`-offset, and `pd.concat`-merged across probes — and then never read again.** This is the file that carries the `label` column the quality filter would use, so the code pays the I/O cost of the QC information and then discards it (see 2-c).
 - `clusters.depths.npy` is likewise loaded, concatenated across probes, and never used.
 - `interpolate_behavior_to_bins()` computes and returns a `good_mask`; both `wheel_good` and `me_good` are assigned and never used, because the caller re-derives the same information with `~np.any(np.isnan(...), axis=1)`.
 - Most consequentially, because no QC filter is applied, roughly 460,000 of the 534,911 stored units are ones the data paper's curation would exclude, plus 12,004 units localised outside the brain (`void`). Storing them inflated the output to 23 GB, which then did not fit in memory: the AI had to build a separate `converted_data_subset.pkl` and train the decoder on 98 of 392 sessions. So the bulk of the binning work and nearly all of the file size went into data that the downstream analysis could not even use.

ii.
```python
    metrics_file = rev_dir / 'clusters.metrics.pqt'
    if metrics_file.exists():
        metrics = pd.read_parquet(metrics_file)
...
        if clusters['metrics'] is not None:
            m = clusters['metrics'].copy()
            if 'cluster_id' in m.columns:
                m['cluster_id'] += cluster_offset
            merged_metrics_list.append(m)
...
        'depths': np.concatenate(merged_depths),
        'metrics': pd.concat(merged_metrics_list, ignore_index=True) if merged_metrics_list else None,
```
```python
        wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)   # wheel_good unused
            me_binned, me_good = interpolate_behavior_to_bins(...)           # me_good unused
```

iii. Not identified as waste anywhere in CONVERSION_NOTES. The memory consequence is acknowledged only after the fact, in Step 12: *"Full dataset (392 sessions, 23 GB) exceeds 64 GB container memory when loaded + processed. Used representative subset (98 sessions, every 4th) for decoder training."*
