# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates `/app/data/sub-*` directories with `os.listdir`, collects every `.nwb` file beneath them, and opens each session file with `pynwb.NWBHDF5IO`. Within each opened NWB file it reads the `units`, `trials`, `BehavioralEvents`, and `BehavioralTimeSeries` objects directly.

ii.
```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for f in nwb_files:
            all_files.append((subj, os.path.join(subj_dir, f)))
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
units = nwb.units
trials = nwb.trials
be = nwb.acquisition['BehavioralEvents']
bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset is organized as one NWB file per session under per-subject folders and that the NWB files contain the needed trials, units, and behavioral streams, so it treated the folder/file layout as the complete dataset.

## 1-b. How are the data split into subjects?

i. The AI uses the subject folder name, e.g. `sub-440956`, as the subject identifier for every session. It does not read `nwb.subject.subject_id`; instead it threads the directory name through processing and builds `subjects` and `subject_idx` from those folder-derived IDs.

ii.
```python
for file_idx, (subj, fpath) in enumerate(nwb_files):
    result = process_session(fpath, subj, ...)
```

```python
return {
    'subject_id': subject_id,
    'session_id': session_id,
    ...
}
```

```python
seen_subjects = {}
subject_list = []
for sd in session_data_list:
    subj = sd['subject_id']
    if subj not in seen_subjects:
        seen_subjects[subj] = len(seen_subjects)
        subject_list.append(subj)
```

iii. The notes describe `/app/data/sub-XXXXXX/` as one directory per subject and report 28 such folders, so the AI treated the directory name as the subject boundary and subject ID.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session boundaries come from the file list itself, and each session is identified by `nwb.identifier`.

ii.
```python
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for f in nwb_files:
        all_files.append((subj, os.path.join(subj_dir, f)))
```

```python
session_id = nwb.identifier
```

iii. In the notes, the AI explicitly states that each subject directory contains one NWB file per session, so it used one-file-per-session as the session split.

## 1-d. How are the data split into trials?

i. The AI uses rows of the NWB `trials` table as trials and indexes behavioral event arrays by those row indices after trial filtering. It assumes `go_start_times` corresponds one-to-one with the trial rows.

ii.
```python
trials = nwb.trials
n_trials_total = len(trials)
trial_instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
early_licks = trials['early_lick'][:]
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
go_cues_valid = go_start_times[valid_trial_indices]
```

iii. The notes summarize the NWB trials table columns and describe the dataset as having one row per behavioral trial, so the AI used the trials table directly rather than reconstructing trials from event streams.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out `auto_water` and `free_water` trials, but keeps early-lick, ignore, miss, and photostimulation trials because they are decoder outputs or inputs. It does not use `units/obs_intervals`, so trials with no neural coverage remain in the dataset as all-zero neural trials. Sessions with fewer than two remaining trials are dropped.

ii.
```python
auto_water = np.array(trials['auto_water'][:])
free_water = np.array(trials['free_water'][:])

# Filter: exclude auto_water and free_water
trial_mask = (auto_water == 0) & (free_water == 0)
valid_trial_indices = np.where(trial_mask)[0]
...
if n_valid < 2:
    return None
```

iii. The notes say: “Since we decode early_lick, outcome (incl. ignore), and use photostim as input, we keep these. We exclude only auto_water and free_water trials.” Later the notes explicitly acknowledge 1,061 all-zero neural trials and justify keeping them as trials with valid behavior but no neural coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units['spike_times']` for units whose `classification` equals `'good'`, using `BehavioralEvents/go_start_times` to define per-trial windows.

ii.
```python
classification = np.array(units['classification'][:])
good_mask = classification == 'good'
...
for i in good_indices:
    st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
    spike_times_list.append(st)
```

```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
go_cues_valid = go_start_times[valid_trial_indices]
```

iii. The notes identify classifier-based QC as the neural unit curation step and describe spike times as absolute session time that must be aligned to go cues.

## 2-b. How is the `neural` data processed?

i. For each retained trial and retained neuron, the AI extracts spikes in the `[-2.5 s, +1.5 s]` window around the go cue, bins them into 50 ms non-overlapping bins with `np.histogram`, and converts counts to firing rates by dividing by bin width. No smoothing or normalization is applied.

ii.
```python
def compute_firing_rates_fast(spike_times_list, go_cue_times, n_neurons):
    ...
    for t_idx in range(n_trials):
        gc = go_cue_times[t_idx]
        t_start = gc - TIME_BEFORE
        t_end = gc + TIME_AFTER
        edges = BIN_EDGES + gc
        ...
        for n_idx in range(n_neurons):
            st = spike_times_list[n_idx]
            i_start = np.searchsorted(st, t_start, side='left')
            i_end = np.searchsorted(st, t_end, side='left')
            spikes_in_window = st[i_start:i_end]

            if len(spikes_in_window) > 0:
                counts = np.histogram(spikes_in_window, bins=edges)[0]
                fr_matrix[n_idx, :] = counts / BIN_WIDTH
```

iii. The notes say the task requires 50 ms bins and that spike times in NWB must be aligned to go cue time, so the AI implemented direct per-window binning into firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'`. It drops any session with zero such units.

ii.
```python
classification = np.array(units['classification'][:])
good_mask = classification == 'good'
n_good = int(np.sum(good_mask))

if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None
```

iii. The notes explicitly map NWB `classification == 'good'` to the classifier-based QC described in the reference materials and say that sessions with zero good units should be excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by converting the relative bin edges `[-2.5, 1.5]` into absolute times for each trial (`BIN_EDGES + gc`) and binning absolute spike times against those edges.

ii.
```python
gc = go_cue_times[t_idx]
t_start = gc - TIME_BEFORE
t_end = gc + TIME_AFTER
edges = BIN_EDGES + gc  # absolute time edges
```

iii. The notes say NWB spike times are absolute session time while the decoder requires go-cue alignment, so the AI used go cue timestamps as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins over a 4 s window, giving 80 bins per trial. The AI does not perform additional temporal rebinning or smoothing beyond this direct binning.

ii.
```python
BIN_WIDTH = 0.050
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
N_BINS = int((TIME_BEFORE + TIME_AFTER) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The notes say the task specification overrides the reference code’s 40 ms / 3.4 ms setup, so the AI used non-overlapping 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does not derive this input from any per-trial raw data field. Instead it assumes a fixed task timing where tone onset is always exactly 1.85 s before the go cue and derives the input from that constant offset plus bin centers.

ii.
```python
TONE_OFFSET = -1.85  # tone onset relative to go cue (sample=0.65s + delay=1.2s)
...
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. The notes say “Tone onset = go_cue - 1.85s” and describe the input mapping as “Continuous ramp: bin_center + 1.85,” so the AI intentionally used the nominal task structure instead of `sample_start_times`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a single session-independent ramp equal to seconds since an assumed fixed tone time. Every trial gets the same 80-length vector; there is no per-trial lookup of sample onset and no correction for repeated sample epochs after early licks.

ii.
```python
# Time from tone onset (same for all trials)
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
...
input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)
```

iii. The notes explicitly justify this as using the nominal task epochs: 0.65 s sample plus 1.2 s delay, hence a fixed 1.85 s offset from go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input to the neural data by expressing it on the same 80 bin centers used for neural data, but with a fixed trial-invariant offset from go cue rather than a per-trial tone time.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. The AI’s notes describe the input as a go-cue-relative ramp, with tone onset fixed at `-1.85 s` relative to the same alignment event used for neural binning.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from the trials table’s `photostim_onset` and `photostim_duration` columns.

ii.
```python
photostim_starts = None
photostim_stops = None
if 'photostim_start_times' in be.time_series:
    photostim_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes say the photostim input comes “From photostim_start/stop_times,” and the trajectory shows the AI explicitly deciding to use those event timestamps for alignment.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI builds a binary time series per trial. For each trial, it scans all session photostim events, keeps those whose absolute interval overlaps the trial window, converts them to go-cue-relative start and stop times, and marks bins whose centers fall within the active interval as `1`.

ii.
```python
for t_idx in range(n_trials):
    gc = go_cue_times[t_idx]
    photostim_binary = np.zeros(N_BINS, dtype=np.float32)
    ...
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        if ps_stop < trial_start or ps_start > trial_end:
            continue
        ps_start_rel = ps_start - gc
        ps_stop_rel = ps_stop - gc
        active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
        photostim_binary[active] = 1.0
```

iii. The notes describe this input as “Binary: 1 if photostim active at timepoint,” and the trajectory says the AI wanted a per-time-bin flag rather than a per-trial indicator.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is performed by expressing each photostim interval relative to the same go cue used for neural alignment and comparing those relative times to the shared bin centers.

ii.
```python
ps_start_rel = ps_start - gc
ps_stop_rel = ps_stop - gc
active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
```

iii. The AI’s reasoning was that both neural data and photostim need to live on the go-cue-centered bin grid, so event timestamps were shifted onto that same axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the trials table’s `trial_instruction` and `outcome` fields. There is no direct choice column in the file.

ii.
```python
trial_instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
...
instruction = trial_instructions[trial_idx]
outcome = outcomes[trial_idx]
choice_val = get_choice(instruction, outcome)
```

```python
def get_choice(trial_instruction, outcome):
    if outcome == 'ignore':
        return 2
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

iii. The notes say choice should be “hit + instruction -> correct side; miss + instruction -> wrong side; ignore -> no_lick.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps derived choice to `0=left`, `1=right`, `2=no_lick` and repeats that value across all 80 time bins in the output tensor.

ii.
```python
choice_val = get_choice(instruction, outcome)
...
output_combined = np.zeros((4, N_BINS), dtype=np.int64)
output_combined[0, :] = choice_val
```

iii. The notes describe choice as a per-trial output with values `['left', 'right', 'no_lick']`, so the AI stored it as a constant row over time.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials table `outcome` column.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome = outcomes[trial_idx]
```

iii. The notes describe outcome as a direct mapping from the NWB trial field `outcome (hit/miss/ignore)`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore'`, `'miss'`, and `'hit'` to `0`, `1`, and `2`, then repeats that value across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_val = outcome_map.get(outcome, 0)
output_combined[1, :] = outcome_val
```

iii. The notes specify `output[1]: outcome` with values `['ignore','miss','hit']`, so the code implements a direct categorical encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials table `early_lick` column.

ii.
```python
early_licks = trials['early_lick'][:]
...
early_val = early_map.get(early_licks[trial_idx], 0)
```

iii. The notes describe early lick as a direct mapping from the trial field `early_lick (early/no early)`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early'` to `0` and `'early'` to `1`, then repeats that value across all 80 bins.

ii.
```python
early_map = {'no early': 0, 'early': 1}
...
output_combined[2, :] = early_val
```

iii. The notes specify `output[2]: early_lick` with values `['no','yes']`, so the code converts the NWB labels into binary categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 is tongue `y`, column 2 is DLC likelihood, and the timestamps are used for alignment.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_timestamps_all = tongue_ts_obj.timestamps[:]
tongue_y_all = tongue_data_all[:, 1]
tongue_lik_all = tongue_data_all[:, 2]
```

iii. The notes identify `Camera0_side_TongueTracking (x, y, likelihood)` as the relevant behavioral stream for tongue position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first defines visible frames as those with DLC likelihood at least `0.9`. It computes the session 40th and 60th percentiles from raw visible-frame `y` values over the whole session. Then, for each trial and each 50 ms bin, it averages visible `y` values within that bin and classifies the bin by those session-level percentile thresholds. Bins with no visible frames remain class `3`.

ii.
```python
DLC_LIKELIHOOD_THRESH = 0.9
...
visible_mask = tongue_lik_all >= DLC_LIKELIHOOD_THRESH
if np.sum(visible_mask) >= 10:
    visible_y = tongue_y_all[visible_mask]
    p40 = float(np.percentile(visible_y, 40))
    p60 = float(np.percentile(visible_y, 60))
```

```python
visible = l_bin >= DLC_LIKELIHOOD_THRESH

if np.any(visible):
    mean_y = np.mean(y_bin[visible])
    if mean_y < p40:
        tongue_y_binned[b] = 0
    elif mean_y <= p60:
        tongue_y_binned[b] = 1
    else:
        tongue_y_binned[b] = 2
```

iii. The notes say “Use DLC likelihood threshold of 0.9 to determine tongue visibility” and describe tongue y as a per-session discretization with session-level percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses per-session thresholds `p40` and `p60` computed from raw visible `y` samples. For each bin, it assigns `0` if mean visible `y < p40`, `1` if `p40 <= y <= p60`, `2` if `y > p60`, and `3` if there were no visible frames in that bin.

ii.
```python
p40 = float(np.percentile(visible_y, 40))
p60 = float(np.percentile(visible_y, 60))
```

```python
tongue_y_binned = np.full(N_BINS, 3, dtype=np.int64)
...
if mean_y < p40:
    tongue_y_binned[b] = 0
elif mean_y <= p60:
    tongue_y_binned[b] = 1
else:
    tongue_y_binned[b] = 2
```

iii. The notes map tongue y to `['low','mid','high','not_visible']` and describe the categories as below 40th percentile, 40th-60th percentile, above 60th percentile, and not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data by taking camera timestamps within each trial’s `[-2.5, 1.5]` go-cue-centered window, converting those timestamps to go-cue-relative time, assigning them to the same 50 ms bins, and storing one class per neural bin.

ii.
```python
t_start = gc + BIN_EDGES[0]
t_end = gc + BIN_EDGES[-1]
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')
...
t_rel = t_slice - gc
bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1
```

iii. The notes say all streams are aligned to go cue with 50 ms bins, so the AI used the camera timestamps on that same go-cue-relative grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles three cases. Sessions with zero good units are dropped. Missing tongue visibility is represented as class `3` (`not visible`) by thresholding on DLC likelihood and leaving bins without visible frames at the default class. Trials with no neural coverage are not removed; instead they are left in place and become all-zero neural matrices.

ii.
```python
if n_good == 0:
    print(f"  Skipping {session_id}: 0 good units")
    io.close()
    return None
```

```python
tongue_y_binned = np.full(N_BINS, 3, dtype=np.int64)
...
visible = l_bin >= DLC_LIKELIHOOD_THRESH
if np.any(visible):
    ...
```

```python
trial_neural = compute_firing_rates_fast(spike_times_list, go_cues_valid, n_good)
```

iii. The notes explicitly justify keeping zero-neural trials by saying the behavior is valid even when Neuropixels recording did not span the full session, and they justify tongue class `3` as representing non-visible tongue periods.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by session-by-session NWB reads, loading spike times for every good unit, and especially the nested firing-rate computation loop over trials and neurons. Tongue processing and photostim scanning are secondary costs.

ii.
```python
for i in good_indices:
    st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
    spike_times_list.append(st)
```

```python
for t_idx in range(n_trials):
    ...
    for n_idx in range(n_neurons):
        ...
        counts = np.histogram(spikes_in_window, bins=edges)[0]
```

iii. The notes report full conversion time of 582 s and describe the implementation as optimized around `np.searchsorted` and `np.digitize`, implying that spike loading and firing-rate generation were the intended main costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain obviously vectorizable: the outer trial loop and inner neuron loop in `compute_firing_rates_fast`, the per-bin loop in `compute_tongue_y_fast`, and the per-trial scan over all session photostim events in `compute_photostim_input_fast`.

ii.
```python
for t_idx in range(n_trials):
    ...
    for n_idx in range(n_neurons):
        ...
```

```python
for b in range(N_BINS):
    in_bin = bin_assignments == b
    ...
```

```python
for t_idx in range(n_trials):
    ...
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        ...
```

iii. The notes claim some optimization, but the code still leaves many Python loops in place, so the AI appears to have accepted partial rather than full vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations across trials that could have been shared: it rebuilds firing-rate windows and histograms for each trial-neuron pair, scans the full photostim event list separately for every trial, and rechecks bin membership for every tongue bin within every trial.

ii.
```python
for t_idx in range(n_trials):
    gc = go_cue_times[t_idx]
    ...
    for n_idx in range(n_neurons):
        ...
```

```python
for t_idx in range(n_trials):
    ...
    for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
        ...
```

iii. The AI’s notes emphasize direct per-trial construction of inputs and outputs, and the code reflects that by recomputing trial-local binning logic repeatedly rather than flattening more work across trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Beyond the required dataset fields, the code does extra work for logging and diagnostics: it computes region-count summaries, output distributions, ETA estimates, and contains optional plotting code. These do not affect the saved converted dataset. The conversion path does not otherwise compute a major derived signal and then discard it.

ii.
```python
print(f"\nNeuron counts by region:")
region_counts = {r: 0 for r in brain_regions}
for sd in session_data_list:
    for r in sd['regions']:
        region_counts[r] += 1
```

```python
print(f"\nOutput distributions:")
for out_idx, out_name in enumerate(['choice', 'outcome', 'early_lick']):
    counts = {}
    ...
```

```python
if show_processing:
    _plot_processing(...)
```

iii. The notes present these as sanity checks and documentation rather than part of the dataset itself, so the AI appears to have considered the extra work acceptable validation overhead.
