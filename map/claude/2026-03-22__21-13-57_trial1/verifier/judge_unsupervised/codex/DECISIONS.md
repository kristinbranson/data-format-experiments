# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every session by globbing all `.nwb` files under `/app/data/sub-*`, then processes each NWB file independently. Within each NWB file it reads subject metadata, the units table, the trials table, behavioral event timestamps, and the tongue tracking time series.

ii. 
```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))

def convert_all(outfile, sample=False, show_processing=False):
    nwb_files = get_nwb_files()
    ...
    for i, nwb_path in enumerate(nwb_files):
        result = process_session(nwb_path, ...)
```

```python
def load_nwb_session(nwb_path):
    import pynwb

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    ...
    units = nwb.units
    ...
    trials = nwb.trials
    ...
    be = nwb.acquisition['BehavioralEvents']
    ...
    bts = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md`, Step 2 says the dataset contains 28 subject directories and 174 NWB files. The trajectory shows the agent decided to use the NWB files as the authoritative source, because the reference code used DataJoint-exported `.mat` files but the provided data were NWB.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB subject ID. Each processed session stores `subject_id`, and the final output builds a unique ordered subject list and a `subject_idx` array pointing from each kept session to its subject.

ii. 
```python
subject_id = nwb.subject.subject_id
subject_desc = nwb.subject.description
```

```python
subjects_set = OrderedDict()
...
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
...
subjects = list(subjects_set.keys())
subject_idx.append(subjects_set[sess['subject_id']])
subject_idx = np.array(subject_idx, dtype=np.int64)
```

iii. The notes explicitly map `subject.subject_id` to `subjects` and say the dataset has 28 mice. The agent treated each unique NWB subject ID as one mouse.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. A session is processed by `process_session`, and only sessions passing session-level filters are retained in the final lists.

ii. 
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is None:
        continue
    all_sessions.append(result)
```

```python
data = {
    'neural': neural,
    'input': inputs,
    'output': outputs,
    ...
}
```

iii. In Step 2 and Step 5 of `CONVERSION_NOTES.md`, the agent states that each NWB file is a session. The trajectory also reflects this assumption throughout the conversion loop.

## 1-d. How are the data split into trials?

i. Trials are defined from rows of the NWB trials table. After session-level filtering, the code builds a boolean `trial_mask`, converts it to `trial_indices`, and uses those trial indices to select `go_times`, trial-level labels, and aligned neural and behavioral data.

ii. 
```python
trials = nwb.trials
trials_data = {
    'start_time': trials['start_time'][:],
    'stop_time': trials['stop_time'][:],
    'trial_instruction': trials['trial_instruction'][:],
    'outcome': trials['outcome'][:],
    'early_lick': trials['early_lick'][:],
    'auto_water': trials['auto_water'][:],
    'free_water': trials['free_water'][:],
    ...
}
```

```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. The notes say trials come from the NWB trials table and that the decoder keeps most trial types. The trajectory also shows the agent verified that `go_start_times` matched the trial count while `sample_start_times` had extra entries due to early-lick replays.

## 1-e. How are trials filtered based on quality controls?

i. The agent uses two levels of filtering. At the session level it requires performance above 65%, at least 50 correct left trials, and at least 50 correct right trials, computed on control non-early-lick non-ignore trials. At the trial level it excludes only `auto_water` and `free_water` trials, keeps early-lick, ignore, and photostimulation trials for decoder targets/inputs, and later drops trials whose alignment window falls outside the recording range.

ii. 
```python
def compute_session_performance(trials_data):
    ...
    control_mask = np.ones(len(outcomes), dtype=bool)
    for i in range(len(outcomes)):
        if auto_water[i] == 1 or free_water[i] == 1:
            control_mask[i] = False
        if photostim_onset[i] != 'N/A':
            control_mask[i] = False
    no_early_mask = np.array([el == 'no early' for el in early_lick])
    eval_mask = control_mask & no_early_mask
    ...
    non_ignore = eval_outcomes != 'ignore'
    ...
    performance = n_correct / len(eval_outcomes)
```

```python
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_LEFT:
    return None
if correct_right < MIN_CORRECT_RIGHT:
    return None

trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
```

```python
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
if not np.all(recording_mask):
    trial_indices = trial_indices[np.where(recording_mask)[0]]
```

iii. The notes say the reference analysis excluded early lick, auto water, free water, ignore, and stimulation trials, but the decoder needs early lick, ignore, and stimulation variation, so the agent intentionally kept those and excluded only auto/free water at the trial level. The trajectory step around decision-making says this explicitly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units['spike_times']` for units whose `classification` is `"good"`, with `anno_name` used to keep only units that can be mapped into one of the agent’s major brain regions.

ii. 
```python
units = nwb.units
classifications = units['classification'][:]
good_mask = classifications == 'good'
anno_names = units['anno_name'][:]
```

```python
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
...
good_indices = np.where(good_mask)[0]
...
good_anno_names = anno_names[good_mask]
```

iii. The notes map NWB `classification == 'good'` to the classifier-based QC described in the papers/code. They also say `anno_name` is the source for brain region information.

## 2-b. How is the `neural` data processed?

i. For each kept neuron and trial, the agent converts absolute spike times into trial-relative firing rates by histogramming spikes from 2.5 s before to 1.5 s after the go cue into 50 ms non-overlapping bins, then dividing counts by bin width.

ii. 
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
```

```python
def bin_spikes(spike_times_list, go_times, align_start, align_end, bin_width, n_bins):
    bin_edges = np.linspace(align_start, align_end, n_bins + 1)
    all_matrices = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
    for n in range(n_neurons):
        st = spike_times_list[n]
        ...
        for t in range(n_trials):
            go_t = go_times[t]
            abs_start = go_t + align_start
            abs_end = go_t + align_end
            idx_lo = np.searchsorted(st, abs_start, side='left')
            idx_hi = np.searchsorted(st, abs_end, side='left')
            if idx_hi > idx_lo:
                rel_spikes = st[idx_lo:idx_hi] - go_t
                counts, _ = np.histogram(rel_spikes, bins=bin_edges)
                all_matrices[t, n, :] = counts / bin_width
```

iii. The notes say the reference code used `sliding_histogram` with 40 ms bins and 3.4 ms stride, but the agent chose 50 ms non-overlapping bins because the decoder task explicitly required them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only units whose NWB `classification` equals `"good"`, then further excludes good units whose `anno_name` cannot be mapped into its 14-region grouping. Sessions with no remaining mapped good units are discarded.

ii. 
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
...
good_anno_names = anno_names[good_mask]
```

```python
region_labels = []
neuron_mask = []
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
...
if n_neurons < 1:
    print(f'    SKIP: no neurons with valid brain region')
    return None
```

iii. The notes say the reference pipeline used classifier-based QC and grouped neurons into 14 major regions. The trajectory also records the agent’s decision to use `classification == 'good'` as the NWB analogue of the reference QC output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the behavioral event `go_start_times`. For each trial, the code subtracts the trial’s go-cue timestamp from spike times and bins the relative spike times over `[-2.5, 1.5)` s.

ii. 
```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

iii. The notes and trajectory repeatedly say the target alignment event is go cue onset and that NWB spike times are absolute session times, so they must be re-expressed relative to each trial’s go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins and 80 time bins per trial across the 4 s window. The code bins raw spikes directly into that resolution; it does not first compute a finer-resolution representation and then rebin it.

ii. 
```python
BIN_WIDTH = 0.050  # 50 ms bins
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

```python
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
all_matrices[t, n, :] = counts / bin_width
```

iii. The notes explicitly describe the 50 ms decision as a decoder-spec override to the reference code’s 40 ms / 3.4 ms sliding histogram.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The time-from-tone input is derived from `sample_start_times` in `BehavioralEvents`, together with each trial’s `go_start_times`, `start_time`, and `stop_time`.

ii. 
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
...
tone_onsets = get_tone_onset_for_trials(
    go_times, data['sample_start_ts'],
    td['start_time'][trial_indices], td['stop_time'][trial_indices]
)
```

iii. The notes map this variable to “sample start time relative to go cue.” The trajectory shows the agent investigated the extra `sample_start_times` events caused by early-lick replays before deciding how to assign a tone onset to each trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the agent picks the last `sample_start_times` event that occurred between the trial start and that trial’s go cue, treating that as the tone onset for the effective trial attempt. It then computes, for each 50 ms bin center, the elapsed time since that tone onset.

ii. 
```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    ...
    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]
```

```python
def compute_tone_onset_input(go_times, tone_onsets, align_start, align_end, bin_width, n_bins):
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
    ...
    tone_rel = tone_t - go_t
    tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. The notes explain that sample/delay events can replay after early licks, so the agent chose the last sample-start event before the go cue within the trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-aligned bin centers used for the neural data, so each trial gets a `(80,)` tone-time vector aligned bin-for-bin with its neural matrix.

ii. 
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0).astype(np.float32)
```

iii. The agent’s notes and plotting code both frame this input as a time-varying signal defined on the same go-cue-centered bins as the neural activity.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from the `BehavioralEvents` time series `photostim_start_times` and `photostim_stop_times`. The trials-table field `photostim_onset` is only used when computing session performance.

ii. 
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

```python
photostim_input = compute_photostim_input(
    go_times, data['photostim_start_ts'],
    data['photostim_stop_ts'],
    ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS
)
```

iii. The notes’ variable-mapping table maps photostimulation from photostim start/stop times to a binary time-varying decoder input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code initializes an all-zero 80-bin vector, computes each global photostimulation event’s start and stop relative to that trial’s go cue, and sets bins to `1.0` when the bin center falls inside an overlapping photostimulation interval.

ii. 
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts,
                            align_start, align_end, bin_width, n_bins):
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
    ...
    for si in range(len(photostim_start_ts)):
        ps_start = photostim_start_ts[si] - go_t
        ps_stop = photostim_stop_ts[si] - go_t
        if ps_stop < align_start or ps_start > align_end:
            continue
        for b in range(n_bins):
            bc = bin_centers[b]
            if bc >= ps_start and bc < ps_stop:
                ps[b] = 1.0
```

iii. The notes describe photostimulation as a binary on/off signal defined at every decoder time point. The trajectory shows the agent kept stimulation trials specifically because photostim is required as a decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Like the other time-varying signals, it is expressed relative to each trial’s go cue and sampled at the same 50 ms bin centers as the neural data.

ii. 
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
...
inp = np.stack([tone_onset_input[t], photostim_input[t]], axis=0)
```

iii. The notes’ “Temporal Parameters” section states that all decoder variables are aligned to go cue onset on the common 50 ms grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives the decoder’s `choice` output from the trials-table field `trial_instruction`, not from lick-event data.

ii. 
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The notes’ mapping table explicitly maps `trial_instruction` to `output[0]: choice`. The trajectory shows the agent focused on guaranteed per-trial labels and did not ultimately use the loaded lick timestamps.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent performs a binary recoding of `trial_instruction` (`left -> 0`, `right -> 1`) and then broadcasts the resulting per-trial label across all 80 time bins in the trial output array.

ii. 
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
...
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),
    ...
], dtype=np.int64)
```

iii. The notes say choice is represented per trial. The code implements that by repeating the same label over time rather than keeping it as a scalar.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the NWB trials-table column `outcome`.

ii. 
```python
outcomes_raw = td['outcome'][trial_indices]
```

iii. The notes’ mapping table maps `outcome` directly to the decoder output with the specified ignore/miss/hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps string labels to integers with `ignore -> 0`, `miss -> 1`, `hit -> 2`, then broadcasts each trial’s outcome across the 80 output time bins.

ii. 
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
...
np.full(N_BINS, outcomes[t], dtype=np.int64)
```

iii. This exactly follows the mapping written in the instructions and repeated in `CONVERSION_NOTES.md`.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the NWB trials-table column `early_lick`.

ii. 
```python
early_lick_raw = td['early_lick'][trial_indices]
```

iii. The notes explicitly map `early_lick` to the decoder output dimension for early-lick status.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent recodes `early -> 1` and everything else to `0`, then broadcasts that per-trial label across the 80 time bins.

ii. 
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
...
np.full(N_BINS, early_licks[t], dtype=np.int64)
```

iii. The notes say the agent keeps early-lick trials so this variable can be decoded rather than filtered out.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-camera tongue tracking time series `Camera0_side_TongueTracking`. The code uses the second column as y-position, the third column as confidence, and the timestamps for alignment.

ii. 
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

```python
tongue_y = tongue_data[:, 1].astype(np.float64)
tongue_conf = tongue_data[:, 2].astype(np.float64)
```

iii. The notes’ mapping table and Step 2 dataset summary both identify side-camera tongue tracking as the raw source.

## 8-b. How is `output` *Tongue y-position* processed?

i. The code computes the session mean tongue y-position over visible frames (`confidence >= 0.9`), imputes low-confidence frames with that mean, extracts frames inside each trial’s go-aligned analysis window, and averages the imputed y-values within each 50 ms bin.

ii. 
```python
visible_mask = tongue_conf >= confidence_threshold
if np.sum(visible_mask) > 0:
    session_mean_y = np.mean(tongue_y[visible_mask])
else:
    session_mean_y = np.mean(tongue_y)

tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y
```

```python
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
trial_y = tongue_y_imputed[mask]
...
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. The notes say the agent followed the paper’s statement that occluded tongue positions should be set to the mean value, and Step 6 notes it later vectorized this processing.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After binning tongue y within the kept trial windows, the agent concatenates all binned values across that session, computes the 40th and 60th percentiles, and maps bins below the 40th percentile to 0, between thresholds to 1, and above the 60th percentile to 2. If the two percentiles are equal, it perturbs them slightly to force three categories.

ii. 
```python
all_values = np.concatenate([t for t in tongue_y_trials])

p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)

if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps
```

```python
d = np.zeros(len(trial_y), dtype=np.int64)
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```

iii. The notes say this follows the instruction’s per-session percentile discretization, and Step 6 records the special handling for the common `p40 == p60` case caused by imputed mean values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue signal is restricted to the same `[-2.5, 1.5)` s window around each trial’s go cue and averaged into the same 80 bins as the neural data, producing one discrete tongue label per neural time bin.

ii. 
```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
...
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. The notes’ temporal-parameter section says all modalities are aligned to go cue onset with 50 ms bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases heuristically: if no tone onset is found it fills the tone-onset input with zeros; if no tongue frame is visible it falls back to the global tongue mean; low-confidence tongue frames are imputed with the session mean; if a trial has no tongue samples in the alignment window it fills the whole trial with the mean; neurons with unmapped brain regions are dropped; sessions with no retained neurons or fewer than two retained trials are skipped; and trials outside the spike recording range are dropped.

ii. 
```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
```

```python
if np.sum(visible_mask) > 0:
    session_mean_y = np.mean(tongue_y[visible_mask])
else:
    session_mean_y = np.mean(tongue_y)
...
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
```

```python
if n_valid_trials < 2:
    return None
...
if n_neurons < 1:
    return None
...
if not np.all(recording_mask):
    trial_indices = trial_indices[np.where(recording_mask)[0]]
```

iii. These choices are summarized in Step 5 and Step 6 of `CONVERSION_NOTES.md`, which mention missing/occluded tongue data, keeping sessions only if they have enough usable trials, and later adding a recording-range filter.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identified spike binning and tongue processing as the main runtime costs, with NWB loading secondary. The full-run log shows per-session times dominated by `Spike binning` and `Tongue processing`.

ii. 
```python
t1 = time.time()
neural_trials = bin_spikes(...)
t_bin = time.time() - t1
print(f'    Spike binning: {t_bin:.1f}s')
```

```python
t2 = time.time()
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
tongue_y_discrete = discretize_tongue_y(tongue_y_trials, session_mean_y)
t_tongue = time.time() - t2
print(f'    Tongue processing: {t_tongue:.1f}s')
```

iii. Step 7 of the notes includes runtime estimates per session and explicitly lists NWB loading, spike binning, and tongue processing as the dominant costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The final script still contains several expensive nested loops that could be vectorized further: the neuron-by-trial spike loop in `bin_spikes`, the trial-by-bin averaging loop in `compute_tongue_y_per_trial`, the trial-by-event-by-bin loop in `compute_photostim_input`, and the trial loop used to search for the last sample start in `get_tone_onset_for_trials`.

ii. 
```python
for n in range(n_neurons):
    ...
    for t in range(n_trials):
        ...
```

```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        in_bin = trial_y[bin_indices == b]
```

```python
for t in range(n_trials):
    ...
    for si in range(len(photostim_start_ts)):
        ...
        for b in range(n_bins):
```

iii. Step 6 of the notes says the agent already optimized spike binning and tongue processing relative to an earlier version, which implies it recognized these loops as bottlenecks.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs the same bin-center or bin-edge grids in multiple functions, repeatedly scans global event arrays trial-by-trial (`sample_start_ts`, `photostim_start_ts`, `photostim_stop_ts`), and repeatedly broadcasts scalar per-trial outputs into full-length 80-bin vectors.

ii. 
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
...
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

```python
for i in range(n_trials):
    mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
```

```python
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),
    np.full(N_BINS, outcomes[t], dtype=np.int64),
    np.full(N_BINS, early_licks[t], dtype=np.int64),
    tongue_y_discrete[t].astype(np.int64),
], dtype=np.int64)
```

iii. The notes discuss repeated runtime hotspots and the agent’s attempts to optimize them, especially for spike and tongue processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads some data it never uses in the converted outputs: `left_lick_ts`, `right_lick_ts`, `photostim_power`, `photostim_duration`, `subject_desc`, and `n_total_units`. It also builds plotting outputs and timing summaries that are not part of `converted_data.pkl`. Most notably, it loads lick timestamps but does not use them to compute the decoder’s `choice` variable.

ii. 
```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
...
'left_lick_ts': left_lick_ts,
'right_lick_ts': right_lick_ts,
```

```python
'photostim_power': trials['photostim_power'][:],
'photostim_duration': trials['photostim_duration'][:],
```

```python
if show_processing:
    plot_processing(...)
```

iii. The notes focus on these variables during exploration, but the final dataset structure does not consume them except for session-level metadata and optional plots.
