# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed all NWB files under `/app/data/sub-*/sub-*_ses-*.nwb`, then loaded each file with `pynwb.NWBHDF5IO`. Inside each file it eagerly read the units table, trials table, behavioral event streams, and tongue tracking time series into Python/Numpy objects.

ii.
```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
...
units = nwb.units
trials = nwb.trials
...
be = nwb.acquisition['BehavioralEvents']
...
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. The notes say the dataset is organized as 174 NWB files across 28 subject directories, and the trajectory says the NWB file is the canonical published format and contains all required streams.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `nwb.subject.subject_id` for each loaded session. The output `subjects` list preserves first-seen order via an `OrderedDict`, and `subject_idx` stores the per-session integer index into that list.

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
...
subject_idx.append(subjects_set[sess['subject_id']])
```

iii. Step 5 of the notes maps `subject.subject_id` directly to `subjects`. The trajectory also treats the NWB subject field as the authoritative mouse identifier.

## 1-c. How are the data split into sessions?

i. The AI treated each NWB file as one session. Sessions are processed one file at a time and appended to the output only if they pass the AI's session-level filters.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(
        nwb_path,
        show_processing=show_processing and i < 2,
        session_idx=i
    )

    if result is None:
        continue

    all_sessions.append(result)
```

iii. The notes repeatedly describe the dataset as "174 NWB files total" and the trajectory discusses the 174 files as the session inventory, then filters sessions by quality/performance.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. The code records `n_trials = len(trials)`, copies selected trial columns into `trials_data`, and later filters trials with a boolean mask and uses the surviving row indices as the kept trial list.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trials_data = {
    'start_time': trials['start_time'][:],
    'stop_time': trials['stop_time'][:],
    'trial_instruction': trials['trial_instruction'][:],
    'outcome': trials['outcome'][:],
    'early_lick': trials['early_lick'][:],
    'auto_water': trials['auto_water'][:],
    'free_water': trials['free_water'][:],
    'photostim_onset': trials['photostim_onset'][:],
    'photostim_power': trials['photostim_power'][:],
    'photostim_duration': trials['photostim_duration'][:],
}
```

```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. The trajectory states that `go_start_times` map 1:1 with trials and that repeated sample events, not repeated trials, are the main complication.

## 1-e. How are trials filtered based on quality controls?

i. The AI excluded `auto_water` and `free_water` trials, then removed entire sessions if they failed behavioral performance criteria (`>65%` performance and at least `50` correct left and right trials). It also removed kept trials whose go-cue-centered neural window fell outside the min/max spike time range of the retained neurons.

ii.
```python
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
...
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_LEFT:
    return None
if correct_right < MIN_CORRECT_RIGHT:
    return None
```

```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
```

```python
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
if not np.all(recording_mask):
    ...
    trial_indices = trial_indices[valid_positions]
```

iii. The notes explicitly justify excluding `auto_water` and `free_water`, keeping early-lick/ignore/stim trials for decoder targets, and applying the paper's session selection criteria. Step 10 of the notes justifies the later "recording range" trial exclusion as a fix for all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units with `classification == 'good'`, together with `BehavioralEvents/go_start_times` to define the go-cue-centered bin windows.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
...
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes say the NWB `classification == 'good'` field is the classifier-based QC output, and the trajectory identifies `spike_times` plus go-cue alignment as the intended neural source.

## 2-b. How is the `neural` data processed?

i. The AI converts absolute spike times into 50 ms go-cue-centered firing rates. For each neuron and each trial it isolates spikes in the `[go-2.5, go+1.5)` window, histograms relative spike times into 80 bins, and divides counts by bin width to produce Hz.

ii.
```python
def bin_spikes(spike_times_list, go_times, align_start, align_end, bin_width, n_bins):
    ...
    bin_edges = np.linspace(align_start, align_end, n_bins + 1)
    all_matrices = np.zeros((n_trials, n_neurons, n_bins), dtype=np.float32)
    ...
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

iii. The notes justify 50 ms non-overlapping bins as a decoder-task override of the reference code's 40 ms / 3.4 ms settings. The trajectory also states the intent was firing-rate binning around the go cue.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, only units with `classification == 'good'` are retained. Second, the AI discards otherwise-good units whose `anno_name` cannot be mapped into one of its 14 coarse brain-region labels. Finally, sessions are dropped if no mapped neurons remain.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
anno_names = units['anno_name'][:]
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
    return None
```

iii. The notes say to use classifier-based QC and to map detailed `anno_name` values into 14 major regions. Step 10 of the notes documents several region-mapping fixes, confirming that the region map was part of the effective filtering path.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to go-cue onset. The code subtracts/centers on each trial's `go_time` and bins spikes in a shared `[-2.5, +1.5]` second window around that event.

ii.
```python
ALIGN_START = -2.5
ALIGN_END = 1.5
BIN_WIDTH = 0.050
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
```

```python
go_t = go_times[t]
abs_start = go_t + align_start
abs_end = go_t + align_end
...
rel_spikes = st[idx_lo:idx_hi] - go_t
```

iii. Both the notes and trajectory explicitly say everything should be aligned to go cue onset because the NWB timestamps are absolute session times rather than already go-aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms non-overlapping bins. No smoothing or secondary rebinning is applied after the initial histogramming.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins (decoder task spec)
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

```python
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
all_matrices[t, n, :] = counts / bin_width
```

iii. Step 5 of the notes explicitly calls 50 ms bins a task-driven override of the paper's 40 ms preprocessing.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `go_start_times`, and the trial `start_time`/`stop_time` columns. For each kept trial it finds the last `sample_start` inside that trial before the go cue.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
```

```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    ...
    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]
```

iii. The trajectory explicitly justifies using the last sample-start before the go cue because early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes bin centers in go-cue coordinates, then converts them to elapsed time since tone onset by subtracting the tone time relative to the go cue. If no tone is found, it fills the trial with zeros.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
...
tone_rel = tone_t - go_t
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
```

iii. The notes describe this variable as continuous time from the last sample start before go cue. The code comment says the zero-fill branch is only a fallback and "shouldn't happen for valid trials."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same 80 go-cue-centered 50 ms bin centers as the neural activity, so each time point directly matches a neural bin.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

```python
neural_trials = bin_spikes(spike_times_list, go_times, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
...
tone_onset_input = compute_tone_onset_input(go_times, tone_onsets, ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
```

iii. The notes state that all streams are aligned to the go cue using the same temporal window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the session-wide behavioral-event streams `photostim_start_times` and `photostim_stop_times`, not from the per-trial `photostim_onset` / `photostim_duration` columns used elsewhere in the NWB trials table.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

```python
photostim_input = compute_photostim_input(go_times, data['photostim_start_ts'],
                                           data['photostim_stop_ts'],
                                           ALIGN_START, ALIGN_END, BIN_WIDTH, N_BINS)
```

iii. Step 5 of the notes says photostimulation should come from "Photostim start/stop times" and be converted into a binary time-varying signal.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI subtracts that trial's go cue from every photostim start/stop event in the session and marks a bin as `1` if its center lies within any event's interval. The result is a binary time series per trial.

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts,
                            align_start, align_end, bin_width, n_bins):
    ...
    bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
    ...
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)

        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            ...
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
```

iii. The notes justify representing photostim as a time-varying binary variable instead of a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to the neural data by expressing each session photostim event in coordinates relative to each trial's go cue and then comparing those intervals against the same go-cue-centered bin centers used for neural and tone inputs.

ii.
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
```

```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
if bc >= ps_start and bc < ps_stop:
    ps[b] = 1.0
```

iii. The notes and trajectory consistently frame go cue onset as the common temporal origin for all derived streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, choice is derived only from the `trial_instruction` column (`left` or `right`). The AI does not use `outcome` to distinguish actual lick direction from misses or no-lick trials.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The trajectory shows the AI originally thought through using outcome labels, but Step 5 of the notes mapped `trial_instruction` directly to `output[0]: choice` as `left=0, right=1`, and that is what the final code implements.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `left -> 0` and `right -> 1`, broadcasts that per-trial value across all 80 bins, and defines only two output labels for choice.

ii.
```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

```python
out = np.array([
    np.full(N_BINS, choices[t], dtype=np.int64),
    np.full(N_BINS, outcomes[t], dtype=np.int64),
    np.full(N_BINS, early_licks[t], dtype=np.int64),
    tongue_y_discrete[t].astype(np.int64),
], dtype=np.int64)
```

iii. The notes justify this as a per-trial decoder output based on the task's left/right structure, but they do not justify the omission of a no-lick class.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trials-table `outcome` column.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
```

iii. The notes explicitly map `outcome` to the decoder output and treat the trials-table field as authoritative.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to `ignore=0`, `miss=1`, `hit=2`, then repeated across all 80 bins for each trial.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

```python
np.full(N_BINS, outcomes[t], dtype=np.int64)
```

iii. Step 5 of the notes gives exactly this coding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` field.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
```

iii. The notes map the NWB `early_lick` field directly onto the decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'early' -> 1` and everything else to `0`, then broadcasts the per-trial result across all bins.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

```python
np.full(N_BINS, early_licks[t], dtype=np.int64)
```

iii. Step 5 of the notes explicitly calls for `no=0, yes=1`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-camera tongue tracking series: `tongue_data[:, 1]` as y-position, `tongue_data[:, 2]` as confidence, and `tongue_timestamps` for time alignment.

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

iii. The notes identify side-camera tongue tracking as the source and mention confidence-based occlusion handling.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI treats frames with confidence below `0.9` as occluded and imputes them to the session mean y-value. For each trial it bins the imputed y-values into 50 ms go-cue-centered bins by averaging frames within each bin. If a whole trial window has no frames, it fills every bin with the session mean. It then discretizes those per-bin values later.

ii.
```python
TONGUE_CONFIDENCE_THRESHOLD = 0.9
```

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
trial_tongue_y = np.full(n_bins, session_mean_y, dtype=np.float64)
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. Step 5 of the notes says "set tongue position to session mean when confidence < 0.9", citing the paper's occlusion language. Step 6 says an epsilon fix was added when the percentile thresholds collapsed.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI concatenates all per-trial binned tongue values, computes the 40th and 60th percentiles over that pooled set, perturbs the thresholds if they are equal, and assigns categories `0`, `1`, or `2`. It does not emit a separate "not visible" category in the final output.

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

iii. The notes say the discretization is per-session at the 40th/60th percentiles, and Step 6 explicitly justifies the epsilon adjustment when `p40 == p60`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-cue-centered `[-2.5, +1.5]` window and 50 ms bins as the neural data. For each trial it subtracts the go cue from the camera timestamps and bins the frames in those aligned coordinates.

ii.
```python
window_start = go_t + align_start
window_end = go_t + align_end
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
```

```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
bin_indices = np.digitize(trial_ts, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
```

iii. The notes consistently describe the tongue output as a time-varying signal aligned to go cue onset, just like the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases with ad hoc rules: sessions with no mapped good units are dropped; trials outside the global spike-time range are dropped; missing tone onsets are filled with zeros; low-confidence tongue frames are imputed to the session mean; trials with no tongue frames get all bins set to the session mean; sessions without optogenetic trials rely on empty/all-zero photostim output.

ii.
```python
if n_neurons < 1:
    print(f'    SKIP: no neurons with valid brain region')
    return None
```

```python
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)
```

```python
tongue_y_imputed[~visible_mask] = session_mean_y
...
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))
```

iii. The notes justify dropping trials beyond recording range and imputing tongue values when occluded. The trajectory also mentions checking that non-optogenetic sessions can safely use all-zero photostim traces.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's notes identify three main costs: NWB loading, spike binning, and tongue processing. The printed timing in `process_session()` also makes those three stages the explicit bottleneck checkpoints.

ii.
```python
t0 = time.time()
data = load_nwb_session(nwb_path)
t_load = time.time() - t0
...
t1 = time.time()
neural_trials = bin_spikes(...)
t_bin = time.time() - t1
...
t2 = time.time()
tongue_y_trials, session_mean_y = compute_tongue_y_per_trial(...)
t_tongue = time.time() - t2
```

iii. Step 7 of the notes reports approximately 1.5 s/session for NWB loading, 3.3 s/session for spike binning, and 1.5 s/session for tongue processing, and Step 6 discusses optimization of the latter two.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the nested neuron x trial loop in `bin_spikes()`, the trial x event x bin loop in `compute_photostim_input()`, the trial x bin loop in `compute_tongue_y_per_trial()`, and smaller loops in performance computation and output assembly.

ii.
```python
for n in range(n_neurons):
    ...
    for t in range(n_trials):
        ...
        counts, _ = np.histogram(rel_spikes, bins=bin_edges)
```

```python
for t in range(n_trials):
    ...
    for si in range(len(photostim_start_ts)):
        ...
        for b in range(n_bins):
            ...
```

```python
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. Step 6 of the notes says the AI already optimized spike and tongue handling relative to earlier versions, but the final code still contains these loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations across functions and trials: it recomputes the same linear bin-center arrays in multiple helper functions, scans the entire session's photostim events for every kept trial, loops through all bins inside each trial for tongue averaging, and rebuilds repeated per-trial output arrays with `np.full()` inside the trial loop.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

```python
for t in range(n_trials):
    ...
    for si in range(len(photostim_start_ts)):
        ...
```

```python
for t in range(n_valid_trials):
    out = np.array([
        np.full(N_BINS, choices[t], dtype=np.int64),
        np.full(N_BINS, outcomes[t], dtype=np.int64),
        np.full(N_BINS, early_licks[t], dtype=np.int64),
        tongue_y_discrete[t].astype(np.int64),
    ], dtype=np.int64)
```

iii. There is no strong explicit justification beyond readability and incremental optimization. The notes only mention that earlier versions were slower and that some hotspots were partially vectorized.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores several fields that are not used to build the final decoder arrays: `subject_desc`, `stop_time`, `photostim_power`, `left_lick_ts`, `right_lick_ts`, and `n_total_units`. It also includes optional plotting code and helper data for visualization that are irrelevant to the final pickle.

ii.
```python
subject_desc = nwb.subject.description
...
'stop_time': trials['stop_time'][:],
...
'photostim_power': trials['photostim_power'][:],
...
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
...
'n_total_units': n_units,
```

iii. The trajectory suggests these extra reads came from exploration, validation, and plotting needs rather than from a deliberate minimal final design. There is no clear justification in the notes for keeping them in the production path.
