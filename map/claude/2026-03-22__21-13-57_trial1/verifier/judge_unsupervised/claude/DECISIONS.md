# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from NWB files stored in `/app/data/sub-*/sub-*_ses-*.nwb`. The code globs for all NWB files, sorts them, then iterates through each file calling `load_nwb_session()` which opens the file with `pynwb.NWBHDF5IO`, reads units, trials, behavioral events, and tongue tracking, then closes the file. Each file corresponds to one session.

ii.
```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))

def load_nwb_session(nwb_path):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    # ... reads subject, units, trials, behavioral events, tongue tracking
    io.close()
    return { ... }

# In convert_all():
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. The agent identified that the reference code loaded from .mat files via DataJoint, but the same underlying data is available in NWB format from DANDI. The NWB files contain the same information in a different format. The agent noted this discrepancy and resolved it by using pynwb to read the NWB files directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `nwb.subject.subject_id` from each NWB file. The code maintains an `OrderedDict` (`subjects_set`) to accumulate unique subject IDs across all sessions, assigning sequential indices. After processing all sessions, it builds a `subject_idx` array mapping each session to its subject.

ii.
```python
subjects_set = OrderedDict()
# In the loop:
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
# After all sessions:
subjects = list(subjects_set.keys())
subject_idx = np.array([subjects_set[sess['subject_id']] for sess in all_sessions])
```

iii. Subjects correspond to individual mice. The agent found 28 subjects across the NWB files, matching the paper's "28 mice."

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions that fail selection criteria are excluded: (1) overall behavioral performance > 65%, (2) at least 50 correct lick-left trials, (3) at least 50 correct lick-right trials. Sessions with no good neurons (or no neurons mapping to valid brain regions) are also skipped. Out of 174 NWB files, 144 passed these criteria.

ii.
```python
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50
MIN_PERFORMANCE = 0.65

# In process_session():
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_LEFT:
    return None
if correct_right < MIN_CORRECT_RIGHT:
    return None
```

iii. The agent cited the paper's session selection criteria: "Overall behavioral performance > 65%" and "at least 50 correct lick left and lick right trials each." The performance metric is computed on control trials (no photostimulation, no auto water, no free water) that are also non-early-lick and non-ignore.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. Each row in the trials table is one trial. Trials are filtered: `auto_water == 1` and `free_water == 1` trials are excluded. The remaining trials are indexed and their go cue times are used for temporal alignment. The code also has a (possibly non-functional) check to exclude trials beyond the neural recording range.

ii.
```python
# In process_session():
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. The agent distinguished between trial filtering for the reference analysis (which excluded early lick, ignore, stim, auto water, free water) and the decoder task (which keeps early lick, ignore, and stim trials since they are decoder inputs/outputs). Only auto_water and free_water are excluded as they alter the task structure.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level quality filtering involves: (1) removing auto_water and free_water trials, (2) excluding trials beyond the neural recording time range (checking spike time bounds). However, the recording range filter appears not to have been applied in the final conversion run (no "Excluding" messages in conversion output, and 1,056 trials have all-zero neural data).

ii.
```python
# Trial filtering:
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

# Recording range filter (added later, may not have been applied):
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
```

iii. The agent documented fixing the "trials beyond recording range" issue in Step 10 (Critical Review 1), but the conversion output file shows no evidence this filter was applied during the final conversion run, resulting in 1,056 all-zero neural data trials in the verification output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times of "good" units. The code reads `units['spike_times']` (via VectorIndex/VectorData) and `units['classification']` from the NWB file. Only units with `classification == 'good'` are kept.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'

spike_times_vi = units['spike_times']  # VectorIndex
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])

good_indices = np.where(good_mask)[0]
good_spike_times = []
for ui in good_indices:
    start_idx = 0 if ui == 0 else int(all_st_idx[ui - 1])
    end_idx = int(all_st_idx[ui])
    good_spike_times.append(all_spike_times[start_idx:end_idx])
```

iii. The agent identified that `classification == 'good'` in NWB corresponds to the classifier-based QC from the reference code (logistic regression classifiers per brain area group). This is documented in the spike sorting QC paper.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into non-overlapping 50ms bins to compute firing rates (spikes/second). For each trial, spikes are aligned to the go cue and binned from -2.5s to +1.5s (80 bins). The firing rate is computed as spike count / bin_width.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
ALIGN_START = -2.5
ALIGN_END = 1.5
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins

def bin_spikes(spike_times_list, go_times, ...):
    bin_edges = np.linspace(align_start, align_end, n_bins + 1)
    for n in range(n_neurons):
        for t in range(n_trials):
            go_t = go_times[t]
            rel_spikes = st[idx_lo:idx_hi] - go_t
            counts, _ = np.histogram(rel_spikes, bins=bin_edges)
            all_matrices[t, n, :] = counts / bin_width  # firing rate
```

iii. The agent noted that the reference code used 40ms bins with 3.4ms stride (sliding histogram), but the decoder task instructions specify 50ms bins. The agent chose to follow the instructions (50ms non-overlapping bins) rather than the reference code's sliding histogram approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: (1) only units with `classification == 'good'` are kept (the classifier-based QC), and (2) only good units whose `anno_name` maps to a valid brain region are kept (unmapped annotation names are excluded).

ii.
```python
# Stage 1: QC classification
good_mask = classifications == 'good'
good_spike_times = [spike_times for good units]

# Stage 2: Brain region mapping
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
```

iii. The agent followed the reference approach of using the classifier-based QC (25.9% pass rate mentioned in the paper). Neurons without a valid brain region mapping are also excluded, which is a secondary filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time from `BehavioralEvents.go_start_times` is used as the reference. Spikes are extracted from -2.5s to +1.5s relative to this go cue.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
# Then in bin_spikes:
go_t = go_times[t]
abs_start = go_t + align_start   # go_t - 2.5
abs_end = go_t + align_end       # go_t + 1.5
rel_spikes = st[idx_lo:idx_hi] - go_t
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue." The agent followed these instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (non-overlapping bins). This differs from the reference code which uses a sliding histogram with 40ms bin width and 3.4ms stride. No rebinning is applied after the initial binning from spike times. 80 time bins cover the -2.5s to +1.5s window.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

iii. The agent explicitly chose 50ms bins per the decoder task specification, overriding the reference code's 40ms/3.4ms sliding histogram. This is documented in CONVERSION_NOTES: "Decoder task specifies 50ms bins; use 50ms."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Time from tone onset is derived from `BehavioralEvents.sample_start_times.timestamps` and `BehavioralEvents.go_start_times.timestamps`. The sample start time (tone onset) is identified per trial, then the time from tone onset is computed for each time bin.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent identified `sample_start_times` as the tone onset event and noted that the sample epoch may replay due to early licks, so the last sample start before the go cue is used.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start_times` event within the trial's time window (between trial start and go cue) is found. Then for each time bin, the time from tone onset is computed as `bin_center_time - tone_onset_time` (both in absolute session time, so effectively `relative_bin_center - relative_tone_onset`).

ii.
```python
def get_tone_onset_for_trials(go_times, sample_start_ts, trial_starts, trial_stops):
    for i in range(n_trials):
        mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
        matching = sample_start_ts[mask]
        if len(matching) > 0:
            tone_onsets[i] = matching[-1]  # last sample start
    return tone_onsets

def compute_tone_onset_input(go_times, tone_onsets, ...):
    for t in range(n_trials):
        tone_rel = tone_t - go_t  # tone onset relative to go cue (negative)
        tone_input = (bin_centers - tone_rel).astype(np.float32)
    return tone_onset_input
```

iii. The agent chose the last sample start to handle early lick replays where the sample period restarts. The computation produces a continuous time-varying signal representing seconds elapsed since tone onset at each time bin.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone onset input uses the same bin centers as the neural data (50ms bins from -2.5s to +1.5s relative to go cue). The bin centers are computed identically.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
# Same bin_centers used for neural binning and tone onset input
```

iii. Using the same temporal grid ensures alignment between neural data and inputs.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `BehavioralEvents.photostim_start_times.timestamps` and `BehavioralEvents.photostim_stop_times.timestamps`.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The agent read photostim start/stop events from the NWB behavioral events.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code iterates over all photostim events globally (not per-trial). For each event, it checks if the event overlaps with the trial window. If a bin center falls within a photostim start-stop interval (relative to go cue), that bin is set to 1.0, otherwise 0.0. This produces a binary time-varying signal.

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts, ...):
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)
        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            if ps_stop < align_start or ps_start > align_end:
                continue
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
        photostim_trials.append(ps)
```

iii. The agent noted photostim was ~25% of trials, active during late delay (last 0.5s), 40Hz sinusoidal at 5mW.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation uses the same bin centers as neural data (50ms bins from -2.5s to +1.5s). The absolute photostim times are converted to times relative to go cue, then compared to bin center positions.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

iii. Same temporal grid as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Lick direction choice is derived from `trials.trial_instruction` in the NWB trials table.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The agent mapped 'left' to 0 and 'right' to 1, following the instruction specification.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The `trial_instruction` string is converted to a binary value (left=0, right=1). This is a per-trial value that is broadcast to all time bins (all 80 bins get the same value for each trial).

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
# In output construction:
np.full(N_BINS, choices[t], dtype=np.int64)  # broadcast to all bins
```

iii. The agent used `trial_instruction` rather than actual lick direction, which represents the instructed direction, not necessarily the animal's choice (in miss or ignore trials, the animal may lick the wrong direction or not at all).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `trials.outcome` in the NWB trials table.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. The agent mapped outcome strings to integers: ignore=0, miss=1, hit=2, matching the instruction specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome string from the trials table is mapped to integer categories (ignore=0, miss=1, hit=2) and broadcast to all time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
np.full(N_BINS, outcomes[t], dtype=np.int64)
```

iii. The agent treated outcome as per-trial and broadcast it across all time bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `trials.early_lick` in the NWB trials table.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

iii. The agent checked for 'early' vs 'no early' in the early_lick field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The early_lick string is converted to binary (no=0, yes=1). 'early' maps to 1, everything else (including 'no early') maps to 0. It is per-trial and broadcast to all time bins.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
np.full(N_BINS, early_licks[t], dtype=np.int64)
```

iii. Per-trial value broadcast to all time bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries.Camera0_side_TongueTracking` which contains (x, y, confidence) per frame at ~294 Hz.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]    # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. The agent identified the tongue tracking data from the side camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The tongue y-position (column 1 of tracking data) is extracted. When confidence (column 2) is below 0.9, the position is replaced with the session mean of visible tongue positions. The data is then binned into 50ms bins by averaging values within each bin. Empty bins are filled with the session mean.

ii.
```python
def compute_tongue_y_per_trial(tongue_data, tongue_timestamps, go_times, ...):
    tongue_y = tongue_data[:, 1].astype(np.float64)
    tongue_conf = tongue_data[:, 2].astype(np.float64)
    visible_mask = tongue_conf >= confidence_threshold  # 0.9
    session_mean_y = np.mean(tongue_y[visible_mask])
    tongue_y_imputed = tongue_y.copy()
    tongue_y_imputed[~visible_mask] = session_mean_y

    for t in range(n_trials):
        # Bin into 50ms bins by averaging
        bin_indices = np.digitize(trial_ts, bin_edges) - 1
        for b in range(n_bins):
            in_bin = trial_y[bin_indices == b]
            if len(in_bin) > 0:
                trial_tongue_y[b] = np.mean(in_bin)
```

iii. The agent cited the paper: "when the tongue was occluded... we set the tongue position to its mean value." The confidence threshold of 0.9 is used for occlusion detection.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Tongue y-position is discretized per session using percentiles computed over all time bins across all trials. Values below the 40th percentile are category 0, between 40th and 60th are category 1, and above 60th are category 2.

ii.
```python
def discretize_tongue_y(tongue_y_trials, session_mean_y):
    all_values = np.concatenate([t for t in tongue_y_trials])
    p40 = np.percentile(all_values, 40)
    p60 = np.percentile(all_values, 60)
    if np.isclose(p40, p60):
        eps = max(1e-6, abs(p40) * 1e-4)
        p40 = p40 - eps
        p60 = p60 + eps
    for trial_y in tongue_y_trials:
        d = np.zeros(len(trial_y), dtype=np.int64)
        d[trial_y >= p40] = 1
        d[trial_y >= p60] = 2
        discretized.append(d)
```

iii. The agent followed the instructions: "0: < 40th percentile, 1: 40th to 60th, 2: > 60th percentile of y-position over the session." The agent handled edge cases where p40==p60 by adding a small epsilon offset.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position uses the same bin edges as the neural data (-2.5s to +1.5s relative to go cue, 50ms bins). Tongue timestamps are converted to time relative to go cue and then binned using the same edges.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
trial_ts = tongue_timestamps[mask] - go_t  # relative to go cue
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. Same temporal grid ensures alignment with neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Missing tone onset (NaN) results in zero time-from-tone input. (2) Missing tongue data in a trial fills bins with the session mean. (3) Low-confidence tongue tracking values are replaced with session mean. (4) Empty spike times for a neuron result in zero firing rate. (5) When percentile thresholds p40==p60 for tongue discretization, a small epsilon is used. (6) The code attempts to exclude trials beyond the recording time range, but this filter was not applied in the final conversion (resulting in 1,056 all-zero neural data trials).

ii.
```python
# Missing tone onset:
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)

# Missing tongue data:
if len(trial_ts) == 0:
    tongue_y_trials.append(np.full(n_bins, session_mean_y, dtype=np.float32))

# Empty spike times:
if len(st) == 0:
    continue  # leaves zeros in the matrix

# Percentile edge case:
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
```

iii. The agent documented these edge cases in CONVERSION_NOTES under Step 10. The recording range filter was identified and coded but appears not to have been applied in the final data.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the conversion output timing, spike binning is the most time-consuming step (~1.7-5.5s per session), followed by tongue processing (~1.2-2.7s per session), and NWB loading (~1.5-2.3s per session). Total conversion time was 19.5 minutes for 144 sessions.

ii.
```python
# Timing output from conversion:
# Spike binning: 1.7-5.5s per session
# Tongue processing: 1.2-2.7s per session
# NWB loading: ~1.5-2.3s per session
```

iii. The agent documented timing in CONVERSION_NOTES Step 7 and noted optimization of tongue processing from 142s to 1.5s via vectorization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function has a double loop over neurons and trials. While the inner loop uses `np.searchsorted` and `np.histogram` (partially vectorized), the outer neuron-trial loop could potentially be vectorized further. The `compute_photostim_input` function has a triple-nested loop (trials x photostim events x bins) that could be vectorized. The `compute_tongue_y_per_trial` function has a per-bin loop inside a per-trial loop.

ii.
```python
# bin_spikes: neuron x trial loop
for n in range(n_neurons):
    for t in range(n_trials):
        # ... per-neuron, per-trial binning

# compute_photostim_input: trial x event x bin loop
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        for b in range(n_bins):

# compute_tongue_y_per_trial: trial x bin loop
for t in range(n_trials):
    for b in range(n_bins):
        in_bin = trial_y[bin_indices == b]
```

iii. The agent noted optimizations in CONVERSION_NOTES Step 6 (vectorized tongue processing using np.digitize, used searchsorted for spike binning), but some inner loops remain.

## 10-c. What processing does the code repeat multiple times?

i. The code computes `bin_edges` and `bin_centers` multiple times across different functions (`bin_spikes`, `compute_tongue_y_per_trial`, `compute_photostim_input`, `compute_tone_onset_input`). These are identical computations using the same parameters. The `map_anno_to_region` function is called per-neuron with string matching, repeating the keyword search pattern.

ii.
```python
# Repeated in multiple functions:
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
```

iii. These are minor inefficiencies; the computation is fast and the main bottleneck is the spike binning.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_mean_y` for tongue tracking for every session, even sessions without photostimulation or tongue data issues. The code reads and processes lick times (`left_lick_ts`, `right_lick_ts`) from the NWB file but never uses them in the conversion. The brain region mapping uses a complex keyword-matching approach when simpler exact matching or a lookup table could suffice. The code also loads all trial fields (including `photostim_power`, `photostim_duration`) that are not used in the final output.

ii.
```python
# Loaded but unused:
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
# Also loaded but unused:
'photostim_power': trials['photostim_power'][:],
'photostim_duration': trials['photostim_duration'][:],
```

iii. The agent loaded these variables for potential use but ultimately did not include them in the final output structure.
