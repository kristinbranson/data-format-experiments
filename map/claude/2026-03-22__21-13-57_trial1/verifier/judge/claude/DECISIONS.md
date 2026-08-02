# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files stored in `/app/data/`. It uses `pynwb.NWBHDF5IO` to open each `.nwb` file and extracts units, trials, behavioral events, and behavioral time series. All NWB files are discovered via `glob.glob` matching the pattern `sub-*/sub-*_ses-*.nwb`, sorted alphabetically. Each file is processed sequentially in the `convert_all` function.

ii.
```python
def get_nwb_files():
    """Get list of all NWB files sorted by subject then session."""
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))

def load_nwb_session(nwb_path):
    """Load all needed data from a single NWB file."""
    import pynwb
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    # ... extracts units, trials, behavioral events, tongue tracking ...
    io.close()
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code loads from `.mat` files exported from DataJoint, but the same underlying data is available in NWB format on DANDI. It chose to load from NWB since that is the format provided.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `nwb.subject.subject_id` extracted from each NWB file. An `OrderedDict` accumulates unique subject IDs as sessions are processed, preserving encounter order. The final `subjects` list and `subject_idx` array map each session to its subject.

ii.
```python
subjects_set = OrderedDict()
# ... in processing loop:
sid = result['subject_id']
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
# ... later:
subjects = list(subjects_set.keys())
subject_idx = np.array([subjects_set[sess['subject_id']] for sess in all_sessions], dtype=np.int64)
```

iii. The AI verified 28 unique subjects matching the paper's reported "28 mice."

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes all 174 NWB files found, but filters sessions based on selection criteria: overall behavioral performance > 65%, at least 50 correct lick-left and 50 correct lick-right trials. Sessions failing these criteria are skipped. 144 sessions pass out of 174 (the paper reports 173 sessions with good units; the difference from 174 is explained by one session having no good units).

ii.
```python
performance, correct_left, correct_right = compute_session_performance(data['trials_data'])
if performance < MIN_PERFORMANCE:  # 0.65
    return None
if correct_left < MIN_CORRECT_LEFT:  # 50
    return None
if correct_right < MIN_CORRECT_RIGHT:  # 50
    return None
```

iii. The AI documented that session selection criteria come from the methods text: "overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."

## 1-d. How are the data split into trials?

i. Trials are extracted from the NWB trials table. Each trial has fields including `start_time`, `stop_time`, `trial_instruction`, `outcome`, `early_lick`, `auto_water`, `free_water`, `photostim_onset`, etc. The go cue times are obtained from `BehavioralEvents/go_start_times`.

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
    # ...
}
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI noted that in the NWB format, trials are stored in a structured trials table with associated behavioral event timestamps.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes two types of trials: `auto_water == 1` and `free_water == 1`. Additionally, trials whose time window falls outside the neural recording range are excluded (trials beyond recording start/end). Early lick, ignore, and photostimulation trials are kept since they serve as decoder inputs/outputs.

ii.
```python
trial_mask = np.ones(data['n_trials'], dtype=bool)
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False

# Later: exclude trials beyond recording range
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
```

iii. The AI justified this decision by noting that auto_water and free_water are artificial conditions that change task structure, while early lick, ignore, and stim trials are meaningful decoder outputs/inputs. The reference code's `get_regular_trial_mask` excludes more trial types (early lick, ignore, stim), but the decoder task instructions specify these as decoder I/O.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units.spike_times` in the NWB file, filtered to only include units where `units.classification == 'good'`. The spike times are absolute session timestamps.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
spike_times_vi = units['spike_times']
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])

for ui in good_indices:
    start_idx = 0 if ui == 0 else int(all_st_idx[ui - 1])
    end_idx = int(all_st_idx[ui])
    good_spike_times.append(all_spike_times[start_idx:end_idx])
```

iii. The AI noted that in NWB files, `classification == 'good'` corresponds to the classifier-based QC described in the paper and implemented by the reference code's QC system.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins to compute firing rates (spikes/second). For each neuron and trial, spikes falling within the time window [-2.5s, +1.5s] relative to the go cue are extracted, histogrammed into 80 bins, and divided by the bin width (0.05s) to get firing rates in Hz.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)  # 81 edges for 80 bins
# For each neuron and trial:
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
all_matrices[t, n, :] = counts / bin_width  # firing rate in Hz
```

iii. The AI chose 50ms non-overlapping bins per the decoder task specification ("Use 50-ms-width bins for computing firing rates"), overriding the reference code's 40ms/3.4ms sliding histogram parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Only units with `classification == 'good'` are included (classifier-based QC from the paper), and (2) only neurons with a valid brain region mapping (non-null `anno_name` that maps to one of 14 major regions) are kept.

ii.
```python
good_mask = classifications == 'good'
# ... later in process_session:
for i, anno in enumerate(good_anno):
    region = map_anno_to_region(anno)
    if region is not None:
        region_labels.append(region)
        neuron_mask.append(i)
```

iii. The AI documented that `classification == 'good'` in NWB implements the same classifier-based QC from the paper (5 region-specific logistic regression classifiers, yielding 25.9% pass rate). Neurons with unmapped annotations are excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time is obtained from `go_start_times`, and spikes are extracted in a window from -2.5s to +1.5s relative to the go cue.

ii.
```python
ALIGN_START = -2.5  # seconds before go cue
ALIGN_END = 1.5     # seconds after go cue
# In bin_spikes:
go_t = go_times[t]
abs_start = go_t + align_start
abs_end = go_t + align_end
idx_lo = np.searchsorted(st, abs_start, side='left')
idx_hi = np.searchsorted(st, abs_end, side='left')
rel_spikes = st[idx_lo:idx_hi] - go_t
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue." The reference code also aligns to the go cue (spike times in .mat files are already relative to go cue time).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (non-overlapping bins), producing 80 time bins for the 4-second window. No rebinning is applied; spikes are directly histogrammed into 50ms bins.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins (decoder task spec)
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
```

iii. The decoder task specification requires "50-ms-width bins." The reference code uses 40ms bins with 3.4ms stride (overlapping sliding histogram), but the AI correctly prioritized the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (timestamps of tone/sample epoch onsets) and `go_start_times` (go cue times) from the NWB `BehavioralEvents` acquisition.

ii.
```python
sample_start_ts = be.time_series['sample_start_times'].timestamps[:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI noted that sample_start_times correspond to tone onset events, and that these may have multiple entries per trial due to early lick replays.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start` event occurring between trial start and go cue is identified as the tone onset. Then for each time bin, the input value is computed as the time elapsed since that tone onset: `bin_center_time - tone_onset_relative_to_go`.

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
    tone_rel = tone_t - go_t  # tone onset relative to go cue (negative)
    tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. The AI takes the last sample start to account for early lick replays where the sample epoch is repeated. The continuous time-from-onset representation matches the instructions.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input uses the same bin centers as the neural data (centered in each 50ms bin), ensuring temporal alignment. Both are relative to the go cue.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
# Same bin_centers used for both neural (bin_edges) and input computation
```

iii. Using the same temporal grid ensures alignment between neural activity and input variables.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_start_times` and `photostim_stop_times` from the NWB `BehavioralEvents` acquisition.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The AI identified these as the relevant NWB fields for photostimulation timing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial and time bin, the AI checks whether any photostimulation event overlaps with that bin center. If the bin center falls within a [photostim_start, photostim_stop) interval, the value is set to 1.0; otherwise 0.0.

ii.
```python
def compute_photostim_input(go_times, photostim_start_ts, photostim_stop_ts, ...):
    for t in range(n_trials):
        go_t = go_times[t]
        ps = np.zeros(n_bins, dtype=np.float32)
        for si in range(len(photostim_start_ts)):
            ps_start = photostim_start_ts[si] - go_t
            ps_stop = photostim_stop_ts[si] - go_t
            for b in range(n_bins):
                bc = bin_centers[b]
                if bc >= ps_start and bc < ps_stop:
                    ps[b] = 1.0
        photostim_trials.append(ps)
```

iii. The AI treats photostimulation as a binary time-varying input per the instructions: "Whether photostimulation is on at every time point (discrete, time-varying)."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Uses the same bin centers as the neural data, with photostim times converted to relative-to-go-cue coordinates.

ii.
```python
ps_start = photostim_start_ts[si] - go_t  # relative to go cue
ps_stop = photostim_stop_ts[si] - go_t
```

iii. Same temporal reference frame (go cue) and bin structure as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from the `trial_instruction` column of the NWB trials table, which contains 'left' or 'right'.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The instructions specify "Lick direction choice (left = 0, right = 1, per-trial)."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping: 'left' → 0, 'right' → 1. The value is per-trial and broadcast to all time bins.

ii.
```python
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
# In output construction:
np.full(N_BINS, choices[t], dtype=np.int64),  # choice (per-trial, broadcast)
```

iii. This is a straightforward categorical encoding as specified in the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from the `outcome` column of the NWB trials table, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
```

iii. The instructions specify "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple mapping: 'ignore' → 0, 'miss' → 1, 'hit' → 2. Per-trial, broadcast to all time bins. Unknown outcomes default to 0 (ignore).

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw], dtype=np.int64)
# In output:
np.full(N_BINS, outcomes[t], dtype=np.int64),  # outcome (per-trial, broadcast)
```

iii. Straightforward categorical encoding per the instructions.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question refers to a template variable that does not apply to this dataset. The AI's task does not include a "Distance to reward zone" output. The Outcome output is per-trial (not time-varying) and is broadcast to all time bins, using the same temporal grid as the neural data.

ii. N/A — no "Distance to reward zone" variable exists in this dataset.

iii. N/A

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from the `early_lick` column of the NWB trials table, which contains 'early' or 'no early'.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
```

iii. The instructions specify "Early lick (no = 0, yes = 1, per-trial)."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Simple mapping: 'early' → 1, anything else → 0. Per-trial, broadcast to all time bins.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw], dtype=np.int64)
np.full(N_BINS, early_licks[t], dtype=np.int64),  # early lick (per-trial, broadcast)
```

iii. Binary encoding as specified in the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` in the NWB `BehavioralTimeSeries` acquisition, specifically the y-coordinate (column index 1) and confidence score (column index 2) of the tongue tracking data.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]  # (n_frames, 3): x, y, confidence
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. The AI identified the tongue tracking time series from the NWB file, recorded at ~294 Hz from a side-view camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing steps: (1) Extract y-position and confidence from tongue tracking data. (2) Compute session mean of y-position when tongue is visible (confidence >= 0.9). (3) Replace occluded frames (confidence < 0.9) with session mean. (4) For each trial, extract tongue data in the [-2.5s, +1.5s] window around go cue. (5) Bin tongue y-values into 50ms bins by averaging within each bin. (6) Empty bins are filled with session mean.

ii.
```python
visible_mask = tongue_conf >= confidence_threshold  # 0.9
session_mean_y = np.mean(tongue_y[visible_mask])
tongue_y_imputed = tongue_y.copy()
tongue_y_imputed[~visible_mask] = session_mean_y

# Per trial binning:
bin_indices = np.digitize(trial_ts, bin_edges) - 1
for b in range(n_bins):
    in_bin = trial_y[bin_indices == b]
    if len(in_bin) > 0:
        trial_tongue_y[b] = np.mean(in_bin)
```

iii. The AI followed the reference paper's approach: "when the tongue was occluded... we set the tongue position to its mean value."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles are computed over all tongue y values across all trials and time bins. Three categories: 0 (< 40th percentile), 1 (40th to 60th percentile), 2 (> 60th percentile). When p40 == p60 (common due to imputed values), a small epsilon offset is applied to ensure three distinct categories.

ii.
```python
all_values = np.concatenate([t for t in tongue_y_trials])
p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)

if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
    p40 = p40 - eps
    p60 = p60 + eps

d = np.zeros(len(trial_y), dtype=np.int64)
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```

iii. The instructions specify: "0: < 40th percentile of y-position over the session, 1: 40th to 60th percentile, 2: > 60th percentile."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are converted to go-cue-relative time, then binned into the same 50ms bins used for neural data. The binning uses `np.digitize` on the relative timestamps.

ii.
```python
trial_ts = tongue_timestamps[mask] - go_t  # relative to go cue
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. Same temporal reference (go cue) and bin structure as neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Missing tone onset**: If no sample_start is found for a trial, tone_input is set to all zeros.
- **Trials beyond recording range**: Trials whose go-cue-aligned window extends beyond the neural recording are excluded.
- **Occluded tongue**: Frames with confidence < 0.9 are replaced with session mean y-position.
- **Empty tongue bins**: Bins with no tongue tracking frames are filled with session mean.
- **Degenerate percentiles**: When p40 == p60 (due to imputed values dominating), a small epsilon offset ensures 3 distinct categories.
- **Neurons with no spikes**: Handled gracefully (result in zero firing rates).
- **Unknown outcome values**: Default to 0 (ignore) via `.get(o, 0)`.

ii.
```python
# Missing tone onset:
if np.isnan(tone_t):
    tone_input = np.zeros(n_bins, dtype=np.float32)

# Trials beyond recording:
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)

# Degenerate percentiles:
if np.isclose(p40, p60):
    eps = max(1e-6, abs(p40) * 1e-4)
```

iii. The AI documented in CONVERSION_NOTES.md that ~1,056 trials across 7 sessions were dropped due to being beyond the recording range. The tongue occlusion handling follows the reference paper.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing information in CONVERSION_NOTES.md:
- **Spike binning**: ~3.3s per session (the nested loop over neurons × trials is the primary bottleneck)
- **NWB file loading**: ~1.5s per session (I/O bound)
- **Tongue processing**: ~1.5s per session (after optimization)
- Total estimated: ~22 minutes for 174 sessions

ii.
```python
# Spike binning: nested loop over neurons and trials
for n in range(n_neurons):
    for t in range(n_trials):
        # searchsorted + histogram per neuron per trial
```

iii. The AI documented timing in CONVERSION_NOTES.md Step 7 and optimized tongue processing from 142s to 1.5s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain that could be further vectorized:
- **Spike binning**: The inner loop over trials (`for t in range(n_trials)`) could be vectorized by computing all trial offsets at once.
- **Photostim computation**: Triple-nested loop (trials × events × bins) could be vectorized using broadcasting.
- **Tongue y binning**: The `for b in range(n_bins)` loop per trial could use `np.bincount` with weights.
- **Tone onset computation**: The per-trial loop searching for matching sample_start_times could use vectorized searchsorted.

ii.
```python
# Photostim: triple nested loop
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        for b in range(n_bins):
            if bc >= ps_start and bc < ps_stop:
                ps[b] = 1.0
```

iii. The AI noted optimizations made (vectorized tongue processing, searchsorted for spike binning) but further vectorization opportunities remain.

## 10-c. What processing does the code repeat multiple times?

i. The code does not obviously repeat major processing steps. Each session is processed once. However, some minor repeated work exists:
- `bin_centers` and `bin_edges` are recomputed in multiple functions (`bin_spikes`, `compute_tongue_y_per_trial`, `compute_photostim_input`, `compute_tone_onset_input`) rather than being computed once and passed through.
- The brain region mapping lookup (`map_anno_to_region`) iterates through all keywords for every neuron.

ii.
```python
# bin_edges/bin_centers computed in multiple places:
bin_edges = np.linspace(align_start, align_end, n_bins + 1)  # in bin_spikes
bin_edges = np.linspace(align_start, align_end, n_bins + 1)  # in compute_tongue_y_per_trial
bin_centers = np.linspace(...)  # in compute_photostim_input
bin_centers = np.linspace(...)  # in compute_tone_onset_input
```

iii. These are minor inefficiencies; the actual computation cost is negligible compared to spike binning and I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of data are loaded but not used in the final output:
- **Left and right lick times** (`left_lick_times`, `right_lick_times`) are loaded but never used.
- **Subject description** (`subject_desc`) is loaded and stored only in metadata session_info, not critical for decoding.
- **Tongue x-position** is loaded as part of the (x, y, confidence) data but only y is used.
- **Various trial fields** (e.g., `photostim_power`, `photostim_duration`) are loaded but not used beyond session performance computation.
- **Plotting code** is included but only runs with `--show-processing` flag.

ii.
```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]   # loaded but unused
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]  # loaded but unused
tongue_data = tongue_ts_obj.data[:]  # all 3 columns loaded, only y used
```

iii. Loading unused data adds I/O overhead. The lick times could have been skipped entirely since lick direction is captured via `trial_instruction`.
