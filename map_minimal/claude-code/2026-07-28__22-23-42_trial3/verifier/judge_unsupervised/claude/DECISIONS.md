# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `glob.glob('data/sub-*/sub-*.nwb')` to discover all 174 NWB files across 28 subject directories. Each NWB file is one session. Files are opened with `h5py` (not `pynwb`) and read directly from HDF5 paths. Trial metadata, spike times, behavioral events, and tongue tracking are extracted from standardized NWB paths. Sessions are processed sequentially in sorted filename order.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
f = h5py.File(nwb_path, 'r')
n_trials = len(f['intervals/trials/id'][:])
outcomes = np.array([o.decode() if isinstance(o, bytes) else o for o in f['intervals/trials/outcome'][:]])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
```

iii. The AI documented in CONVERSION_NOTES.md: "174 NWB files across 28 subjects in `data/sub-*/sub-*.nwb`". It chose h5py over pynwb for performance. The trajectory shows it systematically explored the NWB file structure to identify the correct HDF5 paths.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from each NWB filename by parsing the `sub-XXXXXX` prefix. The unique subjects are collected into a sorted list, and each session is assigned an index into this list via `subject_idx`.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
# ...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The AI's CONVERSION_NOTES mentions 28 subjects. The verification output confirms 28 subjects with varying numbers of sessions per subject (e.g., Subject 440956: 3 sessions, Subject 480927: 10 sessions).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one recording session. Sessions that pass quality filters are included in the final dataset. The AI applies session-level filtering: performance > 65% on control trials, and at least 50 correct left + 50 correct right trials. Sessions with no good units with valid brain regions are also excluded. Result: 144 of 174 sessions kept.

ii.
```python
if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None
```

iii. The AI's CONVERSION_NOTES states: "144 sessions kept, 30 skipped" with reasons including low performance (20), insufficient correct trials per direction (7), no good units (1), and other (2). This matches the methods text criteria.

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. The total trial count per session is determined by the length of `intervals/trials/id`. Each trial is associated with a go cue time from `acquisition/BehavioralEvents/go_start_times/timestamps`. Trials are filtered based on quality criteria (see 1-e).

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials
```

iii. The AI verified that go cue counts match trial counts. The final dataset contains 74,759 trials across 144 sessions, with a mean of ~519 trials per session.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three trial-level filters: (1) exclude `auto_water` trials, (2) exclude `free_water` trials, (3) exclude trials where the neural recording window doesn't cover the full analysis time window around the go cue. Notably, early lick trials, ignore/no-response trials, and photostimulation trials are all retained because they are needed as decoder outputs/inputs.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
# ...
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
```

iii. The AI noted in CONVERSION_NOTES: "Excluded auto_water and free_water trials (as in reference code). Kept early lick trials (needed for early_lick decoder output). Kept ignore/no-response trials (needed for outcome decoder output). Kept photostimulation trials (needed for photostim decoder input)." The recording window filter was added after discovering that some NWB files contain trials beyond the neural recording period.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (raw spike times) indexed by `units/spike_times_index` (ragged array index). Units are filtered by `units/classification` (quality control) and `units/anno_name` (brain region mapping).

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c for c in f['units/classification'][:]])
anno_names = np.array([a.decode() if isinstance(a, bytes) else a for a in f['units/anno_name'][:]])
```

iii. The AI documented that spike times are loaded from the NWB units table, using the ragged array format with index arrays to extract per-unit spike times.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to firing rates using non-overlapping 50ms bins. For each unit and trial, spike times are aligned to the go cue, histogrammed into bins from -2.5s to +1.5s (80 bins), and divided by bin width (0.05s) to get rates in Hz. The result is a (n_neurons, 80) matrix per trial.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

iii. The AI noted: "50ms non-overlapping bins (as specified in decoder task). Spike counts divided by bin width (0.05s) to get rates in Hz. 80 time bins per trial." It acknowledged this differs from the reference code's 100ms bandwidth with 50ms stride (Gaussian-weighted) but follows the decoder task specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied to units: (1) classifier-based QC requiring `classification == 'good'`, and (2) valid brain region annotation (anno_name must map to one of the 14 predefined region categories). Units that fail either filter are excluded. Additionally, the recording observation window is checked to ensure all selected units have data for the included trials.

ii.
```python
good_mask = classification == 'good'
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

iii. The AI documented: "Classifier-based QC (classification='good' in NWB units table). The reference code uses qc_mode='classifier'." and "Units with empty or unmappable annotation names excluded (490 units, <1%)." Final count: 57,925 good units across 144 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset (t=0). Spike times are subtracted by the go cue time to get relative times, then histogrammed into bins from -2.5s to +1.5s relative to the go cue.

ii.
```python
rel_times = spike_times - go_cue_time
# ...
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)  # T_START=-2.5, T_END=1.5
```

iii. The AI's CONVERSION_NOTES state: "Aligned to go cue onset (t=0). Window: -2.5s to +1.5s (as specified in decoder task)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s) bins. There are 80 bins spanning -2.5s to +1.5s. No rebinning is applied; firing rates are computed directly from spike times into these bins using `np.histogram`.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. The AI noted this follows the decoder task specification ("Use 50-ms-width bins for computing firing rates") and differs from the reference code's 100ms bandwidth with 50ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone onset times) and `acquisition/BehavioralEvents/go_start_times/timestamps` (go cue times). The last sample start time before the go cue is used as the tone onset for each trial.

ii.
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
# ...
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time
    return None
```

iii. The AI reasoned in the trajectory: "The sample epoch has 3 tones (150ms each, 100ms ITI) = 650ms. Then delay epoch = 1.2s. So tone onset to go cue = 1.85s. But early lick trials can have replayed sample epochs, so the last sample_start before go is the correct tone onset."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start_time` before the go cue is found and converted to a time relative to the go cue (a negative value, typically around -1.85s). Then for each time bin, the time from tone onset is computed as `BIN_CENTERS - tone_onset_rel`, giving a continuous time-varying signal representing elapsed time since tone onset. If no sample_start is found before the go cue, a fallback value of -1.85s is used.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel  # time since tone onset at each bin
```

iii. The AI documented: "Computed as seconds since the first tone of the (successful) sample epoch. For trials with early lick replays, uses the last sample_start before the go cue. Typical value at go cue: ~1.85s."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Time from tone onset is computed at the same bin centers as the neural data, ensuring temporal alignment. Both use `BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2`, which are the midpoints of the 80 time bins from -2.5s to +1.5s relative to the go cue.

ii.
```python
time_from_tone = BIN_CENTERS - tone_onset_rel  # shape: (80,), aligned with neural bins
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)  # (2, 80)
```

iii. The alignment is implicit through the shared use of `BIN_CENTERS` for both neural and input computations.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from three NWB fields: `intervals/trials/photostim_onset` (onset time relative to trial start, or 'N/A'), `intervals/trials/photostim_duration` (duration of stimulation), and `intervals/trials/start_time` (trial start time, needed to convert onset to absolute time).

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. The AI discovered during development that `photostim_onset` is relative to trial start (not absolute), which required also loading `trial_start_times`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial with photostim (onset != 'N/A'): (1) convert onset from trial-relative to absolute time by adding trial start, (2) convert to go-cue-relative by subtracting go cue time, (3) compute end time as onset + duration, (4) create a binary array marking which time bins have their center within the photostim window. For non-photostim trials, the array is all zeros.

ii.
```python
if ps_onset_str[ti] != 'N/A':
    ps_onset_val = float(ps_onset_str[ti])
    ps_dur = float(ps_dur_str[ti])
    ps_abs_onset = trial_start_times[ti] + ps_onset_val
    ps_start_rel = ps_abs_onset - gc
    ps_end_rel = ps_start_rel + ps_dur
    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The AI initially treated photostim_onset as absolute time, producing nonsensical values (~-132s relative to go cue). After investigation, the agent confirmed `trial_start + photostim_onset = absolute onset` and verified `rel_to_go ~ -1.2s`, matching the methods description of late delay epoch silencing.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim binary signal is computed at the same `BIN_CENTERS` as the neural data, ensuring temporal alignment. The result is a binary (0/1) array of shape (80,), with 1s marking bins where photostim is active. Photostim typically falls around -1.2s to -0.6s relative to go cue.

ii.
```python
photostim_binary = ((BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. Aligned through shared `BIN_CENTERS` array, consistent with neural data binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/trial_instruction` which encodes the instructed lick direction ('left' or 'right') for each trial. This is the direction the animal was supposed to lick, not necessarily the actual lick direction.

ii.
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
# ...
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. The AI's CONVERSION_NOTES state: "Based on trial_instruction field (which port the animal should lick)." The trajectory shows no discussion of using actual lick direction vs. instruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple binary encoding: 'left' instruction maps to 0, 'right' instruction maps to 1. This per-trial value is broadcast across all 80 time bins in the output array.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
# ...
np.full(N_BINS, output_choice[i], dtype=np.int64)  # broadcast per-trial
```

iii. The AI treated choice as a per-trial variable broadcast across time, matching the instruction format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` which contains string labels: 'ignore' (no response), 'miss' (wrong port), or 'hit' (correct).

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The outcome field directly from the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2. This per-trial value is broadcast across all 80 time bins.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:  # hit
    outcome = 2
output_outcome.append(outcome)
# ...
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. Matches the decoder task specification: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` which contains string labels: 'no early' or 'early'.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. Directly from the NWB trial table's early_lick field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary encoding: 'early' maps to 1, everything else (including 'no early') maps to 0. Broadcast across all 80 time bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
# ...
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. Matches the decoder task specification: "Early lick (no = 0, yes = 1, per-trial)".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (shape N x 3, columns: x, y, likelihood) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps`.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. The AI identified the side-view camera tongue tracking from DeepLabCut, extracting the y-coordinate (column index 1).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial: (1) extract tongue tracking data within the time window around the go cue, (2) align to go cue, (3) compute mean y-position per 50ms bin. No confidence/likelihood filtering is applied - all tracking data is used regardless of DLC confidence. Bins with no tracking frames get NaN.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    n_bins = len(bin_edges) - 1
    tongue_y_binned = np.full(n_bins, np.nan)
    t_abs_start = go_cue_time + bin_edges[0]
    t_abs_end = go_cue_time + bin_edges[-1]
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]
    ts_rel = ts_window - go_cue_time
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
    return tongue_y_binned
```

iii. The AI initially used a confidence threshold (likelihood >= 0.9) but found this produced ~80% NaN values and extreme class imbalance. It removed the filter, noting: "All DLC tracking data used (no confidence filtering)."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization: (1) collect all non-NaN tongue y values across all trials in the session, (2) compute 40th and 60th percentiles, (3) categorize: values below p40 -> 0, between p40 and p60 -> 1, above p60 -> 2. NaN bins default to category 1 (middle).

ii.
```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The AI's CONVERSION_NOTES state: "0 = below 40th percentile, 1 = 40th-60th percentile, 2 = above 60th percentile. Percentiles computed over all tongue y-position values across all trials in the session."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is binned into the same 80 time bins as the neural data using `BIN_EDGES`. For each bin, the mean y-position of all tongue tracking frames falling within that bin's time window is computed. The time bins are defined relative to the go cue, matching the neural alignment.

ii.
```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. Alignment is achieved by using the same `BIN_EDGES` (relative to go cue) for both neural and tongue data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several fallback/handling strategies: (1) Missing tone onset: falls back to -1.85s (typical value). (2) Missing tongue tracking in a bin: NaN, which defaults to category 1 (middle) after discretization. (3) Unparseable photostim values: caught by try/except, defaults to no photostim (all zeros). (4) Units with empty/unmappable brain region annotations: excluded. (5) Trials outside neural recording window: excluded. (6) Sessions with no good units: skipped. (7) String decoding: handles both bytes and str types for NWB string fields.

ii.
```python
if tone_onset_rel is None:
    tone_onset_rel = -1.85  # fallback
# ...
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default NaN bins to middle category
# ...
try:
    ps_onset_val = float(ps_onset_str[ti])
except (ValueError, TypeError):
    pass  # defaults to zeros
```

iii. The AI's trajectory shows it discovered and fixed several data issues: all-zero neural data for trials beyond recording window (fixed with obs_intervals check), output dtype float64 vs int64 (fixed), and tongue confidence filtering causing class imbalance (fixed by removing filter).

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike times from NWB files - each file loads the entire `units/spike_times` array which can be very large. (2) Computing firing rates per unit per trial - this involves nested loops over units and trials, calling `np.histogram` for each combination. (3) Extracting tongue y-position per trial - another loop per trial with per-bin operations. The full conversion produced 144 sessions with ~75K trials.

ii.
```python
# Loading all spike times at once per session
spike_times_data = f['units/spike_times'][:]
# Nested loop: for each trial, for each unit
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The trajectory shows the full conversion ran successfully. The conversion output doesn't include timing, but the nested unit x trial loop for firing rate computation is the clear bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized: (1) The inner loop over units for firing rate computation (line 492-493) could be replaced by vectorized binning across all units simultaneously for each trial. (2) The tongue y extraction loop over bins (lines 313-316) could be replaced with `np.digitize` or vectorized bin assignment. (3) The per-trial loop itself (line 487) processes one trial at a time but could potentially batch multiple trials.

ii.
```python
# Currently iterates per-unit:
for j, st in enumerate(unit_spike_times):
    fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)

# Currently iterates per-bin for tongue:
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. No justification given; the AI did not discuss vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. (1) String decoding from bytes - done separately for outcomes, early_lick, instructions, photostim_onset, photostim_duration, classification, and anno_names, each with the same pattern. (2) The `map_anno_name_to_region` function iterates through all region prefixes for each unit, even though many units share the same annotation. (3) Building `brain_regions` and `brain_region_idx` involves redundant `list.index()` lookups.

ii.
```python
# Repeated string decode pattern:
outcomes = np.array([o.decode() if isinstance(o, bytes) else o for o in f['intervals/trials/outcome'][:]])
early_lick = np.array([e.decode() if isinstance(e, bytes) else e for e in f['intervals/trials/early_lick'][:]])
instructions = np.array([i.decode() if isinstance(i, bytes) else i for i in f['intervals/trials/trial_instruction'][:]])
# ... same pattern repeated 7+ times
```

iii. No justification given; this is a code style/efficiency observation rather than a correctness issue.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The observation intervals are loaded and processed for all units (including non-good units) before filtering - although the `min_obs_end`/`max_obs_start` computation only iterates over selected unit_indices. (2) All spike times for all units are loaded into memory even though only "good" units are used. (3) The per-trial output arrays broadcast scalar values (choice, outcome, early_lick) across all 80 time bins, creating 80 copies of the same value per trial - this wastes memory for per-trial variables. (4) The full tongue tracking data array is loaded into memory even if only a small portion overlaps with trial windows.

ii.
```python
# All spike times loaded, but only good units used:
spike_times_data = f['units/spike_times'][:]  # loads ALL units
# ...
for ui in unit_indices:  # only iterates over good units
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. The AI did not explicitly discuss these inefficiencies. Loading all spike times at once (rather than on-demand per unit) is a pragmatic choice for simplicity at the cost of memory.
