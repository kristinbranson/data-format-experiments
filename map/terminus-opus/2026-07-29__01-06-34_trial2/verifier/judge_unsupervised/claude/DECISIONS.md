# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files found in the `data/` directory. It uses `pynwb.NWBHDF5IO` to read each NWB file individually, iterating over all files matching `data/sub-*/*.nwb`. Each NWB file represents one session.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

# In process_session:
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. The AI noted that the reference code uses `.mat` files but the provided data is in NWB format, so it adapted the loading accordingly. This is documented in CONVERSION_NOTES.md Step 1: "Reference code uses .mat files; our data is NWB format."

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a `subject.subject_id` field. The AI extracts this per session and builds a sorted list of unique subject IDs. The `subject_idx` array maps each session to its subject.

ii.
```python
subject_id = nwb.subject.subject_id
# ...
subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The AI identified 28 unique subjects from the data, matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI processes each file individually via `process_session()`. Sessions with zero good neurons are skipped. The AI initially applied session-level performance filtering (>65% correct, >=50 correct L/R) but later removed it to match the paper's 173 sessions.

ii.
```python
nwb_files = get_nwb_files()
for i, path in enumerate(nwb_files):
    r = process_session(path, ...)
    if r: all_sessions.append(r)
    else: skipped += 1
```

iii. From trajectory Step 94-95: The AI initially filtered sessions by performance but realized the paper's 173 sessions include all sessions with recordings. Only one session was skipped (no good neurons), yielding 173 sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by the NWB `trials` table. The AI uses `nwb.trials` to access trial-level data. Within each session, trials are indexed by integer indices into this table.

ii.
```python
n_trials = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
# etc.
```

iii. The AI reads all trial metadata from the NWB trials table and then applies filtering.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using a mask matching the reference code's `get_regular_trial_mask` function. Filtered trials must satisfy: no early lick, no auto water, no free water, no ignore outcome, no photostimulation. Additionally, only trials with neural recordings (based on `obs_intervals`) are included.

ii.
```python
mask_no_early = (early_lick == 'no early')
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
mask_no_photostim = ~has_photostim
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim

recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The AI documented this as matching `get_regular_trial_mask` from the reference code. From CONVERSION_NOTES.md: "Trial filtering: no early lick, no auto water, no free water, no ignore, no photostim."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `units` table in each NWB file, specifically the `spike_times` field for units classified as 'good' by the quality control classifier.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
```

iii. The AI identified the `classification` field as the QC classifier output, matching the reference code's `qc_mode='classifier'`.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms firing rate bins using `np.histogram`. Each trial window is -2.5s to +1.5s relative to the go cue. Spike counts are divided by bin width (0.05s) to produce firing rates in Hz.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    n_neurons = len(spike_times_list)
    result = []
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, st in enumerate(spike_times_list):
            if len(st) == 0: continue
            mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
            st_w = st[mask]
            if len(st_w) > 0:
                counts, _ = np.histogram(st_w, bins=bin_edges)
                fr[i] = counts.astype(np.float32) / bin_size
        result.append(fr)
    return result
```

iii. The AI chose 50ms bins per the task instructions (reference code uses 40ms with 3.4ms stride for video analysis; 50ms is specified by the decoder task).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons with `classification == 'good'` are included. No additional firing rate thresholds or other neural quality filters are applied.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
```

iii. From CONVERSION_NOTES.md: "classification == 'good' (classifier-based QC)". The reference code uses the same classifier-based QC approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. The go cue times are extracted from `BehavioralEvents/go_start_times`. For each trial, spike times are windowed from go_cue - 2.5s to go_cue + 1.5s.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
# In bin_spikes_all_trials:
bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.05s), producing 80 time bins per trial over the 4s window. This is a direct binning of spike times, not a rebinning of pre-binned data.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
n_bins = int(round((t_end - t_start) / bin_size))  # = 80
```

iii. The instructions specify "Use 50-ms-width bins for computing firing rates." The reference code uses 40ms bins with 3.4ms stride for video analysis, but the decoder task specifies 50ms.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times` (the tone onset timestamps) and `BehavioralEvents/go_start_times` (the go cue timestamps). The tone onset time is found by searching `sample_start_times` for the timestamp falling within each trial's start/stop interval.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
# ...
def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

iii. The AI identified that `sample_start_times` contains more entries than trials (due to early lick replays), so it matches each trial to the correct sample start using the trial's time window.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each time bin, the time from tone onset is computed as: `bin_center - (sample_start - go_cue)`. This gives the elapsed time since tone onset at each bin center, using the go-cue-relative coordinate system. If no sample start time is found for a trial, NaN values are used.

ii.
```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The AI computed the time-from-tone as a continuous variable, matching the instruction "Time from tone onset in seconds (continuous, time-varying)."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same bin centers as the neural data, ensuring alignment. Both use the same go-cue-relative time axis with 50ms resolution.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
# Used for both neural binning edges and input computation
tft = bin_centers - (ss - go)
```

iii. By using the same `bin_centers` array for both neural and input data, temporal alignment is maintained.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `nwb.trials['photostim_power']`, `nwb.trials['photostim_onset']`, and `nwb.trials['photostim_duration']`.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. The AI checks both whether the value is 'N/A' and whether the power is > 0 to determine photostimulation status.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created for each trial. The photostim onset is converted from trial-start-relative to go-cue-relative coordinates, then a binary mask marks time bins where photostim is active. However, since photostim trials are filtered out by `get_regular_trial_mask`, this input is always zero.

ii.
```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md Step 12: "Photostim input: Always 0 because photostim trials are filtered out. This is correct per reference code."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is computed at the same bin centers as neural data. The onset is re-referenced from trial-start-relative to go-cue-relative time.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. Uses the same go-cue-aligned bin centers as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `nwb.trials['trial_instruction']`, which contains 'left' or 'right'.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The AI maps 'left' to 0 and 'right' to 1, matching the instructions.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A simple string comparison maps 'right' to 1 and everything else (i.e., 'left') to 0. The value is constant across all time bins in the trial.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
out[0, :] = choice
```

iii. The AI uses `trial_instruction` (the instructed direction) rather than the actual lick direction. This is consistent with using the trial type/instruction as the "choice" variable, though the instructions say "Lick direction choice." Since ignore and no-response trials are filtered out, and the reference code uses `trial_type` (which is the instruction), this is the same as what the reference does.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `nwb.trials['outcome']`, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = nwb.trials['outcome'][:]
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. Maps to the categories specified in the instructions: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped to integers using a dictionary. Since ignore trials are filtered out, only values 1 (miss) and 2 (hit) appear in the final data. The value is constant across all time bins.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The verification output confirms outcome range is [1.0, 2.0] since ignore trials are filtered.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `nwb.trials['early_lick']`, which contains 'no early' or 'early'.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. Maps 'early' to 1 and 'no early' to 0.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String comparison maps to 0/1. However, since early lick trials are filtered out by `get_regular_trial_mask`, this output is always 0 (trivially constant). The value is constant across all time bins.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The AI acknowledged this is trivial: "early_lick is trivial because all early lick trials are filtered out by get_regular_trial_mask. This is consistent with the reference code."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `BehavioralTimeSeries/Camera0_side_TongueTracking` in the NWB file. The tracking data has columns (x, y, likelihood), and the y-position (column index 1) and likelihood (column index 2) are used.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
if 'Camera0_side_TongueTracking' in bts.time_series:
    tt_obj = bts.time_series['Camera0_side_TongueTracking']
    tongue_ts = tt_obj.timestamps[:]
    td = tt_obj.data[:]
    tongue_y_arr = td[:, 1]
    tongue_lk_arr = td[:, 2]
```

iii. The AI correctly identifies the tongue tracking data from the side camera with DLC likelihood filtering.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each session, the AI computes 40th and 60th percentiles of tongue y-position across all valid trials, using only high-confidence detections (likelihood > 0.9). Then for each time bin, it averages tongue y-positions within the bin window and categorizes them.

ii.
```python
# Session percentiles:
all_ty = []
for ti in valid_trials:
    go = go_times[ti]
    i_lo = np.searchsorted(tongue_ts, go + t_start)
    i_hi = np.searchsorted(tongue_ts, go + t_end)
    if i_hi > i_lo:
        lk = tongue_lk_arr[i_lo:i_hi]
        good = lk > 0.9
        if np.any(good):
            all_ty.append(tongue_y_arr[i_lo:i_hi][good])
tongue_y_p40 = np.percentile(concat, 40)
tongue_y_p60 = np.percentile(concat, 60)
```

iii. The AI computes per-session percentiles from high-confidence detections within the trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. For each time bin: 0 if mean y < 40th percentile, 2 if mean y > 60th percentile, 1 otherwise. Default is 1 (mid) when no high-confidence tongue detection exists in the bin.

ii.
```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    if ih > il:
        lk = tongue_lk_arr[il:ih]
        gd = lk > 0.9
        if np.any(gd):
            my = np.mean(tongue_y_arr[il:ih][gd])
            ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. Matches the instructions: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile." Default to class 1 when tongue is not detected.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is computed per bin using the same bin centers as the neural data. For each 50ms bin, tongue tracking samples within that bin window are averaged.

ii.
```python
half_bin = bin_size / 2.0
bc = go + bin_centers[b]
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. Uses the same go-cue-relative bin centers, ensuring temporal alignment with neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Sessions with no good neurons are skipped.
- Sessions with <2 valid trials are skipped.
- Trials without a sample start time get NaN for time-from-tone input.
- Sessions without tongue tracking data: tongue_y defaults to class 1 (mid) for all bins.
- Trials where tongue is not detected (low DLC likelihood) default to class 1.
- Trials at recording boundaries (obs_intervals) with all-zero neural data are included (1 trial flagged in verification).

ii.
```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The AI documented handling of obs_intervals mismatch and edge cases in CONVERSION_NOTES.md Steps 10 and 12.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading spike times from NWB files, and (2) binning spikes into firing rates. The tongue y discretization was initially slow but was noted as needing optimization.

ii.
```python
t_load = time.time()
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
print(f"  Loaded spike times ({time.time()-t_load:.1f}s)")
# ...
print(f"  Processed {len(valid_trials)} trials ({time.time()-t_fr:.1f}s, ...")
```

iii. From trajectory Step 64: "The bottleneck is spike binning (35ms/trial for 526 neurons)." Total conversion took ~30 minutes for 174 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The tongue y-position discretization loop iterates over all bins for each trial with `np.searchsorted` calls per bin. This could be vectorized by computing all bin boundaries at once.

ii.
```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    # ...
```

iii. The AI noted efficiency concerns in Steps 7 and trajectory but did not fully vectorize this loop. The `np.searchsorted` calls for each of 80 bins per trial could be batched.

## 10-c. What processing does the code repeat multiple times?

i. The spike times are loaded once per session but the masking `(st >= bin_edges[0]) & (st < bin_edges[-1])` is computed per trial per neuron, when a single pre-filtering per neuron for the entire session window could suffice. The tongue tracking `np.searchsorted` is also repeated per bin per trial when it could be precomputed for the session.

ii.
```python
# In bin_spikes_all_trials, for each trial and each neuron:
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
st_w = st[mask]
```

iii. No explicit justification was given for not optimizing this further.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two outputs are always constant after trial filtering:
- **Early lick** is always 0 (early lick trials are filtered out)
- **Photostimulation input** is always 0 (photostim trials are filtered out)

Additionally, the **outcome** never includes class 0 (ignore), since ignore trials are filtered. These are still computed and stored but provide no information to the decoder.

The photostim onset computation (converting to go-cue-relative coordinates) is also unnecessary since photostim trials are always excluded.

ii.
```python
# Photostim processing that's always zero:
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:  # Never true after filtering
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    ...

# Early lick always 0:
early_val = 1 if early_lick[trial_idx] == 'early' else 0  # Always 0 after filtering
```

iii. The AI acknowledged this in CONVERSION_NOTES.md Step 12: "Photostim input: Always 0 because photostim trials are filtered out" and "early_lick is trivial because all early lick trials are filtered out."
