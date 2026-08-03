# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files found under `data/` by recursively globbing for `*.nwb` files. Each NWB file represents one session and is opened with `h5py`. From each file, the code extracts: units table (spike times, quality, electrodes), trials table (start/stop times, outcome, choice, early_lick, auto_water, free_water), behavioral events (go cue, sample onset, photostim, lick times), and behavioral time series (tongue tracking). All sessions are processed sequentially and accumulated into lists.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
# ...
for fp in files:
    sess = load_session(fp, brain_regions)
```

```python
def load_session(nwb_path, brain_regions):
    with h5py.File(nwb_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = trials['id'].shape[0]
        # ... extracts all variables from NWB
```

iii. The AI documented that data are organized as per-subject directories under `data/`, containing NWB files. It identified the data structure through exploration in Step 2 of CONVERSION_NOTES.md and processed all 174 NWB files found.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). A subject list and mapping are maintained to assign each session to a subject index.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name

# In main():
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_map[subj])
```

iii. The AI determined the subject identity from the directory structure, noting 28 subjects in the dataset, which matches the data organization.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are processed independently in sorted order. The AI includes all 174 NWB files found without excluding any sessions.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)

def choose_sessions(all_files, sample=False):
    all_files = sorted(all_files)
    return all_files[:2] if sample else all_files
```

iii. The AI noted that the paper reports 173 behavioral sessions, but the NWB dataset contains 174 files. The AI retained all 174 sessions, noting that "the NWB release includes extra non-ogen/extra-region sessions beyond the exact analysis subset in the paper."

## 1-d. How are the data split into trials?

i. Trials are identified from the `intervals/trials` table in each NWB file. The number of trials is taken from `trials['id'].shape[0]`. The code iterates over `min(n_trials, len(go_times))` trials, using the index `i` to access trial-level metadata.

ii.
```python
n_trials = trials['id'].shape[0]
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
    # ... process trial i
```

iii. The AI matched trials to go cue events by assuming a 1:1 correspondence between trial indices and go_times indices, capped at the minimum of both lengths.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on several criteria: (1) valid/finite go cue time, (2) the aligned trial window [-2.5s, +1.5s] fits within trial start/stop times, (3) choice, outcome, and early_lick values are recognized labels, (4) a valid (finite) sample event time exists. However, the code does NOT filter out auto_water or free_water trials despite loading these variables.

ii.
```python
for i in range(n_match):
    go = go_times[i]
    if not np.isfinite(go):
        continue
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
    if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
        continue
    # ...
    sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
    if not np.isfinite(sample_time):
        continue
```

```python
# These are loaded but never used for filtering:
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

iii. The AI identified in its CONVERSION_NOTES that the reference code defines "regular trials" as excluding early_lick, auto_water, free_water, and no-response trials. However, the final code does not filter on auto_water or free_water. The AI justified not excluding early_lick trials since it is a decoder output variable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the spike times array), `units/spike_times_index` (index into the flat spike times array per unit), `units/unit_quality` (for filtering good units), `units/electrodes` (to map units to brain regions), and `general/extracellular_ephys/electrodes/location` (for electrode region labels).

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]

spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. The AI documented in Step 5 mapping that neural data comes from `units/spike_times` for units passing curation.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning -2.5 s to +1.5 s relative to go cue onset, producing firing rates in Hz (spike count / bin_size). This yields 80 time bins per trial. The function uses `np.searchsorted` for efficient indexing and `np.bincount` for fast histogram computation.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))  # 80

def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    if b <= a:
        return np.zeros(N_BINS, dtype=np.float32)
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < N_BINS)]
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE
```

iii. The AI documented in Step 5 that spike binning uses 50 ms bins from -2.5 s to +1.5 s around go cue, converting to firing rates. Optimization using `searchsorted` + `bincount` was added in Step 10 to speed up processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: (1) `unit_quality == 'good'` and (2) the unit's brain region must be in the `MAJOR_REGIONS` set (left/right ALM, Striatum, Thalamus, Midbrain, Medulla). This yields 142,233 total units across 174 sessions.

ii.
```python
MAJOR_REGIONS = {'left ALM','right ALM','left Striatum','right Striatum',
                 'left Thalamus','right Thalamus','left Midbrain','right Midbrain',
                 'left Medulla','right Medulla'}

keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```

iii. The AI noted the discrepancy between 154,948 good units (before region restriction) and the paper's 69,943. It added region filtering to narrow units to major brain regions, getting 142,233, which is still substantially more than the paper total of 69,943. The AI acknowledged this discrepancy but did not resolve it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Spike times are extracted from [go_time - 2.5, go_time + 1.5] seconds, then binned relative to go cue time.

ii.
```python
lo = go_time + T_START   # go_time - 2.5
hi = go_time + T_END     # go_time + 1.5
# ...
rel = spike_times[a:b] - lo  # relative to start of window
```

And in the trial loop:
```python
go = go_times[i]
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The AI documented go-cue alignment as the temporal reference, consistent with the instructions specifying alignment to "Go cue onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms (0.05 s), producing 80 bins across the 4-second trial window [-2.5, +1.5]. No rebinning is applied; spikes are directly binned at 50 ms resolution from raw spike times.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))  # 80
```

iii. The AI documented 50 ms bin width as specified by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample/tone onset event timestamps) and `acquisition/BehavioralEvents/go_start_times/timestamps` (the go cue timestamps). The sample time for each trial is found by searching the global event stream for the last sample event within the trial window before the go cue.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]

sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. The AI initially had a bug with direct trial-indexed access of sample_start_times, then fixed it by implementing `find_last_event_within_trial()` to match sample events to trials by time window.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time relative to go cue is computed (`tone_rel = sample_time - go_time`). Then for each bin center, the elapsed time since tone onset is computed as `BIN_CENTERS - tone_rel`. Values before tone onset (negative elapsed time) are clipped to 0.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The AI designed this as a continuous, time-varying input showing elapsed time since tone onset, with 0 for bins before the tone.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers (`BIN_CENTERS`) relative to go cue onset, ensuring temporal alignment. The tone onset is expressed relative to go cue, and the time series uses the same 80 time bins as neural data.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
# Both neural binning and time_from_tone use go_time as reference
```

iii. The AI ensured alignment by using the same temporal reference (go cue) and bin structure for all data streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `acquisition/BehavioralEvents/photostim_stop_times/timestamps`, filtered to events within each trial's time window.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]

# In trial loop:
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. The AI identified photostim event streams in the NWB files and matched start/stop events within each trial.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series is constructed. For each photostim start/stop pair within the trial, the function marks time bins that overlap with the stimulation period as 1.0.

ii.
```python
def build_photostim_series(start_times, stop_times, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(start_times, stop_times):
        rs = s - go_time
        re = e - go_time
        overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
        x[overlap] = 1.0
    return x
```

iii. The AI verified that photostimulation ends before go cue (consistent with the methods), and constructed a binary time-varying indicator.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim event times are converted to times relative to go cue, then mapped onto the same BIN_EDGES used for neural data. This ensures temporal alignment.

ii.
```python
rs = s - go_time  # relative to go cue
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Same temporal reference (go cue) and bin structure as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. It is derived from `intervals/trials/trial_instruction` in the NWB file.

ii.
```python
choice = decode_arr(trials['trial_instruction'][:])
CHOICE_MAP = {'left': 0, 'right': 1}
```

iii. The AI mapped `trial_instruction` to choice, with left=0, right=1 as specified.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string value is mapped to an integer using `CHOICE_MAP`. This per-trial value is then broadcast to all 80 time bins in the output array.

ii.
```python
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    # ...
])
```

iii. Per-trial categorical value broadcast across all time bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `intervals/trials/outcome` in the NWB file.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
```

iii. Direct mapping from trial table outcome field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string value is mapped to an integer using `OUTCOME_MAP` (ignore=0, miss=1, hit=2). This per-trial value is broadcast to all 80 time bins.

ii.
```python
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```

iii. Per-trial categorical value broadcast across time bins, matching the specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived from `intervals/trials/early_lick` in the NWB file.

ii.
```python
early = decode_arr(trials['early_lick'][:])
EARLY_MAP = {'no early': 0, 'early': 1}
```

iii. Direct mapping from trial table early_lick field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string value is mapped to an integer using `EARLY_MAP` (no early=0, early=1). This per-trial value is broadcast to all 80 time bins.

ii.
```python
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
```

iii. Per-trial categorical value broadcast across time bins.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 for y-position) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. The AI used the side camera tongue tracking y-position data.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The raw tongue y-position is interpolated to bin centers (go-cue-aligned) using `np.interp`. The interpolation handles NaN values by filtering them out first. The continuous interpolated values are then discretized per session using percentile thresholds.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid],
                     left=np.nan, right=np.nan).astype(np.float32)
```

iii. The AI documented that tongue y values are interpolated to trial bins then discretized per session.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-level 40th and 60th percentiles are computed over all finite tongue y values across all trials in the session. Then each bin value is categorized: < 40th percentile = 0 (low), 40th-60th = 1 (mid), > 60th = 2 (high). NaN values in tongue_y default to category 1 (mid) since neither the `< q40` nor `> q60` conditions are True for NaN.

ii.
```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont
                             if np.isfinite(x).any()])
q40, q60 = np.percentile(all_tongue, [40, 60])

disc = np.full(N_BINS, 1, dtype=np.int64)  # default to mid
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. The AI followed the instruction specification: 0 for < 40th percentile, 1 for 40-60th, 2 for > 60th, computed per session.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are converted to times relative to go cue, then interpolated to the same bin centers used for neural data, ensuring temporal alignment.

ii.
```python
rel_t = track_t - go_time
return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], ...)
```

iii. Same temporal reference and bin structure as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- Trials with non-finite go cue times are skipped.
- Trials where the alignment window exceeds trial boundaries are skipped.
- Trials with unrecognized choice/outcome/early_lick labels are skipped.
- Trials without a valid sample event are skipped.
- Sessions with fewer than 2 valid trials are skipped entirely (`return None`).
- Tongue tracking NaN values: during interpolation, NaN values are filtered; during discretization, NaN defaults to category 1 (mid).
- Mismatched photostim start/stop counts are handled by taking `min(len(ps), len(pe))` pairs.

ii.
```python
if not np.isfinite(go): continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]: continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP: continue
if not np.isfinite(sample_time): continue
if len(sess_neural) < 2: return None

# Photostim mismatch handling:
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. The AI documented fixing the sample event matching issue (initially assumed trial-indexed, then fixed to search within trial windows). The CONVERSION_NOTES document this as one of the issues found and resolved.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning across all units and trials. The full conversion took ~1423 seconds (~24 minutes) for 174 sessions. The AI optimized the spike binning from `np.histogram` to `np.searchsorted` + `np.bincount`, reducing sample conversion time from ~30.6s to ~11.0s for 2 sessions.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The AI documented the bottleneck in Step 10 of CONVERSION_NOTES and applied optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop (list comprehension over `kept_spikes`) could potentially be vectorized further. The per-trial loop iterating over all trials and calling spike binning per unit is the main loop. The photostim series construction also loops over start/stop pairs.

ii.
```python
# Per-unit loop within each trial:
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)

# Per-photostim-event loop:
for s, e in zip(start_times, stop_times):
    # ...
```

iii. The AI noted optimization opportunities but the per-unit spike binning loop remains as a list comprehension since each unit has different spike times arrays.

## 10-c. What processing does the code repeat multiple times?

i. The code loads and decodes all unit qualities, electrode regions, and event timestamps for each session, but this is done once per session and is not repeated unnecessarily. The photostim event filtering is done per trial (searching the full event arrays each time), which is somewhat redundant but not a major bottleneck.

ii.
```python
# Per-trial photostim filtering searches entire array:
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
```

iii. No significant repeated processing was identified by the AI.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `auto_water` and `free_water` trial variables but never uses them for any filtering or output. Left and right lick time arrays are loaded but never used. The code also stores `q40`, `q60`, `n_units_kept`, and `n_trials_kept` in the session result dictionary, but these are only used for printing and not saved to the output pickle.

ii.
```python
# Loaded but never used:
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
```

iii. The AI loaded these variables during exploration but did not incorporate them into filtering or processing logic.
