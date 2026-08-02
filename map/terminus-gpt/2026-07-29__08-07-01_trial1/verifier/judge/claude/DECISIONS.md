# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files located in `data/` using `h5py`. It discovers all `.nwb` files via `Path('data').rglob('*.nwb')`, sorts them, and iterates through each file. Each NWB file is opened with `h5py.File` and relevant data tables are extracted: `intervals/trials`, `units/spike_times`, `acquisition/BehavioralEvents/*`, and `acquisition/BehavioralTimeSeries/*`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
# ...
for fp in files:
    sess = load_session(fp, brain_regions)
```
```python
with h5py.File(nwb_path, 'r') as f:
    trials = f['intervals/trials']
    n_trials = trials['id'].shape[0]
    # ... extracts spike_times, behavioral events, tracking data
```

iii. The AI identified the NWB format from the `data/` directory structure. CONVERSION_NOTES.md Step 2 documents the file organization and variable discovery.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). A subject map tracks unique subjects and assigns indices.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
# ...
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
```

iii. The AI observed that data is organized as per-subject directories (Step 2 of CONVERSION_NOTES.md). The directory names serve as subject identifiers.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI processes 174 NWB files total (the paper reports 173 behavioral sessions). No session-level filtering criteria are applied (e.g., no minimum behavioral performance threshold, no minimum trial count per direction).

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
# ...
for fp in files:
    sess = load_session(fp, brain_regions)
    if sess is None:  # only skips if < 2 valid trials
        print('SKIP', fp)
        continue
```

iii. CONVERSION_NOTES.md Step 4 acknowledges the 174 vs 173 discrepancy but does not resolve it. The methods text states: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The AI does not implement these session selection criteria.

## 1-d. How are the data split into trials?

i. Trials are extracted from the NWB `intervals/trials` table. The number of trials is determined by `trials['id'].shape[0]`. The AI iterates over the minimum of the trial count and the number of go cue timestamps.

ii.
```python
n_trials = trials['id'].shape[0]
# ...
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
    # per-trial processing...
```

iii. The AI uses a positional mapping (trial index i corresponds to go_times[i]), which assumes the go cue events are in 1:1 correspondence with trial rows. This is documented implicitly through code structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: (1) go cue must be finite, (2) the [-2.5s, +1.5s] window around go cue must fit within trial start/stop times, (3) trial_instruction must be 'left' or 'right', (4) outcome must be 'ignore', 'miss', or 'hit', (5) early_lick must be 'no early' or 'early', (6) sample_time must be finite. Critically, the AI does NOT filter out auto_water trials, free_water trials, or trials based on the reference code's `get_regular_trial_mask` criteria. The AI also includes early lick trials, no-response ('ignore') trials, and photostimulation trials.

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
    # ... processes trial (no auto_water/free_water check)
```
Note: `auto_water` and `free_water` are loaded (line 147-148) but never used for filtering.

iii. CONVERSION_NOTES.md Step 1 notes the reference code's `get_regular_trial_mask` excludes early lick, auto water, free water, and no-response trials. However, the AI justified including early lick trials because `early_lick` is a decoder output, and photostimulation trials because `photostim` is a decoder input. The AI did not address auto_water and free_water filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (spike timestamps) and `units/spike_times_index` (index boundaries per unit). Units are filtered by `units/unit_quality` and electrode region information from `general/extracellular_ephys/electrodes/location`.

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]

spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. The AI identifies spike times as the raw neural signal and documents the filtering by unit quality and brain region in CONVERSION_NOTES.md Steps 1-5.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning [-2.5s, +1.5s] relative to the go cue (80 bins total). Spike counts per bin are converted to firing rates by dividing by bin size (0.05s). The result is a (n_neurons, n_timepoints) matrix per trial.

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

iii. CONVERSION_NOTES.md Step 5 documents the 50ms bin width as specified in the instructions. The reference code uses 40ms bin width with 3.4ms stride (sliding window), but the instructions explicitly specify 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `'good'` in `unit_quality` AND belonging to one of the 10 MAJOR_REGIONS are retained. No further QC filtering is applied (no minimum firing rate, no variance check across trials, no additional spike sorting metrics).

ii.
```python
MAJOR_REGIONS = {'left ALM','right ALM','left Striatum','right Striatum',
                 'left Thalamus','right Thalamus','left Midbrain','right Midbrain',
                 'left Medulla','right Medulla'}

keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
```

iii. CONVERSION_NOTES.md Step 3 documents that "good units" are defined by the region-specific QC classifier. The paper reports 69,943 good units; the AI's conversion yields ~142,233 in major regions, a significant unresolved discrepancy (noted in Step 9).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. The go cue timestamp for each trial is obtained from `acquisition/BehavioralEvents/go_start_times/timestamps`. Spikes are binned in the window [go_time - 2.5, go_time + 1.5].

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# ...
go = go_times[i]
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. CONVERSION_NOTES.md Step 5 documents go-cue alignment consistent with the instructions ("Temporally align based on Go cue onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (non-overlapping). No temporal rebinning is applied -- spikes are directly binned from raw spike times into 50ms bins. The reference code uses 40ms bin width with 3.4ms stride for its analyses, but the AI follows the instruction specification of 50ms.

ii.
```python
BIN_SIZE = 0.05  # 50 ms
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
```

iii. CONVERSION_NOTES.md Step 5 notes "50 ms bins" as specified in the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample onset event stream) and `acquisition/BehavioralEvents/go_start_times/timestamps` (the go cue). Also uses trial start/stop times for event matching.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
# ...
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. CONVERSION_NOTES.md Step 10 documents that sample_start_times is a global event stream, not trial-indexed, and documents a fix to select the last sample event within each trial before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time is expressed relative to the go cue (`tone_rel = sample_time - go_time`). Then for each bin center, the elapsed time since tone onset is computed. Values before tone onset (negative) are clipped to 0.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. No explicit justification in CONVERSION_NOTES.md for the clipping behavior (setting pre-tone times to 0). The instructions say "Time from tone onset in seconds (continuous, time-varying)".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone input is computed at the same BIN_CENTERS used for neural data, ensuring perfect temporal alignment.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
# ...
inp0 = build_time_from_tone(sample_time, go)
# Combined into input array alongside neural data
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. Alignment is implicit via shared use of BIN_CENTERS.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `acquisition/BehavioralEvents/photostim_stop_times/timestamps`, filtered to events within each trial's [start_time, stop_time] window.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
# ...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. CONVERSION_NOTES.md Step 5 documents the photostim input mapping.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created. For each photostim on/off pair, bins that overlap the stimulation interval are set to 1.0, all others remain 0.0.

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

iii. CONVERSION_NOTES.md Step 5 notes that photostim should end before the go cue per the methods.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Uses the same BIN_EDGES as neural data for alignment, ensuring temporal consistency.

ii.
```python
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Alignment is implicit through shared bin edges.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/trial_instruction` in the NWB file.

ii.
```python
choice = decode_arr(trials['trial_instruction'][:])
CHOICE_MAP = {'left': 0, 'right': 1}
```

iii. CONVERSION_NOTES.md Step 5 maps `trial_instruction` to choice output.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string value 'left' is mapped to 0, 'right' to 1. The per-trial value is broadcast across all time bins (time-varying but constant within trial).

ii.
```python
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    # ...
])
```

iii. The instructions specify "left = 0, right = 1, per-trial".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` in the NWB file.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
```

iii. CONVERSION_NOTES.md Step 5 maps `outcome` to the output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2. The per-trial value is broadcast across all time bins.

ii.
```python
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```

iii. The instructions specify "ignore = 0, miss = 1, hit = 2, per-trial".

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no "Distance to reward zone" output in this dataset or the AI's conversion. This question does not apply. The outcome output is per-trial and constant across all time bins aligned to the go cue.

ii. N/A

iii. N/A

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` in the NWB file.

ii.
```python
early = decode_arr(trials['early_lick'][:])
EARLY_MAP = {'no early': 0, 'early': 1}
```

iii. CONVERSION_NOTES.md Step 5 maps `early_lick` to the output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values are mapped: 'no early' -> 0, 'early' -> 1. The per-trial value is broadcast across all time bins.

ii.
```python
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
```

iii. The instructions specify "no = 0, yes = 1, per-trial".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 for y-position) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps Camera0_side_TongueTracking column 1 to tongue y.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The tongue y tracking data is interpolated to the neural data bin centers for each trial, then discretized per-session using the 40th and 60th percentiles of all finite tongue y values across the session.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 documents the per-session percentile discretization approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles are computed from all finite tongue y values across all valid trials. Values below 40th percentile -> 0, between 40th-60th -> 1, above 60th -> 2. Default is 1 (middle category).

ii.
```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()])
q40, q60 = np.percentile(all_tongue, [40, 60])
for ty, i in zip(sess_tongue_cont, valid_trial_inds):
    disc = np.full(N_BINS, 1, dtype=np.int64)
    disc[ty < q40] = 0
    disc[ty > q60] = 2
```

iii. The instructions specify "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile, per-session discretization".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking is interpolated to the same BIN_CENTERS used for neural data using `np.interp`. Time points outside the tracking data range are filled with NaN (which defaults to category 1 after discretization).

ii.
```python
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
# interpolates to BIN_CENTERS relative to go cue
```

iii. Alignment is achieved through shared bin center timing.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
- Trials with non-finite go cue times are skipped.
- Trials where the [-2.5s, +1.5s] window extends beyond trial bounds are skipped.
- Trials with unrecognized choice/outcome/early values are skipped.
- If sample_time is not finite (no sample event found in trial), the trial is skipped.
- For tongue tracking, if fewer than 2 valid data points exist, NaN is returned and defaults to middle category (1).
- Sessions with fewer than 2 valid trials are skipped entirely.
- When photostim start/stop counts don't match, the minimum count is used.
- When no finite tongue y values exist in a session, a fallback of `[0, 1]` is used for percentile computation.

ii.
```python
if not np.isfinite(go): continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]: continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP: continue
if not np.isfinite(sample_time): continue
if len(sess_neural) < 2: return None
# Tongue tracking fallback:
all_tongue = ... if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 10 documents the sample_start_times fix and other edge case handling.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning -- iterating over all kept neurons and computing spike counts per bin for each trial. For each trial, every neuron's spike times are searched and binned. The conversion output shows 174 sessions processed.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. CONVERSION_NOTES.md Step 10 notes spike binning optimization: "Optimized spike binning using searchsorted + bincount, reducing sample conversion time from ~30.6 s to ~11.0 s for 2 sessions."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main inner loop iterates over trials (line 157: `for i in range(n_match)`) and within each trial iterates over all neurons for spike binning (line 165: list comprehension over `kept_spikes`). The spike binning per neuron per trial could potentially be vectorized using 2D histogram operations. The `build_photostim_series` function also loops over photostim epochs (line 71: `for s, e in zip(start_times, stop_times)`), though this is typically a small number of iterations.

ii.
```python
# Trial loop
for i in range(n_match):
    # Neuron loop within trial
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. CONVERSION_NOTES.md notes the searchsorted optimization but does not discuss further vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. For each trial, the code re-searches the global photostim and sample event arrays to find events within the trial window. These filtered lists could be precomputed once per session. The `parse_region_strings` function parses JSON strings for every electrode on every session, even though region labels are the same across sessions.

ii.
```python
# Repeated per trial:
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. Not discussed in CONVERSION_NOTES.md.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `auto_water` and `free_water` trial fields (lines 147-148) but never uses them -- these values are read into memory and discarded. The code also loads `left_lick_times` and `right_lick_times` (lines 138-139) but does not use them for any input or output construction. The tongue x-position data is loaded as part of the full tracking array (`tongue[:, 1]` selects only y) but the x-column is read and discarded.

ii.
```python
auto_water = trials['auto_water'][:]      # loaded but never used
free_water = trials['free_water'][:]      # loaded but never used
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]   # loaded but never used
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]  # loaded but never used
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]  # full array loaded, only column 1 used
```

iii. Not discussed in CONVERSION_NOTES.md.
