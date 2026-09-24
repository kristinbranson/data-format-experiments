# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent sorts a glob of all NWB files under `data/sub-*`, opens each with `pynwb.NWBHDF5IO`, reads its subject, trials, units, behavioral events, and tongue tracking, and accumulates surviving sessions into one dictionary.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

iii. The notes identify 28 subject folders and 174 NWB files and explain that NWB is used instead of the reference repository's MAT representation while extracting the same variables.

## 1-b. How are the data split into subjects?

i. Each file's `nwb.subject.subject_id` is retained; unique IDs are sorted and each retained session gets an index into that list.

ii.
```python
subject_id = nwb.subject.subject_id
unique_subjects = sorted(set(all_subject_ids))
subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
```

iii. The notes report 28 subjects, matching the paper/data, and treat the NWB subject field as authoritative.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Sessions are additionally dropped unless they have at least two good units, at least two covered trials, control/no-early performance of at least 65%, and at least 50 correct left and right trials.

ii.
```python
if n_good < 2: return None
if correct_rate < 0.65: return None
if correct_left < 50 or correct_right < 50: return None
```

iii. The agent cites paper session-selection criteria and reports retaining 151/174 files; its notes call the 22 behaviorally excluded sessions expected.

## 1-d. How are the data split into trials?

i. Trial labels are indexed by rows of `nwb.trials`, paired positionally with `go_start_times`. Only trials whose complete go-aligned window lies between the earliest and latest spike among good units are retained.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
```

iii. The notes say actual spike-time range was chosen instead of `obs_intervals` to avoid all-zero neural trials, although verification still found rare all-zero trials from recording gaps.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by the inferred continuous spike-time coverage window. Early-lick, ignore, stimulated, auto-water, and free-water trials are not directly excluded. Behavioral criteria are instead applied at session level.

ii.
```python
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
# Outputs for every valid_trial_indices entry are then built.
```

iii. The agent deliberately keeps early-lick, outcome, and photostimulation conditions because they are requested decoder variables. It describes coverage filtering as protection against fabricated zero activity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from `nwb.units['spike_times']` for units whose `classification` is `good`, using `go_start_times` to place trial bins.

ii.
```python
classification = nwb.units['classification'][:]
good_indices = np.where(classification == 'good')[0]
spike_times_all = nwb.units['spike_times'][:]
```

iii. The notes identify classifier QC and spike times as the appropriate NWB equivalents of the reference processing.

## 2-b. How is the `neural` data processed?

i. For each trial and good neuron, spikes in each 50-ms edge interval are histogrammed and divided by 0.05 to yield float32 firing rates in Hz. Spike arrays are sorted first; there is no smoothing or normalization.

ii.
```python
counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
fr[i, :] = counts / bin_width
```

iii. The agent states this matches the reference `sliding_histogram` rate calculation, with the task-required bin size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == 'good'` units are kept. Sessions with fewer than two such units are discarded; no individual metric thresholds or zero-variance filter is applied.

ii.
```python
good_mask = classification == 'good'
n_good = np.sum(good_mask)
if n_good < 2:
    return None
```

iii. The notes connect `classification` to the region-specific classifier QC and report 69,453 good units before session filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are formed by adding offsets from -2.5 to +1.5 seconds to each trial's absolute go-cue time.

ii.
```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The notes explicitly identify go-cue onset as time zero and validate trial firing rates against the NWB source.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over four seconds. Raw spikes are newly binned at this resolution; no later rebinning is performed.

ii.
```python
BIN_WIDTH = 0.050
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)
```

iii. The bin width and window are taken directly from the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from per-trial raw tone events. The agent assumes a constant tone onset at -1.85 seconds relative to every go cue and combines that constant with bin centers.

ii.
```python
TONE_ONSET_REL_GO = -1.85
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The notes cite the nominal sample timing from the paper and say the same time vector is used for all trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The fixed relative tone time is subtracted from every go-relative bin center, producing a vector from -0.625 to 3.325 seconds that is reused unchanged.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
```

iii. The agent's sanity check considered this nominal range expected; it did not account for replayed sample epochs after early licks.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same 80 go-relative bin centers used to define neural histogram intervals and stacked as input row zero.

ii.
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The agent justifies alignment through the common go-cue-centered grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses the session-wide `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, selecting any event that overlaps a trial window.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes describe photostimulation as binary and report spot checks of stimulated and unstimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Absolute bin centers lying inclusively between any selected start and stop are set to 1; all others remain 0.

ii.
```python
mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
photostim[mask] = 1.0
```

iii. The agent intended a per-time-bin on/off representation as required, including all stimulation conditions rather than filtering them out.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Go-relative neural bin centers are converted to absolute time and compared with absolute stimulation event times.

ii.
```python
abs_bin_centers = go_time + bin_centers
```

iii. The common NWB clock and common bin centers provide the alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code uses only `trials['trial_instruction']`, so it encodes instructed side rather than actual lick direction. Outcome is not used to reverse misses or identify ignores.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. The notes claim trial instruction is the choice mapping and validate it against that same column; README documents only left/right.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `left` maps to 0 and `right` to 1, with an unknown default of 0, then the value is repeated across 80 bins. No `no lick` category exists.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64)
```

iii. The agent treats choice as a two-class per-trial variable and broadcasts it to make a common time-varying output shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the NWB trials-table `outcome` column.

ii.
```python
outcome = nwb.trials['outcome'][:]
```

iii. The notes state that NWB already provides the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 (unknown defaults to ignore) and are repeated over all bins.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The notes and README document this categorical mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
```

iii. The agent keeps early trials specifically because early lick is a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1 (unknown defaults to 0), repeated across 80 bins.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. The mapping is documented as no/yes and the per-trial value is broadcast to the shared output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking.data` and the series timestamps. It loads but does not use column 2 tracking likelihood.

ii.
```python
local_y = tongue_data[idx_start:idx_end, 1]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

iii. The notes explicitly decide to use all y values regardless of likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Raw y coordinates are averaged within each trial's 50-ms bins. Percentiles are then computed from all non-NaN bin means among retained trials in that session, and each bin is discretized.

ii.
```python
tongue_y[b] = np.mean(local_y[mask])
all_values = np.concatenate(all_values)
p_low = np.percentile(all_values, 40)
p_high = np.percentile(all_values, 60)
```

iii. The agent says per-session percentiles implement the requested discretization, but deliberately includes low-confidence tracker coordinates.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below the 40th percentile are 0, inclusive 40th–60th are 1, and above the 60th are 2. Missing bins are initialized to category 1 rather than the required category 3.

ii.
```python
discrete = np.ones(len(y), dtype=np.int64)
discrete[valid & (y < p_low)] = 0
discrete[valid & (y > p_high)] = 2
```

iii. Documentation lists only low/mid/high and omits “not visible”; the agent reports an exact 40/20/40 distribution, reflecting inclusion/default handling.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples from `go-2.5` through `go+1.5` are assigned to the same absolute 50-ms intervals as neural data and averaged.

ii.
```python
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. The agent relies on shared NWB timestamps and the common go-centered bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with too few good units/trials or failed behavior criteria are skipped. Coverage is approximated by global spike extrema. Empty tongue bins are NaN during averaging but later imputed as middle category; unknown categorical labels default to class 0. Rare internal all-zero neural trials remain.

ii.
```python
if idx_start >= idx_end: return tongue_y
discrete = np.ones(len(y), dtype=np.int64)  # default middle
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. The notes acknowledge rare recording gaps, call them insignificant, and describe spike-range filtering and session skipping as edge-case handling.

## 10-a. What are the most time-consuming steps of the code?

i. The measured dominant computation is nested per-trial/per-neuron firing-rate histogramming; loading is secondary. Full conversion took 1,143.9 seconds and saving a roughly 10.2-GB pickle is also substantial.

ii.
```python
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(...)
```

iii. The notes estimate about 850 seconds for firing rates versus about 210 seconds for loading and report a 19.1-minute full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-neuron firing-rate loop could search all trial edges per neuron, as the reference does. Tongue binning loops over every bin of every trial, output arrays loop over trials, and summary distributions loop over sessions/trials.

ii.
```python
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list): ...
for b in range(n_bins): ...
for t_idx in range(n_valid): ...
```

iii. The agent calls its histogram method “vectorized” because it narrows spikes with `searchsorted`, but its timing shows the remaining nested Python loops dominate.

## 10-c. What processing does the code repeat multiple times?

i. Unit spike times are loaded once to infer recording range and again for rate computation; every selected spike array is sorted despite NWB spike times already being sorted. Constant per-trial arrays and output fills are also rebuilt repeatedly.

ii.
```python
spike_times_all = nwb.units['spike_times'][:]  # in get_recording_range
spike_times_all = nwb.units['spike_times'][:]  # again in process_session
st = np.sort(np.array(spike_times_all[i], dtype=np.float64))
```

iii. The notes do not acknowledge this duplication and describe the implementation as searchsorted-optimized.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads unused `auto_water` and `free_water`, retains unused `subject_desc`, sorts already sorted spikes, computes behavioral selection statistics used only to discard sessions, and optionally renders extensive diagnostic plots that do not enter the pickle.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
subject_desc = nwb.subject.description
```

iii. The agent presents plotting as visual validation and the behavioral calculations as paper-based session curation, but does not discuss the unused fields or redundant sorting.
