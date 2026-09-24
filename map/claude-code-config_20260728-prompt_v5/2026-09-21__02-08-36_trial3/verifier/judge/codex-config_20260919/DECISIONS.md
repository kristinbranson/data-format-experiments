# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates sorted `sub-*` directories and sorted `.nwb` files, then opens each file with `pynwb.NWBHDF5IO`; each file supplies units, trials, behavioral events, and behavioral time series. Full mode processes all 174 files sequentially.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
for subj in subjects:
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    ...
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The notes say the dataset contains 28 subject directories and 174 NWB files, with one file per session. Sorting makes traversal deterministic; the agent chose NWB fields corresponding to the reference variables.

## 1-b. How are the data split into subjects?

i. Subject membership is taken from the `sub-*` directory name passed into `process_session`, not from `nwb.subject.subject_id`. Unique directory names are accumulated in first-seen order and sessions receive indices into that list.

ii.
```python
all_files.append((subj, os.path.join(subj_dir, f)))
...
subj = sd['subject_id']
if subj not in seen_subjects:
    seen_subjects[subj] = len(seen_subjects)
    subject_list.append(subj)
```

iii. The directory layout was documented as `/app/data/sub-XXXXXX/`; the agent treated those directory labels as stable subject identifiers and verified 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its `nwb.identifier` is retained in metadata, and each surviving `process_session` result becomes one session-level list in the output.

ii.
```python
session_id = nwb.identifier
...
all_neural.append(sd['neural'])
all_session_ids.append(sd['session_id'])
```

iii. The agent documented that each subject directory contains 1–10 NWB files and that one file represents one session. It drops the single file with zero classifier-good units, yielding the expected 173 sessions.

## 1-d. How are the data split into trials?

i. Trial rows are indexed from `nwb.trials`; the same indices select `go_start_times`, labels, and filtered trial outputs. Neural and time-varying data are cut into one 80-bin array per selected trial around that trial's go cue.

ii.
```python
n_trials_total = len(trials)
valid_trial_indices = np.where(trial_mask)[0]
go_cues_valid = go_start_times[valid_trial_indices]
for t_idx, trial_idx in enumerate(valid_trial_indices):
```

iii. The trials table and go-cue event series were identified as corresponding trial-wise structures. The agent did not add the reference assertion that their lengths agree.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes `auto_water` and `free_water` rows, keeps early-lick, ignore, and photostimulation trials because they are requested decoder variables, and drops a session if fewer than two rows remain. It does not use `units.obs_intervals`, so 1,061 trials with no neural recording remain as all-zero neural arrays.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
valid_trial_indices = np.where(trial_mask)[0]
if n_valid < 2:
    return None
```

iii. The notes explicitly justify retaining early-lick, ignore, and stimulated trials because excluding them would undermine the requested inputs/outputs. They call zero-neural trials “acceptable,” although the instructions specifically said to exclude invalid data periods. The extra `auto_water` exclusion was inherited from the paper's regular-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units['spike_times']` for units whose `classification` equals `good`, using `BehavioralEvents/go_start_times` to define trial windows.

ii.
```python
good_indices = np.where(classification == 'good')[0]
st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The agent identified spike times as the NWB neural representation and classifier-based QC as the reference processing mode.

## 2-b. How is the `neural` data processed?

i. Per neuron and trial, sorted spikes in the four-second window are histogrammed into 80 non-overlapping bins, divided by 0.05 s to obtain Hz, and stored as `float32`. No smoothing or normalization is applied.

ii.
```python
counts = np.histogram(spikes_in_window, bins=edges)[0]
fr_matrix[n_idx, :] = counts / BIN_WIDTH
```

iii. The notes say histogram counts divided by bin width are appropriate for the mandated 50-ms bins; this intentionally replaces the paper code's 40-ms sliding bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == 'good'` units are retained; a session with none is skipped. No individual QC-metric thresholds are applied.

ii.
```python
classification = np.array(units['classification'][:])
good_mask = classification == 'good'
if n_good == 0:
    return None
```

iii. The notes state that this column corresponds to the classifier QC used by the reference pipeline. This produces 173 sessions and 69,453 good units, close to the paper's count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue time is added to the fixed relative bin edges; spikes are selected from `go-2.5` through `go+1.5` and histogrammed on those absolute edges.

ii.
```python
t_start = gc - TIME_BEFORE
t_end = gc + TIME_AFTER
edges = BIN_EDGES + gc
```

iii. The agent recognized that NWB spike and event timestamps share a session clock and documented go cue as time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted resolution is 50 ms: 80 non-overlapping bins from -2.5 to +1.5 s. Raw spike events are binned once; there is no subsequent rebinning.

ii.
```python
BIN_WIDTH = 0.050
N_BINS = int((TIME_BEFORE + TIME_AFTER) / BIN_WIDTH)
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
```

iii. The task explicitly mandates 50-ms firing-rate bins, overriding the paper pipeline's 40-ms window and 3.4-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from a raw per-trial tone event. The agent uses fixed bin centers and a hard-coded `TONE_OFFSET = -1.85`, inferred from the nominal 0.65-s sample plus 1.2-s delay.

ii.
```python
TONE_OFFSET = -1.85
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. The notes argue that task structure fixes tone onset at go cue minus 1.85 s. This overlooks replayed sample epochs after early licks, for which the last actual `sample_start_times` event varies by trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent subtracts the fixed relative tone offset from each go-aligned bin center, creating one common ramp from about -0.625 to 3.325 s and reuses it for every trial.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = (BIN_CENTERS - TONE_OFFSET).astype(np.float32)
```

iii. It describes this as the continuous seconds-since-tone input and assumes no trial-specific processing is necessary.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same 80 relative bin centers used by the neural histograms, but the fixed tone offset means alignment is only nominal, not based on each trial's actual tone timestamp.

ii.
```python
input_data = np.stack([time_from_tone, photostim_inputs[t_idx]], axis=0)
```

iii. The agent's plots and metadata place both signals on the go-cue axis. Its alignment rationale is valid only for trials without a replayed sample epoch.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses session-wide `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, plus each selected trial's go cue. It does not use trial-table `photostim_onset`, `photostim_duration`, or `start_time`.

ii.
```python
photostim_starts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stops = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes identify these BehavioralEvents as absolute photostimulation timestamps and report a plausible ~22% stimulated-bin/trial pattern.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For every trial, the code loops over all session photostimulation intervals, skips nonoverlapping intervals, and marks bin centers in each overlapping half-open interval as 1; all other bins remain 0.

ii.
```python
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
    active = (BIN_CENTERS >= ps_start - gc) & (BIN_CENTERS < ps_stop - gc)
    photostim_binary[active] = 1.0
```

iii. The agent chose a binary time series, as required, and used bin-center sampling. It treats events as session-global rather than explicitly pairing each trial with its table row.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation endpoints are converted to go-relative offsets and compared with the same relative bin centers as neural activity.

ii.
```python
ps_start_rel = ps_start - gc
ps_stop_rel = ps_stop - gc
active = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_stop_rel)
```

iii. Shared absolute NWB timestamps and the common go cue make this alignment technically sound.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial-table `trial_instruction` and `outcome`: hit selects the instructed side, miss selects the opposite side, and ignore becomes no lick.

ii.
```python
if outcome == 'ignore': return 2
elif outcome == 'hit': return 0 if trial_instruction == 'left' else 1
elif outcome == 'miss': return 1 if trial_instruction == 'left' else 0
```

iii. The agent notes that actual choice is not a direct trial column but is determined by instruction and correctness/outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It maps left/right/no lick to 0/1/2 and repeats the per-trial category across all 80 output bins.

ii.
```python
output_combined[0, :] = choice_val
```

iii. Repetition lets per-trial and time-varying outputs share a `(4, 80)` array; `output_values` documents the codes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `nwb.trials['outcome']`.

ii.
```python
outcomes = trials['outcome'][:]
outcome = outcomes[trial_idx]
```

iii. The raw column already contains the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 using `.get(outcome, 0)`, then repeated over 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
output_combined[1, :] = outcome_map.get(outcome, 0)
```

iii. The coding matches the requested order. The fallback silently treats unknown/missing labels as ignore, although no such labels were reported.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `nwb.trials['early_lick']`.

ii.
```python
early_licks = trials['early_lick'][:]
```

iii. The raw table explicitly provides `early` and `no early`, so the agent retained these trials and used the column directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1 via `.get(..., 0)`, with the result repeated across time.

ii.
```python
early_map = {'no early': 0, 'early': 1}
output_combined[2, :] = early_map.get(early_licks[trial_idx], 0)
```

iii. The map matches the target category order; repetition reflects that early lick is a per-trial output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 and 2 (y and likelihood) from `BehavioralTimeSeries/Camera0_side_TongueTracking`.

ii.
```python
tongue_data_all = tongue_ts_obj.data[:]
tongue_timestamps_all = tongue_ts_obj.timestamps[:]
tongue_y_all = tongue_data_all[:, 1]
tongue_lik_all = tongue_data_all[:, 2]
```

iii. The notes identify this as the side-camera tongue tracker and use confidence to distinguish visible from not visible.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visible frames are those with likelihood at least 0.9. Session thresholds are raw-frame y percentiles. Within each trial/bin, visible y values are averaged; bins with no visible frame remain category 3. If the series is absent, every bin is category 3; if fewer than ten visible frames exist, thresholds remain zero.

ii.
```python
visible_mask = tongue_lik_all >= DLC_LIKELIHOOD_THRESH
p40 = float(np.percentile(visible_y, 40))
p60 = float(np.percentile(visible_y, 60))
mean_y = np.mean(y_bin[visible])
```

iii. The agent chose a conservative 0.9 DLC threshold and per-session percentiles. It did not justify using raw-frame percentiles rather than percentiles of 50-ms bin means, which is the quantity ultimately categorized.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Mean visible y per bin is 0 below p40, 1 from p40 through p60 inclusive, 2 above p60, and 3 when no frame clears confidence. Thresholds are the 40th/60th percentiles of all confidence-qualified raw frames in the session.

ii.
```python
if mean_y < p40:
    tongue_y_binned[b] = 0
elif mean_y <= p60:
    tongue_y_binned[b] = 1
else:
    tongue_y_binned[b] = 2
```

iii. The category inequalities follow the task. The session-wide scope is correct, but the reference computes percentiles from session-wide 50-ms means and uses a 0.5 visibility threshold.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected by absolute timestamps in `[go-2.5, go+1.5]`, converted to go-relative time, and assigned to the same bin edges as neural activity.

ii.
```python
idx_start = np.searchsorted(tongue_ts, gc + BIN_EDGES[0], side='left')
t_rel = t_slice - gc
bin_assignments = np.digitize(t_rel, BIN_EDGES) - 1
```

iii. The shared NWB clock and common bin edges align camera and neural data without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with zero good units and sessions with fewer than two filtered trials are skipped. Missing tongue series becomes all not-visible; low-confidence/absent bin frames also become not-visible; malformed electrode-group JSON falls back to an empty target; unknown outcome/early labels default to category 0. Crucially, trials outside neural recording coverage are retained as zeros.

ii.
```python
except:
    target = ''
...
trial_tongue = [np.full((1, N_BINS), 3, dtype=np.int64) for _ in range(n_valid)]
```

iii. The agent documented the 1,061 zero-neural trials and judged them acceptable, while suggesting exclusion only as a possible decoder improvement. This conflicts with the task's instruction to exclude invalid recording periods and with the reference use of `obs_intervals`.

## 10-a. What are the most time-consuming steps of the code?

i. The agent's timings indicate firing-rate construction is the main computational cost; unit spike loading/sorting and tongue processing also cost time. Full processing took 582 s, and saving the roughly 12-GB pickle adds I/O cost, though the notes do not quantify saving separately.

ii.
```python
trial_neural = compute_firing_rates_fast(...)
print(f"  Firing rates ({t_fr - t_load:.1f}s)")
```

iii. The notes report about 3.4 s/session and describe `searchsorted` as the key optimization. They provide stage timings but no rigorous aggregate bottleneck analysis.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural computation has nested trial×neuron loops, each calling `searchsorted` and `histogram`; it could vectorize all trial edges per neuron as the reference does. Tongue computation loops trial×80 bins and could use grouped reductions. Photostimulation loops every trial×every session event despite only one relevant trial interval. Output/input assembly loops are also vectorizable.

ii.
```python
for t_idx in range(n_trials):
    for n_idx in range(n_neurons):
        ... np.histogram(...)
...
for b in range(N_BINS):
```

iii. The agent labels these functions “vectorized,” citing `searchsorted` and `digitize`, but substantial Python-level loops remain and explain the slower 582-s conversion versus the reference's more vectorized approach.

## 10-c. What processing does the code repeat multiple times?

i. It re-sorts each unit's already ordered spike times; rebuilds absolute edges and scans neurons per trial; scans the entire photostimulation event list for every trial; repeatedly masks 80 tongue bins; and later traverses all outputs/regions again for summaries before assembling them.

ii.
```python
st = np.sort(np.array(units['spike_times'][i], dtype=np.float64))
for ps_start, ps_stop in zip(photostim_starts, photostim_stops):
```

iii. The notes emphasize optimization but do not acknowledge these repeated operations. Several are avoidable by operating session-wide or indexing each trial's table values directly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It parses electrode-group JSON solely for custom broad brain-region remapping, sorts spike arrays that should already be sorted, computes extensive summaries, and optionally constructs plots. It also computes neural matrices for invalid-coverage trials that should be removed. The broad region mapping is retained in output, but its extra target parsing has limited downstream necessity.

ii.
```python
loc = json.loads(eg.location)
target = loc.get('brain_regions', '')
...
for out_idx, out_name in enumerate(['choice', 'outcome', 'early_lick']):
```

iii. The notes present summary statistics and plots as validation, so they are useful during development but are discarded by downstream decoding. They do not identify any unnecessary work themselves.
