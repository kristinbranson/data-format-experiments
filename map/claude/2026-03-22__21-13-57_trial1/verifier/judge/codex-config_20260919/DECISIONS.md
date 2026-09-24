# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs all subject/session NWB files, sorts them, and opens each with `pynwb`. It reads subject metadata, the units and trials tables, behavioral events, and tongue tracking, then processes files sequentially. Although all 174 files are initially discovered, only 144 sessions are retained.

ii.
```python
return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*_ses-*.nwb')))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
```

iii. The notes justify NWB as the available DANDI representation of the same dataset and report 174 files in 28 subject directories. Sorting makes processing deterministic. They further justify retaining only sessions meeting paper-derived selection criteria.

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `nwb.subject.subject_id`. Retained subjects are inserted into an ordered mapping on first encounter, and `subject_idx` maps each retained session to that list.

ii.
```python
subject_id = nwb.subject.subject_id
...
if sid not in subjects_set:
    subjects_set[sid] = len(subjects_set)
subject_idx.append(subjects_set[sess['subject_id']])
```

iii. The agent treats the NWB subject field as canonical and checks that the result contains the paper's 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. In addition, sessions are excluded unless control, non-early, non-ignore performance is at least 65% and there are at least 50 correct left and 50 correct right trials. Sessions with fewer than two usable trials or no mapped good neurons are also dropped.

ii.
```python
if performance < MIN_PERFORMANCE: return None
if correct_left < MIN_CORRECT_LEFT: return None
if correct_right < MIN_CORRECT_RIGHT: return None
...
all_sessions.append(result)
```

iii. The notes say these are paper session-selection criteria and interpret the resulting 144 sessions as appropriate after filtering, despite also observing that 173 of 174 files contain good units.

## 1-d. How are the data split into trials?

i. Rows of the NWB trials table define trials. The same row indices select go cues and trial-level fields; event streams are aligned by each selected go time.

ii.
```python
n_trials = len(trials)
go_times = be.time_series['go_start_times'].timestamps[:]
trial_indices = np.where(trial_mask)[0]
go_times = data['go_times'][trial_indices]
```

iii. The agent identifies the trials table and one `go_start_times` timestamp per trial as the dataset's explicit trial organization.

## 1-e. How are trials filtered based on quality controls?

i. Auto-water and free-water trials are removed. It also attempts to remove trials outside the neural recording range using the earliest/latest selected spike, but its overlap test keeps any window that merely overlaps that range. Early-lick, ignore, and stimulation trials are kept because they are required outputs/inputs.

ii.
```python
trial_mask[td['auto_water'] == 1] = False
trial_mask[td['free_water'] == 1] = False
recording_mask = (go_times + ALIGN_START <= max_spike_time) & \
                 (go_times + ALIGN_END >= min_spike_time)
```

iii. The notes call water trials artificial conditions and state that early-lick, ignore, and stimulated trials must remain for decoding. After validator warnings, the agent added the recording-range test and described the zero-neural-trial issue as fixed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `units['spike_times']` buffer for units labeled `classification == 'good'`; `go_start_times` supplies trial alignment.

ii.
```python
good_mask = classifications == 'good'
all_spike_times = np.array(spike_times_vi.target.data[:])
all_st_idx = np.array(spike_times_vi.data[:])
```

iii. The agent identifies spike times as the raw neural representation and `classification` as the NWB form of the classifier QC described in the white paper.

## 2-b. How is the `neural` data processed?

i. For every selected neuron and trial, spikes in the four-second window are histogrammed into non-overlapping bins and divided by 0.05 seconds to produce firing rates in Hz. There is no smoothing or normalization.

ii.
```python
rel_spikes = st[idx_lo:idx_hi] - go_t
counts, _ = np.histogram(rel_spikes, bins=bin_edges)
all_matrices[t, n, :] = counts / bin_width
```

iii. The notes say 50-ms bins are mandated by the decoder task and supersede the paper code's 40-ms sliding histogram. Search-based window selection was adopted for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and their annotation must map through the agent's handcrafted 14-region keyword table. Thus otherwise-good units with unmapped annotations are discarded.

ii.
```python
good_indices = np.where(good_mask)[0]
...
region = map_anno_to_region(anno)
if region is not None:
    region_labels.append(region)
    neuron_mask.append(i)
```

iii. The agent justifies classifier-good units from the spike-sorting paper and the 14-region mapping from the analysis repository. The notes record fixes for several initially unmapped or misclassified annotations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike timestamps are windowed relative to each absolute go-cue timestamp, with edges from -2.5 to +1.5 seconds.

ii.
```python
abs_start = go_t + align_start
abs_end = go_t + align_end
rel_spikes = st[idx_lo:idx_hi] - go_t
```

iii. The agent states that NWB spike and event timestamps share a session clock, whereas the original MAT representation was already go-aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over four seconds. Raw spike timestamps are newly binned at that resolution; no later rebinning is applied.

ii.
```python
BIN_WIDTH = 0.050
N_BINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
bin_edges = np.linspace(align_start, align_end, n_bins + 1)
```

iii. The explicit decoder specification is cited as overriding the reference analysis's 40-ms width and 3.4-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial start/stop times, and each trial's go cue. The last sample start within the trial and no later than the go cue is chosen.

ii.
```python
mask = (sample_start_ts >= trial_starts[i]) & (sample_start_ts <= go_times[i])
tone_onsets[i] = matching[-1]
```

iii. The notes explain that early licks can replay the sample epoch, so the last sample onset before the go cue is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each 50-ms bin center, the agent subtracts the selected absolute tone time from the bin's absolute time. If no tone is found it silently emits an all-zero vector.

ii.
```python
tone_rel = tone_t - go_t
tone_input = (bin_centers - tone_rel).astype(np.float32)
...
tone_input = np.zeros(n_bins, dtype=np.float32)
```

iii. The intended quantity is documented as continuous elapsed seconds since tone onset; the zero fallback is described as a case that should not occur for valid trials.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same go-relative bin centers used by the neural histogram, giving one value for each of the same 80 bins.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2,
                          align_end - bin_width/2, n_bins)
tone_input = (bin_centers - tone_rel).astype(np.float32)
```

iii. The agent's justification is that both are expressed on the common go-cue-relative clock.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses session-wide `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, together with each trial's go cue.

ii.
```python
photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes identify these event streams as the direct record of stimulation intervals and use the trials-table photostim field only when computing session performance.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each stimulation interval is converted to go-relative time; every bin center inside `[start, stop)` is set to one, otherwise it remains zero.

ii.
```python
ps_start = photostim_start_ts[si] - go_t
ps_stop = photostim_stop_ts[si] - go_t
if bc >= ps_start and bc < ps_stop:
    ps[b] = 1.0
```

iii. The agent follows the instruction to represent whether stimulation is on at every time point as a binary time series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation timestamps are shifted by the same trial go cue and tested at the same 80 bin centers as the neural data.

ii.
```python
bin_centers = np.linspace(align_start + bin_width/2, align_end - bin_width/2, n_bins)
ps_start = photostim_start_ts[si] - go_t
```

iii. The shared NWB clock and shared go-relative bin grid are the implicit justification.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice solely from `trials['trial_instruction']`, coding the instructed side as if it were the animal's chosen lick side. It loads left/right lick timestamps and outcome but does not use them for choice.

ii.
```python
instructions = td['trial_instruction'][trial_indices]
choices = np.array([0 if ins == 'left' else 1 for ins in instructions], dtype=np.int64)
```

iii. The notes' mapping table explicitly equates `trial_instruction` with choice. No justification addresses miss trials (opposite choice) or ignore trials (no lick).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Instruction strings are mapped to left=0/right=1 and repeated over all 80 time bins. There is no `no lick` class.

ii.
```python
np.full(N_BINS, choices[t], dtype=np.int64)
...
['left', 'right']
```

iii. The agent treats choice as a two-class per-trial output and reports its decoder accuracy, but does not reconcile this with the requested three values.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trials-table `outcome` column.

ii.
```python
outcomes_raw = td['outcome'][trial_indices]
```

iii. The notes state that this column already represents the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped as ignore=0, miss=1, hit=2 and the per-trial code is repeated across all bins. Unknown values would silently map to ignore.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes = np.array([outcome_map.get(o, 0) for o in outcomes_raw])
```

iii. The mapping follows the requested category order; repetition permits all outputs to share a `(4, 80)` array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from `trials['early_lick']`.

ii.
```python
early_lick_raw = td['early_lick'][trial_indices]
```

iii. The trials table supplies the flag directly, so the agent does not reconstruct it from lick events.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` becomes 1 and every other value becomes 0; the code is repeated across all bins.

ii.
```python
early_licks = np.array([1 if el == 'early' else 0 for el in early_lick_raw])
np.full(N_BINS, early_licks[t], dtype=np.int64)
```

iii. The notes describe no=0/yes=1 and deliberately retain early-lick trials because this is a decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 (y) and column 2 (tracking confidence) of `Camera0_side_TongueTracking`, plus that series' timestamps.

ii.
```python
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_conf = tongue_data[:, 2]
```

iii. The notes identify the side-camera stream as the relevant ~294-Hz tongue measurement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Confidence below 0.9 is deemed occluded and replaced by the mean of visible session frames. Values are then averaged within each trial's 50-ms bins; empty bins also receive the session mean. Percentiles are computed over all retained-trial bin means, including imputed values.

ii.
```python
visible_mask = tongue_conf >= confidence_threshold
session_mean_y = np.mean(tongue_y[visible_mask])
tongue_y_imputed[~visible_mask] = session_mean_y
...
trial_tongue_y = np.full(n_bins, session_mean_y)
```

iii. The agent cites a paper statement about setting occluded tongue position to its mean and chose a 0.9 confidence threshold. It notes that imputation often makes the 40th and 60th percentiles equal.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the agent takes the 40th and 60th percentiles of all trial-bin values (including imputed means), forces equal thresholds apart with an epsilon, and assigns three classes. It does not produce category 3, `not visible`.

ii.
```python
p40 = np.percentile(all_values, 40)
p60 = np.percentile(all_values, 60)
if np.isclose(p40, p60):
    p40, p60 = p40 - eps, p60 + eps
d[trial_y >= p40] = 1
d[trial_y >= p60] = 2
```

iii. The notes say the epsilon was added to preserve three categories after mean imputation and list only low/mid/high in `output_values`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples in each `[go-2.5, go+1.5)` window are shifted to go-relative time and assigned to the same 50-ms edges as neural activity.

ii.
```python
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
trial_ts = tongue_timestamps[mask] - go_t
bin_indices = np.digitize(trial_ts, bin_edges) - 1
```

iii. The agent relies on camera, event, and spike timestamps sharing the NWB session clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing tone onsets become zeros; low-confidence or missing tongue samples become the session mean; unknown outcomes become ignore; unmapped neuron regions are dropped. Sessions/trials with too little usable data are dropped, and a spike-range overlap heuristic attempts to suppress unrecorded trials.

ii.
```python
if np.isnan(tone_t): tone_input = np.zeros(n_bins)
tongue_y_imputed[~visible_mask] = session_mean_y
outcome_map.get(o, 0)
if region is not None: neuron_mask.append(i)
```

iii. The agent describes fallbacks as robustness measures and says recording-range filtering fixed all-zero trials. However, the saved verification output still reports many all-zero trials and the overlap condition cannot guarantee complete recording coverage.

## 10-a. What are the most time-consuming steps of the code?

i. The agent's timings identify spike binning as the largest compute cost (about 3.3 seconds/session in its estimate), followed by NWB loading and tongue processing (~1.5 seconds each). Saving a roughly 9.5-GB pickle is also material; the full run took 19.5 minutes.

ii.
```python
t1 = time.time()
neural_trials = bin_spikes(...)
print(f'    Spike binning: {time.time() - t1:.1f}s')
```

iii. These conclusions are based on explicit per-stage timers and the recorded full-conversion output.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural binning still nests neuron and trial loops, and photostimulation nests trial, every session event, and bin loops. Tongue processing loops over trials and all 80 bins. All could be vectorized further, especially the already-sorted spike-edge search and event/bin comparisons.

ii.
```python
for n in range(n_neurons):
    for t in range(n_trials):
...
for t in range(n_trials):
    for si in range(len(photostim_start_ts)):
        for b in range(n_bins):
```

iii. The agent calls spike processing “vectorized” because it uses `searchsorted` within the nested loops and says tongue processing was reduced from 142 seconds to about 1.5 seconds using `digitize`, though its per-bin loop remains.

## 10-c. What processing does the code repeat multiple times?

i. It recomputes identical bin grids in three functions; scans every photostimulation event for every trial; constructs per-trial masks over the full camera timestamp vector; and recreates constant per-trial output vectors. It also loads lick events that are never used.

ii.
```python
bin_edges = np.linspace(align_start, align_end, n_bins + 1)  # multiple functions
mask = (tongue_timestamps >= window_start) & (tongue_timestamps < window_end)
for si in range(len(photostim_start_ts)):
```

iii. The documentation emphasizes a single pass over session files but does not discuss these repeated within-session scans or grid construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `photostim_power`, `photostim_duration`, and left/right lick timestamps; reads `stop_time` only to constrain tone lookup; computes/stores performance counters largely for selection/metadata; and optionally creates diagnostic plots. Most importantly, the lick timestamps that could have validated choice are discarded.

ii.
```python
left_lick_ts = be.time_series['left_lick_times'].timestamps[:]
right_lick_ts = be.time_series['right_lick_times'].timestamps[:]
...
'photostim_power': trials['photostim_power'][:],
```

iii. The notes focus on diagnostics and sanity checks and do not label these fields as waste; they report plotting and decoder validation as useful verification work.
