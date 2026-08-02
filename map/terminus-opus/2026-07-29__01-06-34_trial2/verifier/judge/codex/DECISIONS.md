# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds every NWB file under `data/sub-*/*.nwb`, sorts them, and processes them one file at a time with `pynwb`. Each file is treated as one session and all later subject/trial structures are built from those per-file reads.

ii. ```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. In `CONVERSION_NOTES.md` Step 2 the agent documented "174 NWB files" and in Step 6 it says the implementation "Uses pynwb to read NWB files." This is also consistent with the trajectory after the agent decided to adapt the reference `.mat` workflow to NWB.

## 1-b. How are the data split into subjects?

i. Subjects are read from `nwb.subject.subject_id`. After all sessions are processed, unique subject IDs are sorted into `subjects`, and each session gets a `subject_idx` pointing into that list.

ii. ```python
subject_id = nwb.subject.subject_id

subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. `CONVERSION_NOTES.md` Step 5 lists `subject_id` as session metadata and Step 9 reports 28 subjects matching the paper, so the agent intentionally used NWB subject IDs as the mouse split.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The script runs `process_session(...)` once per file and appends the result if the session has at least one good neuron and at least two valid trials.

ii. ```python
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE,
                       show_processing=args.show_processing and i < 2, session_idx=i)
    if r: all_sessions.append(r)
    else: skipped += 1
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent explicitly changed course to "Include all 173 sessions with good neurons." Trajectory steps 94-95 say the earlier session-level behavioral filter was removed so the final session list would match the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from rows of the NWB `trials` table. The agent first determines which trial rows were actually recorded neurally using `obs_intervals`, then keeps only recorded trials that also pass the regular-trial mask.

ii. ```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)

regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. `CONVERSION_NOTES.md` Step 5 says "`obs_intervals`: Only process trials within recording intervals." Trajectory steps 57-58 explain that the agent discovered some sessions only record a subset of behavioral trials and therefore intersected recorded trials with the trial-quality mask.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering uses the reference `get_regular_trial_mask` logic: keep only trials with no early lick, no auto water, no free water, no ignore outcome, and no photostimulation. It also drops sessions with fewer than two remaining trials.

ii. ```python
mask_no_early = (early_lick == 'no early')
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim

if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 all say the trial curation should match `get_regular_trial_mask`. Trajectory steps 94-95 show the agent deliberately kept this trial-level filter even after removing session-level performance filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the NWB units table, specifically `units['spike_times']` for units whose `classification` is `"good"`. Recorded-trial boundaries are inferred from `units['obs_intervals']`.

ii. ```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]

recorded_trials = get_recorded_trial_indices(nwb, good_indices)
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `units.spike_times (good)` to `neural`, and Step 3 says neuron curation is classifier-based using the NWB `classification` field.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the script bins each good unit's spike times into a go-cue-centered window and converts counts to firing rates by dividing by bin width.

ii. ```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        ...
        counts, _ = np.histogram(st_w, bins=bin_edges)
        fr[i] = counts.astype(np.float32) / bin_size
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `sliding_histogram` in the reference code as the firing-rate construction step, and Step 6 says the agent replaced it with `np.histogram for efficient spike binning`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with zero good neurons are skipped entirely.

ii. ```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. `CONVERSION_NOTES.md` Step 3 states "Neuron curation: classification == 'good' (classifier-based QC)," directly tying this to the spike-sorting QC white paper and the reference code's classifier mode.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural window is aligned to `BehavioralEvents/go_start_times`. For each kept trial, spikes are counted from 2.5 s before to 1.5 s after the go cue.

ii. ```python
go_times = be.time_series['go_start_times'].timestamps[:]
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says "align go cue, [-2.5, 1.5]s," and Step 10 says "Temporal alignment: Go cue onset, same as reference code."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. Spikes are binned directly at that resolution; there is no second-stage rebinning after the initial histogram.

ii. ```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
n_bins = int(round((t_end - t_start) / bin_size))
...
'time_bin_size': BIN_SIZE * 1000,
```

iii. `CONVERSION_NOTES.md` Step 5 lists "Bin 50ms" for neural data, and Step 10 notes this follows the decoder task spec even though some reference video analyses used 40 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, using the first sample-start timestamp that falls inside the current trial's `start_time` to `stop_time` interval.

ii. ```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]

def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent maps "time from sample_start" to `input[0]`, treating sample start as tone onset. The trajectory's manual sanity checks also refer to this variable as the tone-onset input.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the trial's sample-start time, the code converts it into go-cue-relative coordinates and subtracts it from each bin center, producing a continuous value in seconds for every neural bin. Missing sample starts become `NaN`.

ii. ```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes this as a continuous time-varying signal rather than a categorical event. Step 10 records a manual check that one computed value matched the NWB source.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the same 80 go-cue-centered bin centers used for the firing-rate matrix, so each time point in `input[0]` corresponds directly to one neural bin.

ii. ```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
...
input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
```

iii. `CONVERSION_NOTES.md` Step 5 says all streams are aligned to go cue with the same analysis window, and the code uses the shared `bin_centers` array for both neural and input construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trial-table fields `photostim_power`, `photostim_onset`, and `photostim_duration`.

ii. ```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. `CONVERSION_NOTES.md` Step 5 maps `photostim active` to `input[1]`, and trajectory steps 46-49 show the agent explicitly investigated how the onset field was encoded before finalizing this mapping.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The script first identifies stimulated trials from power values, then converts the onset from trial-start-relative time into go-cue-relative time and marks bins as 1 while stimulation is active. Because the earlier trial filter removes all stimulated trials, the retained `photostim_on` vectors are effectively always zero.

ii. ```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. Trajectory steps 47-49 describe the agent's reasoning that `photostim_onset` is trial-start-relative and needs conversion to go-cue-relative time. `CONVERSION_NOTES.md` Step 12 then acknowledges that the input is "Always 0 because photostim trials are filtered out."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim vector is aligned to the same go-cue-centered 50 ms bins as the neural activity by comparing the shared `bin_centers` to the converted stimulation interval.

ii. ```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
```

iii. The agent's notes and trajectory both emphasize that all streams should be go-cue aligned. The photostim implementation follows that plan after converting the raw onset field into the same time basis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trials['trial_instruction']`, not from realized lick-time streams. `"left"` becomes 0 and `"right"` becomes 1.

ii. ```python
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `trial_instruction` to `choice`, with `left=0, right=1` to match the decoder specification.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code performs a binary string-to-integer mapping and then repeats the per-trial choice value across all 80 bins in the output tensor.

ii. ```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
...
out = np.zeros((4, n_bins), dtype=np.int64)
out[0, :] = choice
```

iii. `CONVERSION_NOTES.md` Step 5 says choice is a per-trial output. The implementation keeps it time-aligned by broadcasting that trial label across the neural window.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `trials['outcome']`.

ii. ```python
outcome = nwb.trials['outcome'][:]
...
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. `CONVERSION_NOTES.md` Step 5 maps the NWB outcome strings directly to the decoder's requested three-class coding.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps the raw string outcome into `{ignore: 0, miss: 1, hit: 2}` and then repeats that label across all time bins. In practice, the earlier trial filter removes ignore trials, so the retained data only contain miss and hit.

ii. ```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
...
out[1, :] = out_val
```

iii. `CONVERSION_NOTES.md` Step 12 notes that outcome only shows miss and hit after filtering and treats that as a consequence of matching the reference regular-trial mask.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The script does not create any distance-to-reward-zone output. The closest relevant implementation is `outcome`, which is aligned by repeating the same per-trial label across every neural time bin.

ii. ```python
out = np.zeros((4, n_bins), dtype=np.int64)
out[0, :] = choice
out[1, :] = out_val
out[2, :] = early_val
out[3, :] = ty
```

iii. Neither the instructions, the methods excerpt, nor the reference code define a distance-to-reward-zone target here. The agent therefore only aligned the listed decoder outputs, with `outcome` broadcast across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `trials['early_lick']`.

ii. ```python
early_lick = nwb.trials['early_lick'][:]
...
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. `CONVERSION_NOTES.md` Step 5 maps `early_lick` directly to the decoder output and uses the specified `no=0, yes=1` coding.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code converts the string label to a binary value and repeats it across all bins. Because early-lick trials were already excluded by the regular-trial filter, the retained output is always 0.

ii. ```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
...
out[2, :] = early_val
```

iii. `CONVERSION_NOTES.md` Step 12 explicitly says early lick is trivial because all early-lick trials were filtered out and claims this is "consistent with the reference code."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y comes from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of the data array is y-position and column 2 is the DeepLabCut likelihood used for confidence filtering.

ii. ```python
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr = td[:, 1]
tongue_lk_arr = td[:, 2]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `tongue_y` to the fourth output and Step 10 says the agent used a DLC likelihood threshold to decide which tracking samples were valid.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent collects tongue y values from kept trials within the neural analysis window, keeps only samples with likelihood above 0.9, and for each 50 ms bin averages high-confidence y values within that bin before discretization.

ii. ```python
good = lk > 0.9
...
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    ...
    my = np.mean(tongue_y_arr[il:ih][gd])
```

iii. `CONVERSION_NOTES.md` Step 5 states "Use DLC likelihood > 0.9 threshold," and trajectory step 49 identifies tongue processing as a performance bottleneck that the agent tried to accelerate.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The script computes the 40th and 60th percentiles from concatenated high-confidence tongue-y samples taken only from valid trials and only within the go-cue analysis window. Each binned mean is then assigned 0 if below p40, 2 if above p60, else 1. Missing/low-confidence bins default to 1.

ii. ```python
concat = np.concatenate(all_ty)
tongue_y_p40 = np.percentile(concat, 40)
tongue_y_p60 = np.percentile(concat, 60)
...
ty = np.ones(n_bins, dtype=np.int64)
...
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. `CONVERSION_NOTES.md` Step 5 says the goal was session-wise `<40 / 40-60 / >60` discretization, but the code operationalizes that over kept analysis-window samples and adds a default-middle imputation when tracking is absent or low confidence.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned to the neural data by using the same go-cue-centered 50 ms bins and assigning one categorical tongue state per neural bin.

ii. ```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
...
bc = go + bin_centers[b]
...
out[3, :] = ty
```

iii. `CONVERSION_NOTES.md` Step 5 says all decoder streams should share the go-cue alignment window, and the tongue output follows those same `bin_centers`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses a mix of silent fallback rules. Missing sample starts become all-`NaN` in `time_from_tone_onset`. Missing tongue tracking, low-confidence tracking, or bins with no good frames default to the middle tongue class. Sessions with no good neurons or fewer than two valid trials are skipped. Ambiguous `obs_intervals` are matched to the nearest trial start within 0.1 s.

ii. ```python
return sample_start_times[mask][0] if np.any(mask) else None
...
tft = np.full(n_bins, np.nan, dtype=np.float32)
...
ty = np.ones(n_bins, dtype=np.int64)
...
if diffs[best] < 0.1:
    recorded.append(best)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly justifies one of these defaults: "default to class 1 (mid) when no high-confidence detection." Steps 10 and 12 also say edge cases such as boundary trials and partial recording coverage were accepted if the decoder verifier tolerated them.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs are spike binning across all neurons and trials, and tongue-y processing across bins within each trial. Loading the full tongue tracking array and spike-time lists also contributes.

ii. ```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
...
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
...
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
```

iii. `CONVERSION_NOTES.md` Step 6 says processing was about 10 s/session, and trajectory steps 49 and 64 identify spike binning and tongue-y binning as the main bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-trial, neuron-by-neuron histogram loop in `bin_spikes_all_trials` could be vectorized or parallelized. The per-bin tongue loop inside each trial could also be vectorized against precomputed frame indices. The `valid_trials` list comprehension and repeated `subjects.index(...)` / `regions.index(...)` lookups are smaller avoidable loops.

ii. ```python
for go in go_times_arr:
    ...
    for i, st in enumerate(spike_times_list):
        ...

for b in range(n_bins):
    ...

valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. `CONVERSION_NOTES.md` Step 6 says the agent tried to make the code efficient, and trajectory step 49 specifically calls out tongue-y binning and firing-rate computation as the places to optimize further.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans event arrays and timestamps: it rescans `sample_start_times` for every trial, rescans `tongue_ts` once to estimate session thresholds and again for every bin of every trial, and redoes membership/index lookups while building output metadata.

ii. ```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
...
for ti in valid_trials:
    ...
for b in range(n_bins):
    ...
subj_idx.append(subjects.index(s['subject_id']))
br_idx.append(np.array([regions.index(r) for r in s['neuron_regions']]))
```

iii. This follows directly from the implementation structure in `convert_data.py`. The agent's timing notes in Steps 6-7 and trajectory step 49 are consistent with these repeated scans being costly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes session performance and left/right correct counts but no longer uses them for filtering. It also constructs a photostim input that is always zero after trial filtering, and it expands per-trial categorical outputs across all 80 bins even though choice/outcome/early-lick are conceptually per-trial labels.

ii. ```python
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
correct_left = ...
correct_right = ...
print(f"  Performance: {perf:.1%} ...")

ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    ...

out[0, :] = choice; out[1, :] = out_val; out[2, :] = early_val
```

iii. Trajectory steps 94-95 show that session-level filtering based on performance was explicitly removed, leaving those computations as dead work. `CONVERSION_NOTES.md` Step 12 also admits the photostim input is always zero and early-lick is trivial after the chosen trial filter.
