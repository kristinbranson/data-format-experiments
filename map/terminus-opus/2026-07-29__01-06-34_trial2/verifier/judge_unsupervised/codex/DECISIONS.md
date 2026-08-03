# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds every NWB file under `data/sub-*/*.nwb`, sorts them, and treats each file as one session. It opens each file with `pynwb.NWBHDF5IO`, then reads the session's trials table, units table, behavioral events, and tongue tracking stream.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    n_trials = len(nwb.trials)
```

```python
nwb_files = get_nwb_files()
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE,
                       show_processing=args.show_processing and i < 2, session_idx=i)
```

iii. In `CONVERSION_NOTES.md`, the agent says it "uses pynwb to read NWB files" and that the dataset consists of 174 NWB files across 28 subjects. In the trajectory, it explicitly decided to process "all NWB files" after mapping the `.mat` reference workflow onto the NWB release.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `nwb.subject.subject_id`. After session processing, the script builds a sorted unique `subjects` list and a per-session `subject_idx`.

ii.
```python
subject_id = nwb.subject.subject_id
```

```python
subjects = sorted(set(s['subject_id'] for s in all_sessions))
subj_idx.append(subjects.index(s['subject_id']))
```

iii. `CONVERSION_NOTES.md` states that subjects are the 28 mice in the NWB release. The trajectory shows the agent reading `nwb.subject.subject_id` while exploring the first NWB file and later using those IDs to organize the converted output.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `process_session()` returns one session dictionary, and `main()` appends each successful return value to `all_sessions`.

ii.
```python
def process_session(nwb_path, t_start=-2.5, t_end=1.5, bin_size=0.05,
                   show_processing=False, session_idx=0):
```

```python
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE,
                       show_processing=args.show_processing and i < 2, session_idx=i)
    if r: all_sessions.append(r)
    else: skipped += 1
```

iii. In the notes, the agent describes the NWB organization as one file per session (`sub-{id}_ses-{datetime}_...nwb`). In the trajectory, it repeatedly refers to the 174 NWB files as the session list.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The script reads per-trial arrays from `nwb.trials`, computes which trial indices were actually recorded from `obs_intervals`, applies the regular-trial mask, and keeps the intersection as `valid_trials`.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
```

```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The notes say `obs_intervals` define which trials have neural recordings, and the trajectory shows the agent debugging a time-base mismatch before concluding that only a subset of trials was recorded in some sessions and that `obs_intervals` had to be used.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with the reference-style "regular trial" mask: no early lick, no auto water, no free water, no ignore/no-response outcome, and no photostimulation. The script also excludes trials outside the neural recording intervals and skips sessions with fewer than two valid trials.

ii.
```python
mask_no_early = (early_lick == 'no early')
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
mask_no_photostim = ~has_photostim
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
```

```python
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
```

iii. `CONVERSION_NOTES.md` explicitly says the trial curation is an "Exact match with get_regular_trial_mask logic." The trajectory captures the agent reading `functions_for_r2.py`, where `get_regular_trial_mask` was defined with the same five filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `nwb.units['spike_times']` after restricting units to `classification == 'good'`. Trial selection additionally depends on `nwb.units['obs_intervals']`.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
```

iii. The notes say neuron curation is `classification == 'good'` to mirror the classifier-based QC in the reference materials. The trajectory shows the agent first exploring the NWB `classification` field and then mapping it onto the reference QC mode.

## 2-b. How is the `neural` data processed?

i. The script converts spike times into per-trial firing rates by histogramming spikes into 50 ms bins over `[-2.5, 1.5]` s around go cue and dividing counts by bin width to get Hz.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    ...
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        ...
        counts, _ = np.histogram(st_w, bins=bin_edges)
        fr[i] = counts.astype(np.float32) / bin_size
```

```python
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

iii. In the notes, the agent says it reused the reference idea of spike binning (`sliding_histogram`) but changed the bin width to 50 ms because the task instructions explicitly requested that decoder resolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered in two ways: only units labeled `good` are kept, and only trials with neural recording coverage according to `obs_intervals` are retained. Sessions with zero good units are skipped.

ii.
```python
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The notes justify `classification == 'good'` as the NWB equivalent of the reference classifier-based QC. The trajectory also shows the agent discovering that some sessions recorded only a subset of trials and adding the `obs_intervals` logic.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue onset (`go_start_times`) and extracted from 2.5 s before to 1.5 s after the go cue.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

iii. The notes repeatedly say "Temporal alignment: Go cue onset" and that the chosen window is `[-2.5, 1.5] s` because that is the decoder task specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins. The code directly bins spikes at 50 ms; there is no later rebinning or interpolation step.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
```

```python
n_bins = int(round((t_end - t_start) / bin_size))
counts, _ = np.histogram(st_w, bins=bin_edges)
fr[i] = counts.astype(np.float32) / bin_size
```

iii. The notes explicitly contrast the 50 ms decoder bins with the 40 ms reference preprocessing bandwidth and say the task specification drove that choice.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` in `BehavioralEvents`, together with `trial start/stop` and `go_start_times`.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes map "time from sample_start" to `input[0]: time_from_tone_onset`. The trajectory shows the agent deciding that the first sample-start event in a trial is the first tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each valid trial, the script finds the first `sample_start_time` that falls within the trial bounds, converts it to time relative to go cue, and subtracts that offset from each neural bin center. If no sample start is found, it fills the vector with `NaN`.

ii.
```python
def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The notes say this variable is "Continuous seconds" and the trajectory records the agent's reasoning that trials can have multiple sample starts because of replays, so it should use the first one within the trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 bin centers as the neural firing rates, with values expressed on the go-cue-aligned time axis.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
...
tft = bin_centers - (ss - go)
input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
```

iii. The notes say the input is time-varying and aligned to go cue, and the trajectory includes a manual sanity check verifying that a specific converted bin matched the expected value from raw NWB times.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the trials-table photostim fields: `photostim_power`, `photostim_onset`, and `photostim_duration`, together with `trial start` and `go cue` times.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

iii. The notes map "photostim active" to `input[1]: photostim_on`. The trajectory shows the agent inspecting photostim fields in photostim-capable sessions and deciding they are stored as strings that must be converted.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code first flags photostim trials from non-`N/A` positive `photostim_power`, then converts onset from trial-start-relative time to go-cue-relative time, and creates a binary per-bin vector marking stimulation on/off. In the final dataset, however, this input is effectively all zeros because all photostim trials are filtered out before saving.

ii.
```python
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
...
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

```python
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The notes acknowledge both choices: they say trial filtering excludes photostim trials, and later explicitly note that `photostim_on` is always zero because of that filter. The trajectory shows the agent debating whether to keep stim trials for the decoder, then deciding to keep the reference regular-trial mask.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same go-cue-centered bin grid as the neural data. The onset is shifted from trial-start-relative coordinates into go-cue-relative coordinates before binning.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
d = float(photostim_dur_raw[trial_idx])
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes specifically justify this as an edge-case fix: "Photostim onset: Correctly converted from trial-start-relative to go-cue-relative." The trajectory shows this conversion being added after inspecting the NWB timing fields.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. It is derived from `nwb.trials['trial_instruction']`.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The notes explicitly map `trial_instruction` to `choice`, with `left=0` and `right=1`. The trajectory likewise maps NWB `trial_instruction` onto the reference `trial_type`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `right` to `1` and all other kept trials (`left`) to `0`, then broadcasts the per-trial value across all 80 time bins.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
...
out = np.zeros((4, n_bins), dtype=np.int64)
out[0, :] = choice
```

iii. The notes describe choice as a per-trial output. The trajectory includes spot checks against raw NWB trials to verify that converted choice values matched the original labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `nwb.trials['outcome']`.

ii.
```python
outcome = nwb.trials['outcome'][:]
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. The notes explicitly map NWB `outcome` to the decoder's outcome classes, and the trajectory maps NWB `hit`/`miss`/`ignore` onto the reference `correctness` coding.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then broadcasts the value across all bins. Because the regular-trial mask excludes `ignore` trials, the saved output never actually contains class `0`.

ii.
```python
mask_no_ignore = (outcome != 'ignore')
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
```

```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
out[1, :] = out_val
```

iii. The notes say trial filtering removes ignore trials and later observe that the outcome output is nontrivial only for miss/hit after filtering. The trajectory shows the agent noticing this and still choosing to preserve the reference trial mask.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived from `nwb.trials['early_lick']`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The notes map `early_lick` to the `no/yes` output, and the trajectory explicitly identifies NWB `early` versus `no early` as the source field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `early -> 1` and `no early -> 0`, then broadcasts the value across time bins. In the saved dataset, all values are `0` because the trial filter excludes all early-lick trials.

ii.
```python
mask_no_early = (early_lick == 'no early')
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
```

```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
out[2, :] = early_val
```

iii. The notes say early-lick trials are removed by `get_regular_trial_mask`, and Step 12 explicitly calls the resulting `early_lick` output "trivial (filtered)." The trajectory shows the agent noticing this issue but deciding to keep the reference mask.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps, the second data column (`y`), and the third column (DLC likelihood/confidence).

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

iii. The notes say the tongue output comes from `Camera0_side_TongueTracking y-coordinate` with a likelihood threshold, and the trajectory shows the agent exploring the tracking array and inferring the `(x, y, likelihood)` column convention.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script first collects high-confidence tongue y samples (`likelihood > 0.9`) from each valid trial's aligned window and computes two session-level thresholds from that pool. Then, for each trial and each 50 ms bin, it averages high-confidence y samples in that bin and converts the mean to a category. If there is no high-confidence sample, it leaves the bin at the default middle class.

ii.
```python
for ti in valid_trials:
    go = go_times[ti]
    i_lo = np.searchsorted(tongue_ts, go + t_start)
    i_hi = np.searchsorted(tongue_ts, go + t_end)
    if i_hi > i_lo:
        lk = tongue_lk_arr[i_lo:i_hi]
        good = lk > 0.9
        if np.any(good):
            all_ty.append(tongue_y_arr[i_lo:i_hi][good])
```

```python
ty = np.ones(n_bins, dtype=np.int64)
...
lk = tongue_lk_arr[il:ih]
gd = lk > 0.9
if np.any(gd):
    my = np.mean(tongue_y_arr[il:ih][gd])
    ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The notes justify the likelihood threshold and the "default to class 1 (mid)" policy for missing low-confidence bins. The trajectory shows the agent exploring the DLC likelihood distribution before settling on `> 0.9`.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code computes the 40th and 60th percentiles from the concatenated high-confidence tongue y values gathered from valid, aligned trials. Bins below `p40` become class `0`, above `p60` become class `2`, and the remainder become class `1`.

ii.
```python
concat = np.concatenate(all_ty)
tongue_y_p40 = np.percentile(concat, 40)
tongue_y_p60 = np.percentile(concat, 60)
```

```python
ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The notes say this output should be `0/<40th, 1/40-60th, 2/>60th pctl`. The trajectory and notes do not claim whole-session percentiles; instead they justify the implemented version as a per-session summary over the valid aligned windows.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin, the code converts the bin center to absolute session time (`go + bin_center`) and looks up tongue frames within `± half_bin` around that time.

ii.
```python
half_bin = bin_size / 2.0
...
bc = go + bin_centers[b]
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The notes describe tongue y as a time-varying output aligned to go cue, and the trajectory includes explicit checks that the tongue timestamps and neural alignment window were on the same trial-relative axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses a set of ad hoc fallbacks. Missing sample starts produce an all-`NaN` time-from-tone input. Missing tongue data, missing tongue thresholds, or bins without high-confidence tongue detections default to tongue class `1`. Sessions with no good neurons or fewer than two valid recorded trials are skipped. Recorded trials are inferred heuristically from the first good unit's `obs_intervals`.

ii.
```python
return sample_start_times[mask][0] if np.any(mask) else None
...
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

```python
ty = np.ones(n_bins, dtype=np.int64)
if tongue_ts is not None and tongue_y_p40 is not None:
    ...
```

```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
...
if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
```

iii. The notes explicitly justify the tongue fallback as "default to class 1 (mid) when no high-confidence detection." The trajectory also documents the `obs_intervals` debugging and the choice to accept one all-zero neural trial at a recording boundary instead of dropping it.

## 10-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are per-trial spike binning across all neurons, per-bin tongue alignment/categorization, and loading the large spike-time/tongue arrays for each session.

ii.
```python
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

```python
for t_idx, trial_idx in enumerate(valid_trials):
    ...
    for b in range(n_bins):
        bc = go + bin_centers[b]
        il = np.searchsorted(tongue_ts, bc - half_bin)
        ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The notes say processing took about 10 s/session and 30 minutes total, and the trajectory logs show the slowest sessions were the ones with the most neurons and valid trials.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are the nested `trial x neuron` loop in `bin_spikes_all_trials`, the per-bin tongue loop inside each trial, and some per-trial searches such as repeated sample-start and recorded-trial construction.

ii.
```python
for go in go_times_arr:
    ...
    for i, st in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(st_w, bins=bin_edges)
```

```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

```python
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

iii. The notes claim the code uses `np.histogram` and `np.searchsorted` for efficiency, but the trajectory also shows the full conversion still taking about 30 minutes, so these loops were only partially optimized.

## 10-c. What processing does the code repeat multiple times?

i. It repeats tongue time-window searches twice: once to build the percentile pool and again for every bin of every valid trial. It also repeatedly masks each neuron's spike train for every trial window instead of reusing pre-segmented trial chunks. Session performance statistics are also recomputed even though they are only logged/stored.

ii.
```python
for ti in valid_trials:
    go = go_times[ti]
    i_lo = np.searchsorted(tongue_ts, go + t_start)
    i_hi = np.searchsorted(tongue_ts, go + t_end)
```

```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

```python
for go in go_times_arr:
    ...
    mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
```

iii. The notes mention both the percentile pass and the later per-bin tongue pass, and the trajectory contains timing logs showing these repeated searches dominated the slower sessions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes session performance and correct-left/right counts only for logging/metadata, constructs a photostim input even though all photostim trials are filtered out, and produces trivial `early_lick` and `ignore` outcome categories because those trials are removed before conversion. The optional plotting code is also purely diagnostic.

ii.
```python
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
...
print(f"  Performance: {perf:.1%} ({n_hit}/{n_hit+n_miss}), Correct L={correct_left}, R={correct_right}")
```

```python
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
...
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    ...
```

```python
if show_processing:
    try:
        plot_processing(...)
```

iii. The notes explicitly acknowledge that `photostim_on` is always zero and `early_lick` is trivial because of trial filtering. The trajectory also shows the agent removing session-level filtering but keeping these now-downstream-trivial computations to stay close to the reference trial mask.
