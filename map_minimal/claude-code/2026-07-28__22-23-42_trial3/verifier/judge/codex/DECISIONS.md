# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script finds all `.nwb` files with a relative glob, sorts them, then opens each session file directly with `h5py`. Within each file it reads trial-table datasets from `intervals/trials/*`, event timestamps from `acquisition/BehavioralEvents/*`, unit metadata from `units/*`, and tongue tracking from `acquisition/BehavioralTimeSeries/*`.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
f = h5py.File(nwb_path, 'r')
...
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
```

iii. In the trajectory, the agent explicitly decided to work from the 174 NWB files under `data/sub-*` and used an NWB-structure exploration subagent before writing a direct `h5py` loader. It justified this as matching the published file structure while giving explicit access to spike times, trials, events, and video streams.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB filename, not from the NWB subject metadata. The script strips the `sub-` prefix and `_ses-...` suffix from each session filename, then deduplicates and sorts those IDs to build `subjects` and `subject_idx`.

ii. 
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The trajectory shows the agent inspected the folder layout first and treated the `sub-<id>` / `sub-<id>_ses-...` naming as the subject source. There is no sign it used `nwb.subject.subject_id`; its stated rationale was that the file naming already exposed the subject identity.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order follows the sorted file list, and session identity in metadata is stored as the filename (`session_file`) rather than `nwb.identifier`.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'subject_id': subject_id,
    'unit_regions': unit_regions,
    'session_file': basename,
}
```

iii. In the trajectory the agent repeatedly refers to “174 NWB files” and processes them one by one, so the file boundary was its session boundary from the start. The justification was implicit: the dataset layout already separates sessions into individual NWB files.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table row indices. The script asserts that the number of go-cue timestamps matches `len(intervals/trials/id)`, then selects a subset of trial indices with a boolean mask and iterates over those indices as trials.

ii. 
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials
...
trial_indices = np.where(trial_mask)[0]
for ti in trial_indices:
    gc = go_cue_times[ti]
```

iii. The trajectory shows the agent explored the NWB structure specifically to confirm where trial data and go-cue timings lived. It then treated the trial table as canonical and used the 1:1 go-cue count check as its validation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: `auto_water == 0`, `free_water == 0`, and `recording_valid`. `recording_valid` is computed from the latest common observation window across all kept units, using a trial’s go cue plus the requested `[-2.5, 1.5]` window. The script also drops any session with fewer than 2 surviving trials.

ii. 
```python
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
...
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
trial_indices = np.where(trial_mask)[0]
if len(trial_indices) < 2:
    return None
```

iii. The trajectory shows the agent initially planned to exclude early-lick and ignore trials because of the methods text, then reversed that choice because the decoder outputs required those labels. It also investigated “all-zero neural data” and concluded late trials were outside the recording window, which motivated the `recording_valid` filter based on observation intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index`, with `go_start_times/timestamps` providing alignment times. Unit inclusion additionally depends on `units/classification` and successful mapping of `units/anno_name` to one of the AI’s hand-built region groups.

ii. 
```python
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
anno_names = np.array([a.decode() if isinstance(a, bytes) else a
                      for a in f['units/anno_name'][:]])
...
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
```

iii. The trajectory explicitly says the agent would “load spike times from NWB units table” and “filter by `classification == 'good'`.” It also spent time constructing a manual high-level brain-region mapping from `anno_name`.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each kept unit, absolute spike times are shifted by subtracting the trial’s go-cue time, cropped to `[-2.5, 1.5)`, histogrammed into 50 ms bins, and divided by bin width to yield firing rates in Hz. No smoothing or normalization is applied.

ii. 
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    return counts.astype(np.float64) / (bin_edges[1] - bin_edges[0])
...
for j, st in enumerate(unit_spike_times):
    fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The trajectory states the agent’s neural plan directly: “compute firing rates with 50 ms bins aligned to go cue, window -2.5 s to +1.5 s” and “count spikes in each bin and divide by the bin width.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered to `classification == 'good'`. After that, the AI applies an additional filter: only good units whose `anno_name` can be mapped into one of its 14 manually defined broad region categories are kept. A session is dropped if no such units remain.

ii. 
```python
good_mask = classification == 'good'
...
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)

if len(unit_indices) == 0:
    return None
```

iii. The trajectory clearly records both parts of the decision. The agent said it would use classifier-based QC and later described building a manual mapping from detailed anatomy names to 14 categories, treating only mapped units as valid for the output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. The script takes each trial’s absolute go-cue timestamp, subtracts it from absolute spike times to get go-cue-relative spike times, and bins those relative times on a fixed `[-2.5, 1.5]` grid.

ii. 
```python
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
gc = go_cue_times[ti]
fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The agent’s plan in the trajectory repeatedly centers everything on the go cue because that was the explicit decoder instruction. It did not describe any additional clock-offset correction between streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script uses 80 non-overlapping 50 ms bins covering `-2.5 s` to `+1.5 s` relative to go cue. There is no secondary temporal rebinning or smoothing.

ii. 
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The trajectory directly cites the task requirement of a 4 s window and 50 ms bins, then notes that this yields 80 time bins per trial.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps` and the trial’s `go_start_times/timestamps`. The script chooses the last sample-start event before the trial’s go cue.

ii. 
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time
```

iii. The trajectory explicitly reasons through the task timing and states that because early licks replay the sample epoch, “the last `sample_start` before go is the correct tone onset.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial the AI computes `tone_onset_rel = last_sample_start - go_cue`, then sets the time-varying input to `BIN_CENTERS - tone_onset_rel`, i.e. elapsed time since tone onset at each go-cue-relative bin center. If no prior sample-start is found, it falls back to `-1.85` s.

ii. 
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The trajectory contains the same derivation in prose, including the fixed 1.85 s nominal gap from tone onset to go cue and the idea that replayed epochs can make that interval larger. The fallback value is not separately justified in the trajectory.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same bin centers as the neural data. The scalar offset from tone to go is converted into a value at each go-cue-relative bin center, so each timepoint aligns one-to-one with the neural bins.

ii. 
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - tone_onset_rel
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)
```

iii. The trajectory explains this alignment explicitly: once the data are go-cue aligned, time from tone onset at time `t` is `t - tone_onset_rel`, evaluated on the same time grid as the neural features.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `Photostimulation` is derived from the trial-table fields `photostim_onset`, `photostim_duration`, and `start_time`, plus `go_start_times/timestamps` to express stimulation timing relative to the neural alignment event.

ii. 
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The trajectory notes that photostimulation must become a binary time series and that onsets are relative to trial start, so they need to be repositioned onto the go-cue-aligned time axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stimulated trials, onset and offset are computed in go-cue-relative seconds, then each 50 ms bin center is labeled `1` if it lies within `[stim_start, stim_end)` and `0` otherwise. Trials with `photostim_onset == 'N/A'` remain all zeros.

ii. 
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
if ps_onset_str[ti] != 'N/A':
    ps_onset_val = float(ps_onset_str[ti])
    ps_dur = float(ps_dur_str[ti])
    ps_abs_onset = trial_start_times[ti] + ps_onset_val
    ps_start_rel = ps_abs_onset - gc
    ps_end_rel = ps_start_rel + ps_dur
    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The trajectory’s stated decoder-input plan was “photostimulation on/off (binary).” The handling of `'N/A'` as all-zero comes from the implementation; there is no separate textual justification beyond that.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The light-on interval is converted into go-cue-relative time and then compared against the same `BIN_CENTERS` used for neural and other input time series, so photostimulation is aligned bin-by-bin with neural activity.

ii. 
```python
ps_abs_onset = trial_start_times[ti] + ps_onset_val
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The trajectory explicitly reasoned that all decoder inputs had to be represented on the go-cue-aligned grid. No alternative alignment scheme was discussed.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, `choice` is derived only from `trial_instruction`. Left-instructed trials are labeled `0`, right-instructed trials are labeled `1`; the `outcome` column is not used for this output.

ii. 
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
...
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. The trajectory shows the agent originally understood that choice should be an output and recognized the tension with ignore trials, but the final implementation drifted. There is no explicit later justification for dropping the outcome-dependent derivation or the no-lick class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script maps `left -> 0` and `right -> 1`, stores the result as a per-trial scalar, and then broadcasts it across all 80 bins in the output array. It does not encode a separate no-lick choice class.

ii. 
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
...
np.full(N_BINS, output_choice[i], dtype=np.int64)
```

iii. The trajectory does not contain a direct justification for the final two-class implementation. Earlier reasoning suggests the agent knew ignore trials needed to be retained, which makes this final processing choice inconsistent with its earlier stated intent.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` comes directly from `intervals/trials/outcome`.

ii. 
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The trajectory lists outcome as one of the required decoder outputs and treats the trial-table outcome strings as the raw source variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code converts outcome strings into integers: `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats the per-trial category across all 80 bins.

ii. 
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:
    outcome = 2
...
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. The trajectory matches this mapping exactly when listing the decoder outputs to create.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is derived directly from `intervals/trials/early_lick`.

ii. 
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The trajectory explicitly called out `early_lick` as a required output and discussed not excluding those trials, so the raw trial-table field is the source it chose.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script converts `early -> 1` and everything else (specifically `no early`) to `0`, then broadcasts that per-trial label across the 80 bins.

ii. 
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
...
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. The trajectory lists the target mapping as “Early lick (no=0, yes=1),” which the final code implements in a direct if/else form.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The code derives tongue position from the side-camera tongue-tracking timestamps and the second column of the `data` array (`tongue_y`). It does not use the tracking-likelihood column in the final implementation.

ii. 
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]
```

iii. The trajectory shows the agent knew the raw series included x, y, and likelihood and initially discussed likelihood-thresholding, but the final script dropped that column from the computation. The later trajectory text acknowledges a problem caused by how missing/low-confidence tongue data were being handled.

## 8-b. How is `output` *Tongue y-position* processed?

i. For each kept trial, the code extracts all tongue `y` samples whose timestamps fall inside the `[-2.5, 1.5]` window around go cue and averages them within each 50 ms bin. After all trials are processed, it concatenates all non-NaN per-bin means across the session, computes the 40th and 60th percentiles, and uses those thresholds to discretize each trial’s binned tongue trace.

ii. 
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]
    ...
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
...
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)
```

iii. The trajectory first planned to apply a tongue-likelihood threshold, then later noted that defaulting missing tongue values to the middle class was creating a distorted distribution. That later reflection is the clearest justification available for the final implementation choices: the agent was trying to produce session-level percentile bins from observed tongue traces but handled missingness poorly.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The script uses three categories only: below the 40th percentile (`0`), between the 40th and 60th percentiles inclusive (`1`), and above the 60th percentile (`2`). Missing bins are initialized to the middle class (`1`), and if a session has no valid tongue data at all, every bin in every trial becomes the middle class.

ii. 
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
valid_mask = ~np.isnan(ty)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
...
tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The later trajectory explicitly comments that this default-to-middle behavior heavily skewed the tongue distribution and might be wrong. That is the agent’s own retrospective justification that the final thresholding scheme was a compromise rather than a well-settled choice.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data are aligned by selecting camera samples in the absolute time window from `go + T_START` to `go + T_END`, subtracting the trial’s go cue to get relative times, and averaging those samples inside the same 50 ms bins used for neural data.

ii. 
```python
t_abs_start = go_cue_time + bin_edges[0]
t_abs_end = go_cue_time + bin_edges[-1]
idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
...
ts_rel = ts_window - go_cue_time
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. The trajectory consistently describes a single go-cue-centered time axis for all modalities. The tongue extractor follows that plan directly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several fallback behaviors. Missing tone onset falls back to `-1.85` s. Non-numeric photostimulation fields are ignored via `try/except`, leaving the photostim trace all zeros. Bins with no tongue samples remain `NaN` during averaging but are later converted to the middle category, and an entire session with no valid tongue data is filled with the middle category. Sessions with no mapped good units, no responding control trials, or too few valid trials are dropped.

ii. 
```python
if tone_onset_rel is None:
    tone_onset_rel = -1.85
...
try:
    ps_onset_val = float(ps_onset_str[ti])
    ...
except (ValueError, TypeError):
    pass
...
ty_disc = np.ones(N_BINS, dtype=np.int64)
...
if len(valid_tongue_y) == 0:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]
```

iii. The trajectory shows explicit concern about two of these cases. The agent reasoned from the task timing to justify the `-1.85` fallback for missing tone onset, and later recognized that collapsing missing tongue data into the middle class was problematic. The remaining fallbacks are implementation-level choices without separate textual defense.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI code is reading large NWB arrays (`spike_times`, tongue tracking, trials/units tables), then running nested loops over trials and units to histogram spikes for every trial-unit pair. Trial-wise tongue binning is another substantial per-session cost.

ii. 
```python
spike_times_data = f['units/spike_times'][:]
...
for ti in trial_indices:
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
...
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. The trajectory records a sample run and a full run, then highlights the neural-zero investigation and the much smaller retained dataset after filtering. It does not explicitly profile the code, but its investigation focused on the neural histogramming path and session-wide file processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the nested per-trial/per-unit spike histogram loop, the per-trial scan through all `sample_start_times` to find the last tone onset, the per-bin loop inside `extract_tongue_y_for_trial`, and the repeated list-building loops for region assignment and spike-time extraction.

ii. 
```python
for ti in trial_indices:
    ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
...
before = sample_start_times[sample_start_times < go_cue_time]
...
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. The trajectory does not explicitly discuss vectorization, but it does show the agent troubleshooting runtime-relevant neural loops and using a simpler, more direct implementation rather than the reference solution’s flatter `searchsorted` approach.

## 10-c. What processing does the code repeat multiple times?

i. Several computations are repeated across trials even though they could be shared or batched: scanning `sample_start_times` anew for each trial, recomputing spike histograms unit-by-unit inside every trial, and independently computing trial-level tongue bin means before a second pass discretizes them using session percentiles.

ii. 
```python
for ti in trial_indices:
    ...
    tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
    ...
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
...
for ty in tongue_y_all_trials:
    ty_disc = np.ones(N_BINS, dtype=np.int64)
```

iii. There is no explicit trajectory statement defending these repeated computations. They are a consequence of the straightforward trial-by-trial implementation the agent settled on.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does several pieces of processing that are not needed for the decoder arrays themselves: it constructs a large manual 14-region mapping solely to coarsen `anno_name`, computes control-trial session-performance statistics only to exclude sessions, prints summary region distributions, and stores `session_files`/summary metadata that are not used by the decoder. It also computes broad-region membership as a gate on unit inclusion rather than just descriptive metadata.

ii. 
```python
REGION_MAPPING = {
    'ALM': ['Secondary motor area'],
    ...
}
...
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
...
region_counts = {}
for regions in all_unit_regions:
    for r in regions:
        region_counts[r] = region_counts.get(r, 0) + 1
```

iii. The trajectory explicitly shows the agent spending time building the 14-category region mapping and applying paper-level session filters because it believed those were part of the required processing. It does not justify the extra summary reporting as necessary for downstream use.
