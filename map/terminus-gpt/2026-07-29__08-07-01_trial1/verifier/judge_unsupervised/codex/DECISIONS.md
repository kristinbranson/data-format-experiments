# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`. In normal mode it processes all files; in sample mode it keeps only the first two sorted files.

ii. 
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)

for fp in files:
    sess = load_session(fp, brain_regions)
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented the dataset as per-subject NWB directories and counted all NWB sessions. In the trajectory, it described the plan as loading NWB sessions directly from `data/` and later noted that this yielded 174 files, one more than the 173 sessions quoted in the papers.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the parent directory name of each NWB file, e.g. `sub-440956`. A `subject_map` assigns each unique subject string to an integer index, and `subject_idx` stores one subject index per session.

ii. 
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name

subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_map[subj])
```

iii. The notes in Step 2 say the data are organized as per-subject directories under `data/`. The agent therefore used the folder name as the subject identifier rather than reading a subject field from NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session identifier is the file stem, and each successfully converted file contributes one entry to `data['neural']`, `data['input']`, `data['output']`, and `data['brain_region_idx']`.

ii. 
```python
def get_session_id(nwb_path):
    return nwb_path.stem

return {
    'session_id': get_session_id(nwb_path),
    'subject': get_subject_id(nwb_path),
    ...
}

for fp in files:
    sess = load_session(fp, brain_regions)
    ...
    data['neural'].append(sess['neural'])
```

iii. In Step 2 the agent documented NWB filenames like `sub-<mouse>_ses-<timestamp>_behavior+ecephys+ogen.nwb`, and in the trajectory it repeatedly referred to “one NWB session file” as the unit of conversion.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `intervals/trials` table. The code takes `n_trials = trials['id'].shape[0]`, pairs those rows with `go_start_times` by shared index up to `min(n_trials, len(go_times))`, and iterates over trial index `i`. Trial-level labels and start/stop times come from row `i`.

ii. 
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
n_match = min(n_trials, len(go_times))

for i in range(n_match):
    go = go_times[i]
    ...
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
```

iii. The trajectory shows that the agent discovered `sample_start_times` was a global event stream rather than a per-trial array, but it continued to assume `go_start_times` was effectively trial-aligned by index. The notes describe the trial table as the native source of per-trial metadata.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by basic validity and alignment checks: finite go cue, full `[-2.5, 1.5]` s window contained inside the trial, valid categorical labels for choice/outcome/early lick, a detected sample event before go within the trial, and at least two kept trials in the session. The script does not apply the “regular trial” mask described in the reference notes, even though it loads `auto_water` and `free_water`.

ii. 
```python
if not np.isfinite(go):
    continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
if not np.isfinite(sample_time):
    continue
...
if len(sess_neural) < 2:
    return None
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 3, the agent explicitly recorded that the reference code’s `get_regular_trial_mask` excludes early-lick, auto-water, free-water, and no-response trials. The later conversion script does not implement that rule; the notes justify trial inclusion mainly as “valid go cue and sufficient aligned data.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from unit spike times, with unit selection based on `units/unit_quality`, `units/electrodes`, and electrode `location` strings that encode `brain_regions`.

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

iii. The notes’ Step 5 mapping table says `neural` comes from `units/spike_times` for curated units, with region labels mapped through the electrode table. The trajectory also shows the agent inspecting `units/unit_quality` and electrode `location` JSON fields before deciding how to build neural arrays.

## 2-b. How is the `neural` data processed?

i. For each kept unit and kept trial, spikes are cropped to the go-cue-centered window, binned into 50 ms bins, and divided by bin size to produce firing rates in Hz. Trial neural data are stacked into a `(n_neurons, 80)` float32 matrix.

ii. 
```python
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    ...
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE

neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. In Step 5 the agent planned to “bin spikes in 50 ms bins from -2.5 s to +1.5 s around each trial’s go cue” and in Step 6 it documented that the script performs “50 ms spike binning” and constructs neuron-by-time arrays.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `unit_quality == 'good'` and their broad region string is one of ten hard-coded “major regions” (left/right ALM, Striatum, Thalamus, Midbrain, Medulla). No additional trial-count, classifier-score, or session exclusion rules are applied in the code.

ii. 
```python
MAJOR_REGIONS = {
    'left ALM','right ALM','left Striatum','right Striatum',
    'left Thalamus','right Thalamus','left Midbrain','right Midbrain',
    'left Medulla','right Medulla'
}
...
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```

iii. The notes’ Step 3 says the paper uses “good” units from region-specific QC classifiers. In Step 4, the agent saw that raw `unit_quality == "good"` gave far more units than the paper’s 69,943 and justified the extra region filter as a way to approximate the paper’s emphasized broad regions, while also admitting the discrepancy remained unresolved.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue onset. For each trial, the script uses that trial’s `go_start_times` timestamp as time zero and extracts spikes from `go_time - 2.5 s` through `go_time + 1.5 s`.

ii. 
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
...
go = go_times[i]
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The task instructions explicitly required go-cue alignment, and the notes repeatedly state that the reference code used go-cue-centered alignment for neural and marker data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins, yielding 80 bins over the 4 s window. No additional temporal rebinning is applied after this initial binning.

ii. 
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The task instructions specified 50-ms-width bins, and the notes’ Step 5 mapping and Step 6 implementation notes say the agent followed that requirement directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from the `sample_start_times` event stream together with the per-trial go cue time. For each trial, the agent chooses the last sample-start event within the trial and before go cue.

ii. 
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
sample_time = find_last_event_within_trial(
    sample_event_times, trial_start[i], trial_stop[i], before_time=go
)
inp0 = build_time_from_tone(sample_time, go)
```

iii. In the trajectory, the agent first tried indexing `sample_start_times` by trial and then corrected that after discovering it was a global event stream. Step 10 of the notes says the time-from-tone bug was fixed by selecting “the last sample event within each trial before go cue.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code computes a continuous time-since-tone ramp on the shared 50 ms bin centers. It converts sample onset into go-relative time, subtracts that from each aligned bin center, and clips all pre-tone bins to 0.

ii. 
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The trajectory records an explicit ambiguity: the instructions called this input “continuous, time-varying,” while another formatting note suggested event times could be binary. The agent chose the continuous ramp interpretation and justified it as “time since tone onset within the aligned window.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned on the exact same go-cue-centered 50 ms bin centers as the neural data, so each trial’s first input row has length 80 and is time-locked to the same bins as that trial’s firing-rate matrix.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
inp0 = build_time_from_tone(sample_time, go)
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. The notes’ Step 5 says all modalities would be aligned to go cue and extracted on a shared time base. The helper function uses `BIN_CENTERS`, which are also the basis for spike bins and tongue interpolation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `photostim_start_times/timestamps` and `photostim_stop_times/timestamps`.

ii. 
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
```

iii. The Step 5 mapping table in the notes lists `photostim_start_times` and `photostim_stop_times` as the chosen source variables, with the trial table’s `photostim_onset` and `photostim_duration` treated only as cross-check information.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the script selects photostim start and stop events whose timestamps fall inside that trial window, truncates to the shorter of the start/stop arrays if they are unmatched, and marks each 50 ms bin as 1 if any part of the bin overlaps a photostim interval.

ii. 
```python
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

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

iii. The notes’ Step 4 says the methods state photostimulation ended before go cue, and the agent planned to “build photostim input relative to go cue and verify all stimulation bins are pre-go.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is converted into the same 80 go-cue-centered 50 ms bins used for neural activity. Start and stop times are converted into go-relative coordinates before bin overlap is computed.

ii. 
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. The justification in the notes is the same shared-time-base rule used elsewhere: all modalities are aligned to go cue and represented on the same bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the trial-table field `trial_instruction`.

ii. 
```python
choice = decode_arr(trials['trial_instruction'][:])
```

iii. The Step 5 mapping table in the notes explicitly maps `trial_instruction` to the decoder’s “choice” output.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. String values are decoded from bytes, mapped with `{'left': 0, 'right': 1}`, and expanded into a constant 80-bin row for each kept trial.

ii. 
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. The notes say this output should be per-trial categorical choice. The agent chose to store it as a time-varying constant row rather than a scalar so it would match the decoder’s per-timepoint format cleanly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-table field `outcome`.

ii. 
```python
outcome = decode_arr(trials['outcome'][:])
```

iii. The notes’ Step 5 mapping table explicitly maps `outcome` to the decoder’s outcome output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are decoded and mapped with `{'ignore': 0, 'miss': 1, 'hit': 2}`, then repeated across all 80 bins of the trial.

ii. 
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. The agent followed the decoder task’s explicit category mapping and kept the variable constant across the aligned trial window.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trial-table field `early_lick`.

ii. 
```python
early = decode_arr(trials['early_lick'][:])
```

iii. The notes’ Step 5 mapping table and Step 3 curation notes both identify `early_lick` as the relevant raw variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Early-lick strings are decoded and mapped with `{'no early': 0, 'early': 1}`, then repeated across the full 80-bin trial window as a constant categorical row.

ii. 
```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. The agent treated early lick as a decoder output rather than only a curation flag, because the task instructions explicitly asked for it as an output variable even though the reference code’s “regular trial” mask excludes such trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its timestamps. The code assumes column 1 of the tracking matrix is the y coordinate.

ii. 
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. In the trajectory, the agent inspected the `Camera0_side_TongueTracking` array and described it as “likely x/y/likelihood.” The notes’ Step 5 mapping table records the decision to use `data[:,1]` as tongue y.

## 8-b. How is `output` *Tongue y-position* processed?

i. The code takes the full session tongue-y trace, interpolates it onto the 80 aligned bin centers separately for each kept trial, collects all finite aligned tongue values across kept trials in that session, and later discretizes each trial’s aligned tongue trace into three categories.

ii. 
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)

ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
sess_tongue_cont.append(ty)
```

iii. The notes’ Step 5 mapping plan says tongue y should be “interpolate/aligned tongue y to trial bins; discretize per session by 40th/60th percentiles.” There is no recorded justification for ignoring the third tracking column or any likelihood thresholding; that was an implicit assumption.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After collecting finite aligned tongue-y values from all kept trials in a session, the script computes the 40th and 60th percentiles and assigns each aligned time bin to low, mid, or high. Bins with `NaN` tongue values remain in the default middle category because `disc` starts at 1.

ii. 
```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()]) \
    if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
q40, q60 = np.percentile(all_tongue, [40, 60])
...
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. The notes’ Step 5 mapping table says the task required session-wise percentile discretization, and the agent implemented that over the concatenated aligned values from kept trials. The fallback array `[0,1]` and the implicit `NaN -> 1` behavior are not separately justified in the notes.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. It is aligned to go cue and sampled on the same 80 bin centers as the neural activity. The tongue trace is converted to go-relative time and interpolated onto `BIN_CENTERS`.

ii. 
```python
rel_t = track_t - go_time
...
return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

iii. The notes’ Step 1 and Step 5 both emphasize that the reference workflow aligned marker/tracking streams to go cue, so the agent mirrored that alignment strategy for tongue y.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles minor data issues mostly by skipping or defaulting. Trials are dropped if go time is non-finite, the aligned window extends outside the trial, labels are unmapped, or no within-trial sample event is found. Tracking with fewer than two valid points returns all-`NaN`; sessions with no finite tongue data use `[0,1]` as a fake fallback for percentile computation; `NaN` tongue bins default to the middle category; mismatched photostim start/stop counts are truncated to the shorter length; sessions with fewer than two kept trials are discarded.

ii. 
```python
if valid.sum() < 2:
    return np.full(N_BINS, np.nan, dtype=np.float32)
...
m = min(len(ps), len(pe))
...
if not np.isfinite(sample_time):
    continue
...
all_tongue = ... if any(...) else np.array([0,1], dtype=np.float32)
disc = np.full(N_BINS, 1, dtype=np.int64)
...
if len(sess_neural) < 2:
    return None
```

iii. Step 10 of the notes explicitly says the agent fixed one mistake by changing sample-event matching from direct indexing to within-trial selection. Otherwise, most of these behaviors are not justified philosophically; they are pragmatic guards added so conversion and decoder verification would not fail.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is repeated spike binning for every kept unit in every kept trial. Secondary costs come from per-trial interpolation of the full tongue trace and repeated per-trial event filtering.

ii. 
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
...
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
```

iii. Step 10 of `CONVERSION_NOTES.md` says the full conversion was “initially too slow” and specifically credits the `searchsorted` + `bincount` spike-binning optimization with reducing sample conversion time from about 30.6 s to about 11.0 s for two sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial list comprehension over all units for spike binning is the clearest vectorization target. Other candidates are the loop that maps region strings to indices, the per-trial photostim interval loop, and the repeated per-trial search through event streams and tracking timestamps.

ii. 
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)

region_idx = []
for r in unit_regions[kept_idx]:
    if r not in brain_regions:
        brain_regions.append(r)
    region_idx.append(brain_regions.index(r))

for s, e in zip(start_times, stop_times):
    ...
```

iii. The agent itself only documented the spike-binning optimization, but the structure of the final code shows several Python-level loops that could have been moved to more vectorized array logic.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans session-length arrays inside the trial loop: it re-searches the full sample event stream for each trial, re-filters the photostim event streams for each trial, and re-interpolates the full tongue trace for each trial.

ii. 
```python
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. There is no separate high-level justification in the notes for this repetition. It appears to be a straightforward implementation choice made for simplicity after the agent corrected the sample-event matching bug.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several variables that it never uses downstream, including `left_lick`, `right_lick`, `auto_water`, and `free_water`. It also returns `session_id`, `q40`, and `q60` from `load_session` even though only `session_id` is printed and neither percentile is stored in the exported dataset. Finally, it inflates scalar trial labels into full 80-bin constant arrays for choice, outcome, and early lick, which is convenient but memory-heavy.

ii. 
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
return {
    'session_id': get_session_id(nwb_path),
    ...
    'q40': float(q40),
    'q60': float(q60),
}
```

```python
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
    np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
    disc,
])
```

iii. The notes show that the agent knew `auto_water` and `free_water` were relevant to the reference trial mask, but in the final script they are only read and then ignored. There is no explicit justification for carrying these unused loads or for returning percentile values that are not exported.
