# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans recursively for every `.nwb` file under `data/`, treats each file as one session, and opens files directly with `h5py` rather than `pynwb`. Within each file it reads HDF5 groups for trials, units, behavioral events, and behavioral time series.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
```

```python
with h5py.File(nwb_path, 'r') as f:
    trials = f['intervals/trials']
    unit_quality = decode_arr(f['units/unit_quality'][:])
    go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. In `CONVERSION_NOTES.md`, the AI documented that the dataset is organized as per-subject directories containing NWB files and concluded that iterating those files was the complete session list. The trajectory also shows it deliberately chose low-level NWB/HDF5 access after exploring the file layout.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the parent directory name of each NWB file, e.g. `sub-440956`, not from the NWB subject metadata. A session-level `subject_map` is built incrementally and `subject_idx` is appended in session order.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
```

```python
subjects = []
subject_map = {}
...
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_map[subj])
```

iii. The notes describe the data as stored in per-subject directories under `data/`, so the AI treated the folder name as the canonical subject identifier. No separate justification for preferring the folder name over `nwb.subject.subject_id` was recorded.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identifiers come from the file stem rather than `nwb.identifier`, and session order is the sorted recursive file order.

ii.
```python
def get_session_id(nwb_path):
    return nwb_path.stem
```

```python
files = sorted(Path('data').rglob('*.nwb'))
...
return {
    'session_id': get_session_id(nwb_path),
```

iii. In the notes, the AI described the data as one NWB file per session and repeatedly referred to the file list as the session list. The trajectory also shows it noticed some `behavior+ecephys` files without `+ogen` but still kept the file-as-session rule.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`, but the AI pairs them to go cues by row index and only iterates over `min(n_trials, len(go_times))`. It assumes the `i`th trial row and `i`th go cue correspond, then further drops trials whose requested neural window falls outside `start_time`/`stop_time`.

ii.
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
```

iii. The trajectory shows the AI initially tried direct indexing of `sample_start_times` by trial, discovered that event streams are global rather than trial-indexed, and then only fixed the sample-event matching. It did not add an explicit `len(go_times) == n_trials` check, so the indexed trial/go pairing remained.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial QC. Instead, it keeps trials only if the aligned 4 s window lies within `start_time`/`stop_time`, the choice/outcome/early labels are recognized, and a sample event exists before the go cue within the trial. It also drops any session with fewer than two kept trials. `auto_water` and `free_water` are loaded but never used, and `obs_intervals` is ignored.

ii.
```python
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
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

iii. The notes show the AI knew the reference code's "regular trial" mask excluded early lick, auto water, free water, and no-response trials, but it did not follow that in the final code. Its later Step 10 notes instead justify the implemented filtering as "valid go cue and sufficient aligned data."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are built from `units/spike_times`, filtered by `units/unit_quality` and by electrode `location`-derived broad brain region labels, then aligned to `BehavioralEvents/go_start_times`.

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
```

```python
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The AI's Step 4 notes and trajectory show a deliberate decision to use `unit_quality == "good"` and then restrict to "major" regions to reconcile paper counts. It treated the paper's 69,943 units as evidence that ECT/BLA and similar regions should be excluded.

## 2-b. How is the `neural` data processed?

i. For each kept unit and trial, spikes in `[go-2.5, go+1.5)` are binned into 50 ms bins, converted to counts with `np.bincount`, and divided by bin width to produce firing rates in Hz. There is no smoothing, normalization, or baseline subtraction.

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
```

```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The notes state the intended transform was "Bin spikes in 50 ms bins from -2.5 s to +1.5 s around each trial's go cue; convert to firing rate or spike count per bin consistently across sessions." Later trajectory entries justify the optimized `searchsorted`/`bincount` version as a speedup over the original slower nested histogramming.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered with `unit_quality == 'good'` and restricted to a hard-coded set of broad regions (`ALM`, `Striatum`, `Thalamus`, `Midbrain`, `Medulla`, left/right). The code does not use `units/classification`, does not use the QC classifier verdict from the white paper, and does not explicitly drop sessions with zero kept units before trying to stack them.

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

iii. The AI explicitly justified this in Step 4 and the trajectory: raw `unit_quality == "good"` gave too many units, so it chose a major-region subset to get closer to the paper's reported totals. This was a deliberate curation decision rather than an accident.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the session's go cue time. The neural window for a trial is defined relative to `go_times[i]`, and spike times are converted to offsets from `go + T_START`.

ii.
```python
for i in range(n_match):
    go = go_times[i]
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

```python
lo = go_time + T_START
hi = go_time + T_END
rel = spike_times[a:b] - lo
```

iii. The notes repeatedly say all modalities should be aligned to go cue and extracted from `[-2.5 s, +1.5 s]`. The AI followed that rule consistently across the conversion.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 bins of width 50 ms covering `[-2.5, 1.5]` seconds around go cue. There is no later rebinning; this fixed grid is used throughout.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The task instructions required 50 ms bins and the notes record that as a fixed design choice. No alternate binning strategy was documented.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from the global `sample_start_times` event stream plus each trial's go cue and trial boundaries. For each trial, the AI selects the last sample event within `[trial_start, trial_stop]` and before the go cue.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
sample_time = find_last_event_within_trial(
    sample_event_times, trial_start[i], trial_stop[i], before_time=go
)
```

iii. The trajectory shows this was an explicit correction. The AI initially assumed `sample_start_times` was trial-indexed, then observed that there were more sample events than go cues and changed to "the last sample event within the trial before go."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI represents this input as time since tone onset on the go-aligned bin centers. It computes `BIN_CENTERS - (sample_time - go_time)` and clips negative values to `0.0`, so bins before tone onset are flattened to zero rather than left negative.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The notes and trajectory show the AI debated whether a "time" input should be binary or continuous. After debugging an indexing bug, it settled on a continuous ramp-like "time since tone" representation and judged a nonnegative range such as `0.0` to `5.7` to be "sensible."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on the exact same 80 go-aligned bin centers used for neural binning, one value per neural time bin.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
inp0 = build_time_from_tone(sample_time, go)
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. The notes explicitly planned that all modalities would share the same go-cue-centered 50 ms grid, and the final code follows that.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, restricted to events whose timestamps fall within the current trial. The trial-table fields `photostim_onset` and `photostim_duration` are read in notes exploration but not used in the final code.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
```

iii. The notes say photostim should be built from event intervals relative to go cue and verified against raw timestamps. The trajectory shows the AI considered both trial-table and event-stream representations, then implemented the event streams.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts each photostim interval into a binary time series by marking a bin as `1` if the stimulus interval overlaps that 50 ms bin at all. Non-overlapping bins remain `0`.

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

iii. The notes state that photostim should be a binary on/off input aligned to go cue. The trajectory mentions later sanity checks that the binary series matched raw photostim timestamps for a sampled trial.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI subtracts the trial's go cue from each start/stop time and compares the resulting relative interval against the same `BIN_EDGES` used for neural binning.

ii.
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. The notes planned go-cue-centered alignment for every modality, and the Step 10 notes claim the raw-to-converted photostim series matched a direct reconstruction for a checked trial.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI effectively treats the `trial_instruction` field as the choice output. Although it loads `left_lick_times` and `right_lick_times`, it does not use them, and it does not derive actual lick direction from `outcome`.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
choice = decode_arr(trials['trial_instruction'][:])
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. The Step 5 mapping table in `CONVERSION_NOTES.md` explicitly mapped `trial_instruction` to output choice, so this was a deliberate semantic choice. No further justification for ignoring actual lick behavior was recorded.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The strings `'left'` and `'right'` are mapped to `0` and `1` and repeated across all 80 bins for each trial. There is no third "no lick" class for `ignore` trials.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
    np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
    disc,
])
```

iii. The notes only describe a direct mapping from `trial_instruction` to choice. The trajectory does not contain any explicit reconsideration of choice semantics after implementation.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` field.

ii.
```python
outcome = decode_arr(trials['outcome'][:])
```

iii. The Step 5 mapping table in the notes lists `outcome` as a direct mapping from the NWB trial table to decoder output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped with `ignore -> 0`, `miss -> 1`, `hit -> 2` and then repeated across all time bins for that trial.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. The code follows the mapping documented in the notes and matches the instruction-specified label order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` field.

ii.
```python
early = decode_arr(trials['early_lick'][:])
```

iii. The Step 5 mapping table in the notes lists `early_lick` as a direct mapping from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings `'no early'` and `'early'` are mapped to `0` and `1`, then repeated across all 80 bins for each trial.

ii.
```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. The notes describe this as a straightforward direct trial-field mapping and do not record any additional processing choice.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The output is derived from `Camera0_side_TongueTracking/data[:, 1]` and the corresponding timestamps. The AI does not use the tracking likelihood channel in the final code.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. The Step 5 notes say the relevant source is `Camera0_side_TongueTracking/data[:,1]` and that the `(n_frames, 3)` array was "likely x/y/likelihood." The trajectory shows the AI intended to inspect the columns more carefully but then proceeded with the second column as tongue y.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI interpolates the session-wide tongue-y trace onto each trial's go-aligned bin centers, collects all finite interpolated values across kept trials in the session, computes the 40th and 60th percentiles on those values, and then discretizes each trial's interpolated trace with those thresholds. It does not threshold by tongue-likelihood or compute per-bin means from raw frames.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

```python
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
sess_tongue_cont.append(ty)
...
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()])
q40, q60 = np.percentile(all_tongue, [40, 60])
```

iii. The Step 5 notes planned "interpolate/aligned tongue y to trial bins; discretize per session by 40th/60th percentiles into 3 classes." The trajectory shows the AI never implemented the more careful missing-data/likelihood handling it had flagged as still needing confirmation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories are used: values below the session 40th percentile become `0`, above the 60th percentile become `2`, and everything else stays `1`. NaNs are left at the default middle class because no explicit hidden/not-visible class is created.

ii.
```python
q40, q60 = np.percentile(all_tongue, [40, 60])
for ty, i in zip(sess_tongue_cont, valid_trial_inds):
    disc = np.full(N_BINS, 1, dtype=np.int64)
    disc[ty < q40] = 0
    disc[ty > q60] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The notes explicitly chose a 40th/60th percentile, per-session, 3-class discretization. No justification was recorded for treating missing bins as the middle class or for omitting a "not visible" category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y by interpolating the session-wide tracking timestamps into the same go-aligned `BIN_CENTERS` used for neural data, once per trial. It does not restrict camera samples to the trial before interpolation except through `np.interp`'s `left`/`right` NaN behavior.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

```python
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. The notes and trajectory consistently state that all modalities should share the go-cue-centered 50 ms grid. The AI chose interpolation onto that grid rather than frame-to-bin averaging.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness mostly by skipping trials. Trials with non-finite go cues, unrecognized trial labels, or missing pre-go sample events are dropped. If a session has fewer than two surviving trials it is dropped. For tongue data, bins outside the interpolation support become `NaN`, but missing tongue visibility is not represented explicitly; if an entire session has no finite tongue values, percentile thresholds fall back to `[0, 1]`.

ii.
```python
if not np.isfinite(go):
    continue
...
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
if not np.isfinite(sample_time):
    continue
...
if len(sess_neural) < 2:
    return None
```

```python
if valid.sum() < 2:
    return np.full(N_BINS, np.nan, dtype=np.float32)
...
all_tongue = ... if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
```

iii. The notes justify dropping trials when there is not enough aligned data and mention that raw-to-converted sanity checks were run on selected inputs/outputs. There is no explicit note justifying the fallback percentile array or the lack of a dedicated missing-tongue class.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant costs in the AI's code are repeated per-trial, per-unit spike binning and repeated per-trial interpolation of the entire tongue tracking trace. Reading full spike and tracking arrays from each NWB file is also expensive. The trajectory confirms these were the runtime bottlenecks the AI noticed and tried to optimize.

ii.
```python
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
...
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. In the trajectory, the AI explicitly estimated the original implementation would take 40-50 minutes, identified the nested per-trial/per-neuron binning as the bottleneck, and later reported that optimizing this reduced sample runtime from about 30.6 s to 11.0 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest remaining vectorization opportunity is the per-trial loop that recomputes neural matrices by looping over all kept units for every trial. The tongue interpolation is also repeated one trial at a time across the full session trace. Smaller loops include the photostim interval loop and the brain-region indexing loop.

ii.
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    ...
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

```python
for s, e in zip(start_times, stop_times):
    ...
for r in unit_regions[kept_idx]:
    if r not in brain_regions:
        brain_regions.append(r)
```

iii. The trajectory shows the AI recognized exactly this issue and specifically called out the Python-level histogram/binning inside the trial loop as the main optimization target.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rebins the same unit spike trains separately for each trial rather than binning all trials for a unit together, and it repeatedly interpolates the full tongue trace for every trial. These are repeated computations over session-wide arrays.

ii.
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    ...
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan)
```

iii. The trajectory documents that the AI noticed and partially optimized repeated spike processing, but it never restructured the code to compute trial-by-trial neural counts in the more vectorized reference style or to avoid repeated tongue interpolation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several arrays are read and then ignored: `left_lick_times`, `right_lick_times`, `auto_water`, and `free_water`. The session-level `q40` and `q60` thresholds are computed and returned from `load_session` but never saved into the final output structure.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

```python
return {
    ...
    'q40': float(q40),
    'q60': float(q60),
}
```

```python
data['neural'].append(sess['neural'])
data['input'].append(sess['input'])
data['output'].append(sess['output'])
data['subject_idx'].append(subject_map[subj])
data['brain_region_idx'].append(sess['brain_region_idx'])
```

iii. There is no explicit justification in the notes for these unused reads or the unsaved percentile metadata. They appear to be leftovers from exploration or intermediate processing choices.
