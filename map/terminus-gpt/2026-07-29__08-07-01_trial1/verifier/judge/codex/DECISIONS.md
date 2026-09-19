# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by recursively finding every `.nwb` file under `data/`, sorting the paths, and opening each session directly with `h5py`. Within each file it reads HDF5 groups such as `intervals/trials`, `units`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries`.

ii. 
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
...
with h5py.File(nwb_path, 'r') as f:
    trials = f['intervals/trials']
    ...
    go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    ...
    tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. In `CONVERSION_NOTES.md`, Step 2 says the data are organized as per-subject directories containing NWB files, and the trajectory repeatedly refers to the dataset as “174 NWB files”. There is no explicit justification for using `h5py` rather than `pynwb`; the apparent rationale is that the dataset is already HDF5-backed and easy to inspect by path.

## 1-b. How are the data split into subjects?

i. Subjects are split by directory name, not by NWB metadata. The subject id stored in the output is the parent folder name such as `sub-440956`.

ii. 
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
...
'subject': get_subject_id(nwb_path),
```

iii. The notes say the data are organized as “per-subject directories under `data/`”, and the code follows that filesystem structure directly. There is no evidence in the notes or trajectory that the AI reconsidered this and switched to `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. Sessions are split one file per session. The session id saved in the output is the filename stem, not `nwb.identifier`.

ii. 
```python
def get_session_id(nwb_path):
    return nwb_path.stem
...
files = sorted(Path('data').rglob('*.nwb'))
...
'session_id': get_session_id(nwb_path),
```

iii. Step 2 of the notes says each subject directory contains “NWB files named like `sub-<mouse>_ses-<timestamp>_behavior+ecephys+ogen.nwb`”, which the AI appears to have treated as the session boundary and session identifier.

## 1-d. How are the data split into trials?

i. Trials are taken from the rows of `intervals/trials`, paired by row index with the ordered `go_start_times` timestamps. The code does not assert that the lengths match; it silently uses `min(n_trials, len(go_times))` and iterates over that many indexed trials.

ii. 
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
n_match = min(n_trials, len(go_times))
...
for i in range(n_match):
    go = go_times[i]
```

iii. The notes describe the trials table and go-cue stream as the native trial structure. A later note in Step 10 says the AI had to fix `sample_start_times` because it is a global event stream rather than trial-indexed; that implies the AI was relying on trial-row indexing plus event ordering rather than on more explicit consistency checks.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial QC. Instead it keeps only trials that have a finite go cue, fully contain the requested `[-2.5, +1.5]` go-aligned window within `start_time`/`stop_time`, have recognized `trial_instruction`/`outcome`/`early_lick` labels, and have at least one `sample_start_times` event within the trial before the go cue. Sessions with fewer than 2 retained trials are dropped. `free_water` and `auto_water` are read but not used.

ii. 
```python
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
...
for i in range(n_match):
    go = go_times[i]
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

iii. The notes initially discussed reference “regular trials” and `free_water`, but the final code diverged. Step 5 says “Keep trials with valid go cue and sufficient aligned data”, and Step 10 says the AI fixed time-from-tone by selecting the last sample event within each trial. There is no explicit justification in the final notes for omitting `obs_intervals`-based filtering or for reading but not using `free_water` / `auto_water`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `units/spike_times` and `units/spike_times_index`, using `go_start_times` to define trial-aligned windows. Unit inclusion also depends on `units/unit_quality`, `units/electrodes`, and the electrode `location` JSON.

ii. 
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
...
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The notes map `units/spike_times` to `neural` and mention mapping unit regions from electrode `location`. Step 4 of the notes also says the AI was trying to reconcile paper unit counts by combining `unit_quality` and broad regions.

## 2-b. How is the `neural` data processed?

i. For each kept trial and kept unit, the AI bins spikes from `go_time - 2.5 s` to `go_time + 1.5 s` into 80 non-overlapping 50 ms bins and converts counts to firing rate by dividing by 0.05 s. No smoothing, normalization, baseline subtraction, or trial averaging is applied.

ii. 
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
...
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

iii. The instructions required 50 ms firing-rate bins around the go cue. The trajectory shows the AI explicitly optimized this step from `np.histogram` to `searchsorted` + `bincount` because it was too slow on the full dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `units/unit_quality == 'good'` and their electrode-level broad region is in a hard-coded `MAJOR_REGIONS` set containing bilateral ALM, Striatum, Thalamus, Midbrain, and Medulla.

ii. 
```python
MAJOR_REGIONS = {'left ALM','right ALM','left Striatum','right Striatum',
                 'left Thalamus','right Thalamus','left Midbrain',
                 'right Midbrain','left Medulla','right Medulla'}
...
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```

iii. Step 4 of `CONVERSION_NOTES.md` says `unit_quality == "good"` alone gave 154,948 units, but restricting to the major broad regions produced totals “close to” the paper’s 69,943 units. That is the clearest justification the AI gave for this choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, the code uses the trial’s `go_start_times` timestamp and bins spikes in a fixed window from `go - 2.5 s` to `go + 1.5 s`.

ii. 
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
for i in range(n_match):
    go = go_times[i]
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. This matches the explicit decoder instruction in the task and is also stated in the notes under “Temporal alignment”.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data has 50 ms bins over a 4 s window, giving 80 time bins per trial. No additional temporal rebinning is applied after spike binning.

ii. 
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes and trajectory consistently reference the decoder’s required 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` timestamps together with each trial’s go cue. The code finds the last sample event within the trial before the go cue and treats that as the tone onset.

ii. 
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
...
inp0 = build_time_from_tone(sample_time, go)
```

iii. Step 10 of the notes says the AI corrected an earlier bug after discovering that `sample_start_times` is a global event stream, not one value per trial. The trajectory explicitly mentions selecting “the last sample event within each trial before go cue.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes time from tone onset at each 50 ms bin center relative to the go cue, then clips all negative values to `0.0`. That turns the feature into “time since tone onset, floored at zero” rather than signed time relative to tone onset.

ii. 
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The notes justify the sample-event selection fix, and Step 7 reports that the range became `[0.0, 5.7]` after that change. There is no explicit justification for clipping pre-tone values to zero.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same 80 go-cue-centered bin centers used for neural activity, so each timepoint corresponds to the same aligned time axis as the binned spikes.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
inp0 = build_time_from_tone(sample_time, go)
...
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. The notes describe “align every modality to `go_start_times` and extract `[-2.5 s, +1.5 s]` windows with 50 ms bins,” and the code implements the time-from-tone input on that same grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the event streams `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, restricted to the current trial by `start_time` and `stop_time`.

ii. 
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
```

iii. The notes originally considered either event streams or the trial-table `photostim_onset` / `photostim_duration` fields. The final code chose event timestamps, likely because Step 2 dataset exploration identified these event streams explicitly.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI pairs photostim start and stop times by order, converts them to go-relative times, and sets a binary 80-bin vector to `1` for any bin whose interval overlaps the stimulation interval.

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
...
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. Step 5 of the notes says photostimulation should be represented as a binary time series in 50 ms bins. There is no explicit note justifying the overlap rule or the `min(len(ps), len(pe))` truncation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim times are aligned to neural data by subtracting the trial’s go cue from the event timestamps and then placing those go-relative intervals on the same 50 ms bin grid as the spikes.

ii. 
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. The notes repeatedly state that all modalities should be aligned to go cue onset.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. In the final code, “choice” is derived only from the trial-table column `trial_instruction`. The code also reads `left_lick_times`, `right_lick_times`, and `outcome`, but it does not use lick events or outcome to determine the choice label.

ii. 
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
choice = decode_arr(trials['trial_instruction'][:])
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. Step 5 of the notes explicitly mapped `trial_instruction` to `output[0]` and said “Use per-trial categorical output named choice.” That note is the clearest justification for the implemented decision, even though it does not distinguish instructed side from actual choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps `trial_instruction` strings `'left'` and `'right'` to integers `0` and `1`, repeats that value across all 80 bins, and exposes only two choice classes. There is no `'no lick'` class.

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
...
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The notes framed choice as a per-trial categorical label derived from `trial_instruction`, and the script implements that directly. There is no separate justification for omitting `ignore` / no-lick trials as a third choice state.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table column `outcome`.

ii. 
```python
outcome = decode_arr(trials['outcome'][:])
```

iii. The notes’ variable mapping table lists `outcome` directly as the source for `output[1]`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings `'ignore'`, `'miss'`, and `'hit'` are mapped to `0`, `1`, and `2`, then repeated across all 80 bins for each trial.

ii. 
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. This follows the notes and the task specification directly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table column `early_lick`.

ii. 
```python
early = decode_arr(trials['early_lick'][:])
```

iii. The notes map `early_lick` directly to `output[2]`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The strings `'no early'` and `'early'` are mapped to `0` and `1`, then repeated across all 80 bins for each trial.

ii. 
```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. This follows the task specification directly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The final code derives tongue y-position from `BehavioralTimeSeries/Camera0_side_TongueTracking/data[:, 1]` and the matching `timestamps`. It ignores the likelihood column in `data[:, 2]`.

ii. 
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. Step 5 of the notes planned to use `Camera0_side_TongueTracking/data[:,1]` and also said it still needed to “handle low-likelihood / missing tracking.” The final code never implemented that second part.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, the AI interpolates the full-session tongue y signal onto the 80 go-aligned bin centers. It then pools all finite interpolated values from all retained trials in the session, computes session-wide 40th and 60th percentiles, and discretizes each bin as low/mid/high relative to those cutoffs.

ii. 
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
...
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
...
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()]) if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
q40, q60 = np.percentile(all_tongue, [40, 60])
```

iii. The notes explicitly planned “Interpolate/aligned tongue y to trial bins; discretize per session by 40th/60th percentiles”. The trajectory also cited reference marker-alignment code. The final code follows that interpolation plan rather than the human reference’s per-bin averaging with visibility masking.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. It is thresholded into only three categories: `0` for values below `q40`, `1` for values between `q40` and `q60` or NaN, and `2` for values above `q60`. There is no fourth `'not visible'` category.

ii. 
```python
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
...
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

iii. The notes mention a 40th/60th percentile discretization but do not discuss the required explicit “not visible” class. The final script therefore collapses missing tongue values into the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue signal is aligned to neural data by taking the camera timestamps relative to the trial’s go cue and linearly interpolating tongue y at the same 80 bin centers used for the neural representation.

ii. 
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
...
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. The notes repeatedly reference go-cue-centered alignment for behavioral streams, and the trajectory mentions marker alignment from the reference notebooks. The AI’s concrete choice was to use interpolation to the bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid data are mostly handled by skipping trials. Trials are dropped if the go cue is missing, if the full analysis window does not fit within `start_time`/`stop_time`, if the categorical labels are not recognized, or if no valid pre-go sample event is found. For tongue tracking, if fewer than two finite samples are available after interpolation, the code returns all-NaN and later maps those NaNs to the middle tongue class; if no finite tongue values exist in an entire session, it falls back to percentiles from `[0, 1]`. Sessions with fewer than 2 surviving trials are dropped.

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
if valid.sum() < 2:
    return np.full(N_BINS, np.nan, dtype=np.float32)
...
all_tongue = ... if any(np.isfinite(x).any() for x in sess_tongue_cont) else np.array([0,1], dtype=np.float32)
...
if len(sess_neural) < 2:
    return None
```

iii. Step 10 of the notes says the AI fixed one data issue by selecting the last sample event within each trial before the go cue. The notes also flagged the need to handle low-likelihood / missing tracking, but the final implementation did not create a dedicated missing-data category for tongue bins.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is neural spike binning across all kept units and trials within each session. The trajectory shows this was the main runtime bottleneck, and the AI rewrote it from a slower `np.histogram` loop to the current `searchsorted` + `bincount` implementation. Loading full tongue tracking arrays per session is also nontrivial because the code interpolates them trial by trial.

ii. 
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    ...
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. In the trajectory, the AI explicitly estimated the initial full run would take roughly 40–50 minutes, interrupted it, and optimized spike binning. Sample runtime improved from about 30.6 s to about 11.0 s for 2 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization target is the nested trial-by-unit neural binning loop, which still performs one spike-binning call per kept unit per kept trial. The tongue interpolation is also repeated once per trial over full-session arrays. Smaller targets are the region indexing loop and photostim overlap loop.

ii. 
```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
...
for r in unit_regions[kept_idx]:
    if r not in brain_regions:
        brain_regions.append(r)
    region_idx.append(brain_regions.index(r))
...
for s, e in zip(start_times, stop_times):
    ...
```

iii. The trajectory specifically identifies the old per-trial per-neuron histogram loop as the bottleneck and calls for vectorization. The final code is improved but still not fully session-vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations across trials: it re-interpolates the full-session tongue trace separately for every kept trial, re-filters photostim start/stop arrays for every trial, and repeatedly bins each kept unit for each kept trial. It also performs linear list-based region indexing while building `brain_region_idx`.

ii. 
```python
for i in range(n_match):
    ...
    ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
    pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
    ...
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The notes do not call this out directly, but the trajectory’s performance discussion centers on these repeated per-trial operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads several arrays that are never used in the final outputs: `left_lick_times`, `right_lick_times`, `auto_water`, and `free_water`. It also computes session-specific `q40` and `q60` values and returns them from `load_session`, but `main()` never stores them in the final dataset. The full `tongue` array is loaded even though only column 1 is used.

ii. 
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
return {
    ...
    'q40': float(q40),
    'q60': float(q60),
}
...
data['neural'].append(sess['neural'])
data['input'].append(sess['input'])
data['output'].append(sess['output'])
data['subject_idx'].append(subject_map[subj])
data['brain_region_idx'].append(sess['brain_region_idx'])
```

iii. There is no explicit justification for this discarded work in the notes. It appears to be leftover exploration or partial implementation.
