# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter scans the local `data/` tree for every `.nwb` file, sorts the paths, and treats each NWB file as one session. Each session is opened with `h5py`, and the script reads the `intervals/trials`, `units`, `BehavioralEvents`, and `BehavioralTimeSeries` groups needed for downstream trial conversion.

ii. ```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)

for fp in files:
    sess = load_session(fp, brain_regions)
```

```python
with h5py.File(nwb_path, 'r') as f:
    trials = f['intervals/trials']
    unit_quality = decode_arr(f['units/unit_quality'][:])
    spikes_flat = f['units/spike_times'][:]
    go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented that the dataset is organized as per-subject directories containing NWB files. In Step 10, it acknowledged that this strategy yields 174 sessions, which does not exactly match the paper's 173-session summary, but it still kept loading the full NWB set.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the NWB file path: the parent directory name, such as `sub-440956`, becomes the subject ID. A `subject_map` then assigns each unique subject a stable integer index used in `subject_idx`.

ii. ```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name

subjects = []
subject_map = {}
...
subj = sess['subject']
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
data['subject_idx'].append(subject_map[subj])
```

iii. Step 2 of `CONVERSION_NOTES.md` says the data are organized as per-subject directories under `data/`. The code follows that directory structure directly.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is the file stem, and the converted output stores one list entry per NWB file in `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
def get_session_id(nwb_path):
    return nwb_path.stem

for fp in files:
    sess = load_session(fp, brain_regions)
    ...
    data['neural'].append(sess['neural'])
    data['input'].append(sess['input'])
    data['output'].append(sess['output'])
```

iii. Step 2 of `CONVERSION_NOTES.md` explicitly describes each NWB file as a session file named like `sub-<mouse>_ses-<timestamp>_behavior+ecephys+ogen.nwb`.

## 1-d. How are the data split into trials?

i. Trials are indexed by position through the trial table and the go-cue event array. The loop uses `min(n_trials, len(go_times))` and converts only trials whose go cue is finite, whose full [-2.5 s, +1.5 s] window lies inside the recorded trial start/stop times, whose categorical labels are recognized, and whose sample event can be recovered.

ii. ```python
n_match = min(n_trials, len(go_times))
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
```

iii. The notes emphasize valid go-cue-centered windows and later mention a bug fix for `sample_start_times` being a global event stream rather than a trial-indexed array. That explains why the converter reconstructs trial membership from event times and trial start/stop bounds.

## 1-e. How are trials filtered based on quality controls?

i. The converter does not apply the reference code's regular-trial mask. Instead, it keeps any trial with a valid go cue, a complete alignment window, recognized `choice`/`outcome`/`early_lick` labels, and a recoverable sample event. It ignores `auto_water`, `free_water`, explicit no-response filtering, and photostimulation filtering when deciding whether to keep a trial.

ii. ```python
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
...
sess_neural.append(neural.astype(np.float32))
```

iii. In Step 1 and Step 3, the agent noted that the reference code defines regular trials by excluding early lick, auto water, free water, and no-response trials. Despite that, the implementation keeps those trial types, likely because the decoder task asked it to predict `early_lick` and `outcome`, though the notes never fully resolve that conflict.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the NWB `units/spike_times` ragged array, indexed by `units/spike_times_index`. Unit inclusion additionally depends on `units/unit_quality`, `units/electrodes`, and `general/extracellular_ephys/electrodes/location` so the script can keep only selected neurons and attach region labels.

ii. ```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
...
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. Step 5 of the notes explicitly maps `units/spike_times` to the target `neural` field and says region labels come from electrode-location metadata routed through `units/electrodes`.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each kept trial, the code extracts spikes from 2.5 s before to 1.5 s after go cue, bins them into 50 ms bins, and divides by bin width to produce firing rates. The final per-trial neural array has shape `(n_neurons, 80)`.

ii. ```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
...
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    ...
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE

neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The task specification and Step 5 of the notes both call for go-cue alignment and 50 ms bins. Step 10 also says the spike binning was optimized with `searchsorted` and `bincount`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps units only if `unit_quality == 'good'` and the parsed broad region string belongs to a hard-coded set of ten major bilateral regions: ALM, Striatum, Thalamus, Midbrain, and Medulla.

ii. ```python
MAJOR_REGIONS = {
    'left ALM','right ALM','left Striatum','right Striatum',
    'left Thalamus','right Thalamus','left Midbrain','right Midbrain',
    'left Medulla','right Medulla'
}
...
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
kept_idx = np.where(keep_units)[0]
```

iii. Step 4 of the notes says the agent found that restricting to major broad regions brought the unit count closer to the paper's 69,943-unit summary. Step 5 then records a deliberate decision to start from `good` units and exclude non-paper regions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the go cue. Absolute spike times are converted to trial-relative time by subtracting the trial's `go_time`, then cropped to [-2.5, 1.5] s and binned on that relative axis.

ii. ```python
lo = go_time + T_START
hi = go_time + T_END
...
rel = spike_times[a:b] - lo
bins = np.floor(rel / BIN_SIZE).astype(np.int64)
```

iii. Step 5 states, 'Align every modality to `go_start_times` and extract [-2.5 s, +1.5 s] windows.'

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins throughout. There is no second-stage temporal rebinning after the initial spike binning and the construction of bin centers/edges.

ii. ```python
BIN_SIZE = 0.05
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The agent followed the explicit decoder-task requirement for 50 ms bins and repeated that decision in Step 5 of the notes.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The converter derives this input from the global event stream `acquisition/BehavioralEvents/sample_start_times/timestamps` and the per-trial go-cue time. Within each trial, it chooses the last sample-start event before the go cue and treats that as tone onset.

ii. ```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
sample_time = find_last_event_within_trial(
    sample_event_times, trial_start[i], trial_stop[i], before_time=go
)
inp0 = build_time_from_tone(sample_time, go)
```

iii. In Step 5, the notes say '`sample_start_times` or tone-on event' would be used for this variable, and in Step 10 the agent records a bug fix: `sample_start_times` is a global event stream, so the code now selects the last sample event within each trial before go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding a trial's sample-start time, the code computes `BIN_CENTERS - tone_rel`, where `tone_rel = sample_time - go_time`, and clips negative values to 0. This yields a time-varying ramp that is zero before tone onset and increases in seconds after tone onset.

ii. ```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The Step 10 notes say the time-from-tone signal was initially wrong and was corrected by first recovering the trial's sample event. The remainder of the processing appears to be the agent's own design choice for a continuous time-varying input.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-centered bin centers used for neural firing rates, so each trial's time-from-tone input is already aligned to the neural time axis.

ii. ```python
inp0 = build_time_from_tone(sample_time, go)
...
sess_input.append(np.stack([inp0, inp1], axis=0).astype(np.float32))
```

iii. Step 5 states that all modalities should be aligned to `go_start_times` with a shared trial window.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the event streams `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`. The converter does not use the trial-table `photostim_onset`, `photostim_duration`, or `photostim_power` fields.

ii. ```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
```

iii. Step 5 maps either the event streams or the trial-table onset/duration fields to the photostim input and says they will be aligned relative to go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Within each trial, the converter selects photostim start and stop events whose timestamps fall inside the trial, truncates mismatched start/stop arrays to the common minimum length, and marks any 50 ms bin that overlaps a stimulation interval as 1.

ii. ```python
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

iii. The notes say the photostim input should be built relative to go cue and later report a direct sanity check comparing the binary series against raw event timestamps.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Start and stop times are shifted by subtracting the trial's go-cue time, then converted onto the same bin edges used for neural firing rates.

ii. ```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Step 4 of the notes explicitly says to 'build photostim input relative to go cue and verify all stimulation bins are pre-go.'

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The converter derives this output from the trial-table field `intervals/trials/trial_instruction`, which contains `left` or `right` labels. It does not use the actual lick-response streams or the reference code's `lick_directions` behavioral variable.

ii. ```python
choice = decode_arr(trials['trial_instruction'][:])
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. Step 5 of the notes maps `trial_instruction` to the target `choice` output and does not mention using actual lick times or `lick_directions`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script maps `left -> 0` and `right -> 1`, then repeats that per-trial value across all 80 time bins.

ii. ```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
```

iii. The notes only justify this at the level of 'map left/right to choice'; there is no deeper discussion of using instruction instead of actual lick behavior.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-table field `intervals/trials/outcome`.

ii. ```python
outcome = decode_arr(trials['outcome'][:])
```

iii. Step 5 of the notes explicitly maps `outcome` from the NWB trial table to the decoder output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then broadcasts that per-trial category across all 80 time bins.

ii. ```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. The mapping follows the decoder task exactly and is recorded in the converter constants.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no `distance to reward zone` output in `convert_data.py`. For the actual implemented output (`outcome`), the converter aligns it to neural data by repeating the trial-level outcome category across every neural time bin.

ii. ```python
out = np.vstack([
    np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
    np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
    np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
    disc,
])
```

iii. This appears to be a typo in the evaluation questionnaire: the converter never defines a reward-zone-distance output. The implemented outcome output is simply expanded across the common bin axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trial-table field `intervals/trials/early_lick`.

ii. ```python
early = decode_arr(trials['early_lick'][:])
```

iii. Step 5 maps `early_lick` from the NWB trial table directly to the decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then repeats that trial-level category across all 80 bins.

ii. ```python
EARLY_MAP = {'no early': 0, 'early': 1}
...
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. The mapping is explicit in the converter constants. The notes do not add much justification beyond following the decoder-task label definition.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is taken from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data[:, 1]`, paired with `.../timestamps`. The converter ignores the third column, which appears to be tracking confidence or likelihood.

ii. ```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. Step 5 of the notes explicitly maps `Camera0_side_TongueTracking/data[:,1]` to tongue y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each kept trial, the code linearly interpolates the session-wide continuous tongue y trace onto the 80 go-cue-centered decoder bins. If fewer than two finite points exist, it returns all-NaN for that trial. The confidence column is ignored.

ii. ```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

iii. Step 5 says tongue y will be aligned and interpolated to trial bins, and Step 1 notes that the reference code contains go-cue-centered marker alignment helpers. The exact interpolation rule is the converter author's own choice.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The converter pools all finite aligned tongue-y values from the kept trials of a session, computes the 40th and 60th percentiles, initializes every bin to the middle class, then assigns bins below the 40th percentile to class 0 and bins above the 60th percentile to class 2.

ii. ```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont if np.isfinite(x).any()]) ...
q40, q60 = np.percentile(all_tongue, [40, 60])
...
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. This follows the decoder-task specification and was planned explicitly in Step 5 of the notes.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The aligned tongue trace is evaluated on the same 80 go-cue-centered bin centers as the neural firing rates, then discretized on that shared grid.

ii. ```python
ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
...
sess_output.append(out)
```

iii. Step 5 says all modalities should share the go-cue-centered alignment window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter uses a mix of skipping, truncation, and default filling. Trials are skipped if key events or valid windows are missing; mismatched photostim start/stop arrays are truncated to the shorter length; tongue traces with too little valid data become all-NaN and later default mostly to the middle category; and sessions with no finite tongue values fall back to an artificial `[0, 1]` array for percentile computation.

ii. ```python
if not np.isfinite(go):
    continue
...
if not np.isfinite(sample_time):
    continue
...
m = min(len(ps), len(pe))
...
if valid.sum() < 2:
    return np.full(N_BINS, np.nan, dtype=np.float32)
...
all_tongue = ... if any(...) else np.array([0,1], dtype=np.float32)
```

iii. The Step 10 notes mention one concrete data-handling fix for `sample_start_times`, but most of the missing-data policy is implicit in the code rather than explicitly justified in the notes.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session, per-trial, per-unit spike binning, followed by stacking the binned neural matrices. The agent explicitly identified spike binning as the main bottleneck and optimized it.

ii. ```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. Step 10 of the notes says runtime was initially too slow and that optimizing spike binning with `searchsorted` and `bincount` reduced the sample runtime from about 30.6 s to 11.0 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop repeatedly bins every kept unit one by one, scans global event arrays trial by trial, and loops over photostim intervals bin by bin. These are the most obvious places where more vectorization or batched preprocessing could reduce runtime.

ii. ```python
for i in range(n_match):
    ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
    ...
    ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
```

iii. The notes do not enumerate every vectorization opportunity, but the runtime discussion in Step 10 makes clear that the author recognized loop-heavy spike binning as the main inefficiency.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the same session-wide event arrays inside the trial loop to recover per-trial sample and photostim times, and it repeatedly broadcasts per-trial categorical outputs into full-length 80-bin vectors for every trial.

ii. ```python
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64)
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64)
```

iii. This repetition is visible directly in the code; the notes only partially mention it through the runtime discussion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes continuous aligned tongue traces only to discard them after discretization, returns `q40` and `q60` in the session dict but never writes them into the final dataset, and expands trial-level categorical outputs into 80-bin constant arrays even though those labels are fundamentally per-trial.

ii. ```python
sess_tongue_cont.append(ty)
...
return {
    ...
    'q40': float(q40),
    'q60': float(q60),
}
...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64)
```

iii. This follows from the implementation itself; the notes do not explicitly call out all of these discarded intermediates.
