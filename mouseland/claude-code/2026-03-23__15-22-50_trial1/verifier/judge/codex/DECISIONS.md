# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script loads `data/beh/Imaging_Exp_info.npy` to enumerate recordings, deduplicates sessions by `mname_datexp_blk`, and then loads each session's behavior, spike, and retinotopy files separately during processing. Unlike the human reference, it does not group sessions by behavior file and reuse a single loaded behavior dict across sessions.

ii.
```python
def get_unique_sessions():
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()
```
```python
def load_behavior_for_session(session_key, exp_type, ndb):
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()
```
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. `CONVERSION_NOTES.md` says the AI chose one unique physical session per recording and explicitly tried to "use reference code's `load_spk` approach (concatenate planes)." No separate justification was given for reloading behavior files per session instead of grouping them once.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `mname` from the experiment index. The final `subjects` list is the sorted set of mouse names in the processed sessions, and `subject_idx` maps each session to that list.

ii.
```python
for sess_key, (exp_type, ndb) in session_map.items():
    mname = ndb['mname']
```
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. `CONVERSION_NOTES.md` states there are 19 subjects and that sessions should be assigned to mice using the metadata in `Imaging_Exp_info.npy`.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `mname_datexp_blk` combination. If the same physical session appears under multiple experiment types, only the first occurrence is kept.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_map:
    session_map[key] = (exp_type, ndb)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as "Each physical session used once" because the same recording can appear under multiple experiment types.

## 1-d. How are the data split into trials?

i. Trials are split with contiguous frame windows from rounded `StartFr` to rounded `GrayFr`. The AI does not use `ft_trInd` plus `ft_CorrSpc` to define the kept frames; it assumes the trial is the full contiguous frame range between those boundary markers.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
```
```python
sfr = start_frs[trial_idx]
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The justification in `CONVERSION_NOTES.md` is that the trial window should be "StartFr to GrayFr (corridor entry to grey space entry) - captures full textured corridor portion."

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if the rounded frame range is invalid (`sfr < 0`, `gfr > n_frames`, or `sfr >= gfr`) or if the resulting trial has fewer than 2 frames. The AI does not apply the human reference's dataset-wide long-trial outlier filter.

ii.
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue

n_trial_frames = gfr - sfr
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. The later notes justify the skip only as handling an "invalid frame range" and explicitly say very long trials were retained because the AI believed the reference code did not filter them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` arrays stored in each session's `*_neural_data.npy` file, concatenated across imaging planes. Retinotopy `iarea` is loaded separately to build brain-region labels.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. `CONVERSION_NOTES.md` says this matches the reference loading path and that the neural data are already deconvolved fluorescence traces, so no upstream derivation step is needed.

## 2-b. How is the `neural` data processed?

i. Neural processing is minimal: concatenate planes, slice frames trial-by-trial, and cast each per-trial matrix to `float16`. No interpolation, smoothing, padding, or further signal processing is applied.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
neural_trials.append(trial_neural)
```

iii. The AI's notes justify this by stating that the source data are already deconvolved and that `float16` reduces the size of the converted dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out. It assigns every neuron to one of five labels, including an `"other"` category for `iarea` values outside V1/mHV/lHV/aHV, and retains them all.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
```
```python
region_idx = np.full(len(iarea), 4, dtype=int)
region_idx[iarea == 8] = 0
region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
```

iii. `CONVERSION_NOTES.md` repeatedly states "No neuron filtering" and says all Suite2p-detected neurons were kept because the AI believed the reference code used all neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to corridor entry by starting each trial at rounded `StartFr`; the aligned window then runs contiguously until rounded `GrayFr`.

ii.
```python
sfr = start_frs[trial_idx]
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The justification in the notes is that the decoder should be "Temporally aligned based on trial start (corridor entry)" and that `StartFr` to `GrayFr` captures the textured corridor traversal.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the imaging-frame resolution. For each session the AI estimates frame duration from the median difference of `ft`, uses that `dt_sec` for time-valued inputs, and stores the median across sessions in metadata. No rebinning is applied.

ii.
```python
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
```
```python
'time_bin_size': float(median_dt * 1000)
```

iii. `CONVERSION_NOTES.md` justifies this as matching the approximately 3.17 Hz imaging cadence while using the timestamps to compute the actual frame duration.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the frame interval estimated from `ft`.

ii.
```python
sound_frs = beh['SoundFr']
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
```

iii. The code comment says `SoundFr` is kept "as float for precise time computation," which is the clearest available justification.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI builds a frame index array from `sfr` to `gfr - 1` and computes `time_to_sound = (SoundFr[trial] - frame_indices) * dt_sec`, yielding positive values before the cue and negative values after it.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. `CONVERSION_NOTES.md` says the intended transform is `(SoundFr - frame_idx) * dt_sec`, and the code comment adds that `SoundFr` was intentionally left as a float for precision.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same frame indices used to slice the per-trial neural matrix, so its time axis has exactly the same length as the neural data for that trial.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The AI's notes repeatedly justify temporal alignment by saying all decoder variables are constructed on the same trial frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` in the experiment index, using session metadata rather than a variable in the behavioral arrays.

ii.
```python
mname = ndb['mname']
datexp = ndb['datexp']
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_sessions[mname].append((sess_key, date))
```

iii. `CONVERSION_NOTES.md` says the AI used "calendar days from mouse's first recording date."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by actual recording date and encoded as the integer calendar-day offset from that mouse's first recording, not the ordinal count of recorded sessions.

ii.
```python
sessions.sort(key=lambda x: x[1])
first_date = sessions[0][1]
for sess_key, date in sessions:
    day_map[sess_key] = (date - first_date).days
```

iii. The notes explicitly justify this as "Days since mouse's first recording" and repeat that the variable is a calendar-day difference.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the frame duration estimated from `ft`.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
```

iii. The justification is implicit in the notes' mapping table: use `(frame_idx - StartFr) * dt_sec` to express elapsed time from corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI rounds `StartFr` to an integer frame index `sfr`, constructs contiguous `frame_indices`, and computes `time_since_start = (frame_indices - sfr) * dt_sec`.

ii.
```python
sfr = start_frs[trial_idx]
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. `CONVERSION_NOTES.md` states the transform as `(frame_idx - StartFr) * dt_sec`, with Step 5 also saying fractional frame markers were rounded to the nearest frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is created from the same trial frame indices used for neural slicing, so it is one value per neural time bin in the `sfr:gfr` window.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes justify this with the same general alignment rationale used elsewhere: one shared trial window for all streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`.

ii.
```python
is_rew = beh['isRew']
```

iii. `CONVERSION_NOTES.md` maps `isRew` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to `0.0` or `1.0` once per trial and then broadcast across all time bins in that trial.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
np.full(n_trial_frames, rew_val, dtype=np.float32)
```

iii. The notes justify this as a per-trial discrete input: "1 if rewarded corridor, 0 otherwise."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii.
```python
wall_names = beh['WallName']
stim_name = wall_names[trial_idx]
```

iii. `CONVERSION_NOTES.md` says the AI chose `WallName` rather than `stim_id` and used it directly for stimulus labeling.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all unique `WallName` strings across processed sessions, sorts them, maps each unique name to an integer, stores that index, and broadcasts it across all frames of the trial. This yields 15 fine-grained categories rather than collapsing them to 4 base textures.

ii.
```python
def build_stimulus_mapping(session_results):
    all_stim_names = set()
    for result in session_results:
        for (_, stim_name) in result['output']:
            all_stim_names.add(stim_name)
    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```
```python
stim_idx = stim_to_idx[stim_name]
out_data[0, :] = stim_idx
```

iii. The notes explicitly justify this by saying "Stimulus categories: Use WallName directly" and reporting 15 unique stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`. The function also reads `LickTrind` but does not use it in the final logic.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```

iii. `CONVERSION_NOTES.md` says licking is built from `LickFr` and treats `LickTrind` as part of the available metadata.

## 8-b. What processing is involved in computing `output` *Licking*?

i. `LickFr` values are rounded to the nearest integer frame, restricted to the current trial's `start_fr:end_fr` window, and converted into a binary vector where a frame is 1 if one or more licks map to it.

ii.
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
frame_offsets = lick_frs_int[mask] - start_fr
lick_binary[frame_offsets] = 1.0
```

iii. The notes justify this as "Binary 1 at any frame that has >=1 lick (using LickFr rounded to nearest integer frame)."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is built over the same `sfr:gfr` frame span as the neural slice, so it matches the neural trial length exactly.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. The code structure itself is the justification: both arrays are defined from the same trial frame window, and the notes say all streams share that window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[sfr:gfr]
```

iii. `CONVERSION_NOTES.md` maps `ft_Pos` directly to corridor position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI takes the contiguous per-trial `ft_Pos` samples and discretizes them into 4 equal bins spanning the 4 m textured corridor using `np.digitize`.

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes justify this as "4 equal bins of 10 dm each (0-10, 10-20, 20-30, 30-40)."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position thresholds are the fixed edges `[0, 10, 20, 30, 40]` decimeters, giving four 1 m bins after clipping into category IDs `0..3`.

ii.
```python
bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The justification in the notes is that the decoder task explicitly requested four equal 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced from the same per-trial frame range `sfr:gfr` as the neural data.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_pos = ft_pos[sfr:gfr]
```

iii. The notes justify this with the same single-window temporal alignment choice used for the other per-frame variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii.
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. `CONVERSION_NOTES.md` maps `ft_RunSpeed` directly to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first gathers all `ft_RunSpeed` values from `ft_CorrSpc` frames across the chosen sessions, computes global 25th/50th/75th percentile thresholds after removing non-finite values, and then digitizes each trial's speeds against those thresholds.

ii.
```python
def collect_all_corridor_speeds(session_map, max_sessions=None):
    corr_mask = beh['ft_CorrSpc'][:n_frames]
    speeds = beh['ft_RunSpeed'][:n_frames]
    all_speeds.append(speeds[corr_mask])
```
```python
def compute_speed_bin_edges(all_speeds):
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
```
```python
speed_binned = discretize_speed(trial_speed, speed_quartiles).astype(np.float32)
```

iii. The notes justify this as "Running speed quartiles: Computed across ALL corridor frames in the dataset," later adding that `right=True` was used so exact zeros would fall into the lowest bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by global percentile edges computed with `np.percentile(valid, [25, 50, 75])`, then assigned with `np.digitize(..., right=True)` into categories `0..3`.

ii.
```python
quartiles = np.percentile(valid, [25, 50, 75])
```
```python
binned = np.digitize(speed, quartiles, right=True)
binned = np.clip(binned, 0, 3)
```

iii. The justification in the notes is that the task asked for four bins corresponding to 25% of the data, so percentile cutoffs were used.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sliced from the same `sfr:gfr` frame window as the neural data before discretization.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_speed = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles)
```

iii. The justification is the same shared-frame-window alignment used throughout the AI script.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates all per-frame behavioral streams to `min(n_frames_neural, n_frames_beh)`, removes non-finite values when computing speed quartiles, skips trials with invalid frame ranges or fewer than 2 frames, and returns all-zero licking for trials with no licks.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
```
```python
valid = all_speeds[np.isfinite(all_speeds)]
```
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. The notes justify these checks as needed for data validity and specifically mention one skipped trial due to an invalid frame range.

## 12-a. What are the most time-consuming steps of the code?

i. The code suggests the expensive steps are reading and concatenating the large per-session spike files, computing global speed quartiles by scanning all sessions once, and writing the very large pickle. The notes' runtime table says session processing dominates and that saving the full pickle takes minutes.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```
```python
all_speeds = collect_all_corridor_speeds(
    session_map if sample_mode else all_session_map,
    max_sessions=None
)
```
```python
with open(output_file, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` explicitly estimates per-session processing at about 15s and full saving at about 3 minutes, indicating that I/O-heavy steps dominate.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop inside `process_session`, the per-session loop in `collect_all_corridor_speeds`, and the second pass that revisits every trial to fill stimulus indices after collecting names.

ii.
```python
for trial_idx in range(ntrials):
    ...
    lick_binary = build_lick_binary(beh, sfr, gfr)
```
```python
for sess_key, (exp_type, ndb) in sessions:
    beh = load_behavior_for_session(sess_key, exp_type, ndb)
```
```python
for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        stim_idx = stim_to_idx[stim_name]
        out_data[0, :] = stim_idx
```

iii. The notes mention efficiency work around speed quartiles and runtime estimates, but they do not explicitly justify keeping these loops unvectorized.

## 12-c. What processing does the code repeat multiple times?

i. The script repeats several expensive or avoidable passes: it rebuilds the session map twice, reloads behavior once for global speed collection and again for actual conversion, and stores temporary `(trial_output, stim_name)` tuples only to do a second traversal that fills the stimulus index later.

ii.
```python
session_map = get_unique_sessions()
...
all_session_map = get_unique_sessions()
```
```python
all_speeds = collect_all_corridor_speeds(...)
...
result = process_session(...)
```
```python
output_trials.append((trial_output, stim_name))
...
for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        ...
```

iii. No explicit justification for these repeated passes was found in the notes or trajectory beyond general runtime reporting.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some work that is not needed for the final saved dataset: it imports and configures `matplotlib` unconditionally, computes unused constants such as `GREY_SPACE_DM` and `TOTAL_LENGTH_DM`, reads `LickTrind` but never uses it, and, when `--show-processing` is enabled, stores extra behavior/frame data only for plotting before discarding them.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```
```python
GREY_SPACE_DM = 20
TOTAL_LENGTH_DM = 60
```
```python
lick_trinds = beh['LickTrind']
```
```python
if show_processing:
    result['beh'] = beh
    result['start_frs'] = start_frs
    result['gray_frs'] = gray_frs
```

iii. No explicit justification was found beyond support for optional plotting and inspection.
