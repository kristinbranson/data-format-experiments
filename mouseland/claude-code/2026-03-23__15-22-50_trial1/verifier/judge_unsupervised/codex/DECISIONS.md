# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent first loads `/app/data/beh/Imaging_Exp_info.npy`, builds a unique-session map keyed by `mname_datexp_blk`, and then processes each session by loading three streams: neural data from `data/spk`, behavior from the matching `Beh_<exp_type>.npy`, and retinotopy from `data/retinotopy`. If the plain session key is missing from a behavior file, it falls back to `session_key + "_" + stimtype`.

ii. 
```python
def get_unique_sessions():
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()
    session_map = {}
    for exp_type in exp_info:
        for ndb in exp_info[exp_type]:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_map:
                session_map[key] = (exp_type, ndb)

def load_behavior_for_session(session_key, exp_type, ndb):
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()
    if session_key in beh_data:
        return beh_data[session_key]
    stimtype = ndb.get('stimtype', None)
    if stimtype:
        key_with_stim = f"{session_key}_{stimtype}"
        if key_with_stim in beh_data:
            return beh_data[key_with_stim]

for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(sess_key, exp_type, ndb, ...)
```

iii. In `CONVERSION_NOTES.md`, the agent says it wanted to match the paper’s 89 physical recordings and therefore deduplicated repeated experiment-type entries down to unique sessions while preserving `stimtype` fallback for behavior lookups.

## 1-b. How are the data split into subjects?

i. Subjects are defined by mouse name `mname`. After session processing, the agent collects the unique subject names, sorts them, and stores a `subject_idx` per session.

ii. 
```python
mname = ndb['mname']

all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

for result in session_results:
    subject_idx.append(subject_to_idx[result['subject']])
```

iii. The notes justify this as matching the paper’s 19 mice and the required target format fields `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique physical recordings identified by `mname`, `datexp`, and `blk`, not by experiment-type entries. If the same recording appears in multiple experiment types, only the first encountered entry is kept.

ii. 
```python
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
        if key not in session_map:
            session_map[key] = (exp_type, ndb)
```

iii. The notes explicitly say: “Each physical session used once,” motivated by the paper’s count of 89 recordings rather than 142 experiment-type entries.

## 1-d. How are the data split into trials?

i. Trials are defined by frame windows from rounded `StartFr` to rounded `GrayFr`, so each trial spans corridor entry through entry into gray space exclusion, effectively keeping the textured corridor segment only.

ii. 
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)

for trial_idx in range(ntrials):
    sfr = start_frs[trial_idx]
    gfr = gray_frs[trial_idx]
    n_trial_frames = gfr - sfr
    trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The notes justify this as aligning to trial start and “captur[ing] full textured corridor portion,” which the agent considered the relevant interval for the requested decoder variables.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies only minimal trial validity checks. It skips trials with negative start frames, end frames beyond available data, non-positive width, or fewer than 2 frames. It does not apply behavioral quality filtering beyond this.

ii. 
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue

if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. The notes say there was no reference trial curation rule to reproduce, so only clearly invalid frame ranges were removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` list stored in each `{mouse}_{date}_{block}_neural_data.npy` file, concatenated across imaging planes.

ii. 
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The notes cite the reference `load_spk` function and state that the dataset already contains deconvolved fluorescence traces, so no dF/F computation was added.

## 2-b. How is the `neural` data processed?

i. The agent keeps the already deconvolved traces, concatenates planes, slices them into trial windows, and casts each trial array to `float16` for storage. It does not do further denoising, interpolation, temporal smoothing, or rebinning.

ii. 
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The justification in the notes is that the reference data are already “deconvolved fluorescence traces” and that storing them as `float16` reduces file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not filter neurons at all, and it does not apply the reference analysis mask of running-only corridor frames. The only implicit temporal restriction is the `StartFr:GrayFr` trial window.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. In the notes, the agent justified the no-neuron-filter decision by saying the reference code uses all Suite2p-detected neurons. It also claimed “all trials included,” but it did not implement the paper/reference-code restriction to running timepoints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry. The first frame of each trial is `StartFr`, and `metadata['temporal_alignment_event']` is set to “Trial start (corridor entry)” with `off_start = 0.0`.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)

'metadata': {
    'temporal_alignment_event': 'Trial start (corridor entry)',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. This follows the task instructions directly; the notes repeatedly describe the chosen alignment event as corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the original imaging frame interval, computed per session from the median difference of `beh['ft']`, about 314.7 ms. No temporal rebinning is applied.

ii. 
```python
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600

'time_bin_size': float(median_dt * 1000),
```

iii. The notes say this matches the calcium imaging frame rate reported as about 3.17 Hz and that the agent intentionally kept the native frame bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr`, `StartFr`, and the per-frame index within the trial, plus the frame duration inferred from `ft`.

ii. 
```python
sound_frs = beh['SoundFr']
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The notes describe this as the continuous, time-varying representation of time relative to the cue required by the decoder task.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the agent subtracts the frame index from the trial’s `SoundFr` and converts frame units to seconds. Positive values mean before the cue and negative values mean after the cue.

ii. 
```python
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The notes explicitly justify the sign convention as “positive before cue.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the exact same per-trial frame indices as the neural slice `spk[:, sfr:gfr]`, so input frame `t` corresponds to neural frame `t`.

ii. 
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
trial_neural = spk[:, sfr:gfr].astype(np.float16)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The notes emphasize that all time-varying inputs and outputs were built on the same trial frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from per-session metadata in `Imaging_Exp_info.npy`, specifically `mname` and `datexp`.

ii. 
```python
for sess_key, (exp_type, ndb) in session_map.items():
    mname = ndb['mname']
    datexp = ndb['datexp']
    date = datetime.strptime(datexp, '%Y_%m_%d')
```

iii. The notes say there is no direct “day of training” variable in the raw behavior streams, so the agent inferred it from session dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are grouped by mouse, sorted by calendar date, and assigned the integer day difference from that mouse’s first recording day. The scalar is then broadcast across all frames in the trial.

ii. 
```python
sessions.sort(key=lambda x: x[1])
first_date = sessions[0][1]
for sess_key, date in sessions:
    day_map[sess_key] = (date - first_date).days

np.full(n_trial_frames, day_val, dtype=np.float32)
```

iii. The notes justify this as a continuous per-trial training-progress variable. They explicitly chose calendar-day difference rather than session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr`, per-frame indices, and the frame duration from `ft`.

ii. 
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes describe this as the straightforward continuous timebase anchored to corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, the agent subtracts the trial’s start frame index and converts the result from frames to seconds.

ii. 
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The justification is simply that the decoder input spec asked for a time-varying “Time since trial start.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same `frame_indices` used to slice neural data, so each timepoint lines up exactly with the corresponding neural frame within the trial.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes say all trialwise arrays share the same frame axis.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii. 
```python
is_rew = beh['isRew']
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes justify this as directly matching the decoder input definition “1 if in rewarded corridor, 0 if not.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts `isRew` to `0.0/1.0` and broadcasts that scalar across every frame in the trial.

ii. 
```python
np.full(n_trial_frames, rew_val, dtype=np.float32)
```

iii. The notes explicitly call this a per-trial input rather than a time-varying event series.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, not from `stim_id` or `TrialStim`.

ii. 
```python
wall_names = beh['WallName']
stim_name = wall_names[trial_idx]
```

iii. The notes justify this by saying the decoder target should be the actual corridor label, including variants such as `leaf1_swap1`, and they explicitly preferred `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent collects all unique `WallName` values across sessions, sorts them, maps each name to an integer category, and then fills the first output row with that category for every frame in the trial.

ii. 
```python
all_stim_names = set()
for result in session_results:
    for (_, stim_name) in result['output']:
        all_stim_names.add(stim_name)
stim_names = sorted(all_stim_names)
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}

out_data[0, :] = stim_idx
```

iii. The notes say this produces a consistent categorical encoding across all sessions and matches the requested output examples better than the reference helper `get_cat_id`, which collapses stimuli into reward-related categories for a different analysis.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` within each trial’s frame range. The code reads `LickTrind`, but does not actually use it.

ii. 
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```

iii. The notes justify the choice as frame-based licking aligned to neural frames; they also state that this is “equivalent to `LickTrind`,” although the implementation does not use `LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent rounds `LickFr` to integer frames, selects lick frames falling inside the trial, converts them to trial-relative frame offsets, clips them to bounds, and writes `1.0` into a binary lick vector.

ii. 
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
frame_offsets = lick_frs_int[mask] - start_fr
frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
lick_binary[frame_offsets] = 1.0
```

iii. The notes say the requested decoder output was binary, time-varying licking, so multiple licks within a frame were intentionally collapsed to a single 1.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned on the same `StartFr:GrayFr` frame window as neural data, with trial-relative indices defined by subtracting `start_fr`.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. The notes justify this as direct frame-level alignment to the imaging frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the per-frame position series `ft_Pos`.

ii. 
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[sfr:gfr]
```

iii. The notes say `ft_Pos` is the natural frame-aligned position variable for time-varying decoder outputs.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent slices `ft_Pos` to the trial window and discretizes the resulting values into four bins over the textured corridor range 0-40 dm. It does not interpolate position or restrict to running-only frames.

ii. 
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)

trial_pos = ft_pos[sfr:gfr]
pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)
```

iii. The notes justify this as implementing the decoder spec “4 equal 1-m bins” directly from the raw frame-aligned position stream.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is binned using equally spaced edges over 0-40 dm, producing four 10-dm categories corresponding to 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. 
```python
bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes explicitly list the binning as `0-10, 10-20, 20-30, 30-40 dm`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position series is sliced with the same `sfr:gfr` interval as neural data, so each position bin is frame-aligned to the same neural timepoint.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_pos = ft_pos[sfr:gfr]
```

iii. The notes repeatedly say inputs and outputs were all constructed on the same trial frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. The notes identify `ft_RunSpeed` as the frame-aligned running-speed variable and map it directly to the decoder output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first gathers all finite `ft_RunSpeed` values from frames where `ft_CorrSpc` is true, computes global quartile cutoffs, then slices each trial’s speed series and bins it with those cutoffs. It does not enforce the paper’s running-only analysis mask.

ii. 
```python
def collect_all_corridor_speeds(session_map, max_sessions=None):
    corr_mask = beh['ft_CorrSpc'][:n_frames]
    speeds = beh['ft_RunSpeed'][:n_frames]
    all_speeds.append(speeds[corr_mask])

quartiles = np.percentile(valid, [25, 50, 75])
trial_speed = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles).astype(np.float32)
```

iii. The notes justify this as matching the decoder instruction to create four bins each containing 25% of the data, and they documented a later `right=True` fix for zero-valued speeds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized by quartiles. Values are digitized against the global 25th, 50th, and 75th percentile thresholds, with `right=True` and clipping to `0..3`.

ii. 
```python
def discretize_speed(speed, quartiles):
    binned = np.digitize(speed, quartiles, right=True)
    binned = np.clip(binned, 0, 3)
    return binned
```

iii. The notes say the quartile binning was chosen to satisfy the task specification and later adjusted so exact-zero values fell into the lowest quartile.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned frame-for-frame with neural data by slicing the same trial window `sfr:gfr`.

ii. 
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_speed = ft_run_speed[sfr:gfr]
```

iii. The notes describe speed as another frame-aligned behavioral output built on the neural frame timestamps.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few specific edge cases: fallback to `stimtype`-qualified behavior keys, truncation to the minimum of neural and behavior frame counts, skipping invalid or too-short trials, and dropping non-finite values when computing speed quartiles. It does not add broader imputation or NaN-repair logic inside trials.

ii. 
```python
n_frames = min(n_frames_neural, n_frames_beh)

if session_key in beh_data:
    return beh_data[session_key]
...
if stimtype:
    key_with_stim = f"{session_key}_{stimtype}"

if sfr < 0 or gfr > n_frames or sfr >= gfr:
    continue

valid = all_speeds[np.isfinite(all_speeds)]
```

iii. The notes justify this as “sensible defaults” for minor dataset issues; they specifically mention one skipped invalid trial and a speed-binning fix, but no more extensive missing-data treatment.

## 12-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are loading and slicing the very large neural arrays session by session, the first pass that scans all behavior files to compute speed quartiles, and writing the final 148 GB pickle to disk.

ii. 
```python
all_speeds = collect_all_corridor_speeds(...)

for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(...)

with open(output_file, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly estimate per-session processing time, quartile-computation time, and save time, and report that the full conversion took about 33.3 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the Python loop over trials inside `process_session`, the session loop used to build the all-speed array, and the later loop that revisits every trial just to replace stimulus-name placeholders with integer category IDs.

ii. 
```python
for trial_idx in range(ntrials):
    ...

for sess_key, (exp_type, ndb) in sessions:
    ...

for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        stim_idx = stim_to_idx[stim_name]
        out_data[0, :] = stim_idx
```

iii. The notes discuss runtime and speedups, but the code still relies on repeated Python loops for work that could mostly be array-based.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats session discovery by calling `get_unique_sessions()` twice, loads behavior once for the global speed-quartile pass and again during actual session processing, and stores `(trial_output, stim_name)` tuples only to revisit them later and overwrite the placeholder stimulus row.

ii. 
```python
session_map = get_unique_sessions()
all_session_map = get_unique_sessions()

all_speeds = collect_all_corridor_speeds(...)
...
result = process_session(...)

output_trials.append((trial_output, stim_name))
...
for i, (out_data, stim_name) in enumerate(result['output']):
    out_data[0, :] = stim_idx
```

iii. The notes acknowledge the separate quartile pass and the later stimulus-fill pass indirectly through their runtime discussion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code carries around extra plotting/debug state in `show_processing` mode (`beh`, `start_frs`, `gray_frs`), creates temporary `(output_array, stim_name)` tuples before discarding the names, and spends time generating plots that are not part of the final converted dataset or downstream decoder inputs.

ii. 
```python
output_trials.append((trial_output, stim_name))
...
result['output'][i] = out_data

if show_processing:
    result['beh'] = beh
    result['start_frs'] = start_frs
    result['gray_frs'] = gray_frs
...
plot_processing(result, i, speed_quartiles, stim_to_idx)
```

iii. The notes justify these steps as validation aids, not as part of the actual dataset used downstream.
