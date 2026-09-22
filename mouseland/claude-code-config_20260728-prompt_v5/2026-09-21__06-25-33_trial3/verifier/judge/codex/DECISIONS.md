# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `/app/data/beh/Imaging_Exp_info.npy` first, builds a unique-session dictionary keyed by `mname_datexp_blk`, then loads behavior, spike, and retinotopy files on demand per session. Behavior is found by searching across the session's experiment types and preferring entries without `stimtype`; spike and retinotopy files are loaded directly from the session key. Unlike the reference solution, behavior files are reopened repeatedly rather than grouped and read once.

ii. ```python
def build_session_list():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    ...
    key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"

def find_behavior_for_session(session_key, sessions_info, exp_info):
    ...
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()
    ...

def load_neural_data(mname, datexp, blk):
    path = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    data = np.load(path, allow_pickle=True).item()

def load_retinotopy(mname, datexp):
    path = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    data = np.load(path, allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent states that each unique `(mname, datexp, blk)` is one session and that repeated entries across experiment types should be collapsed. In the trajectory it justifies the behavior lookup by saying swap/test3 sessions share the same physical recording, so it can load a session once and use any matching behavior file, preferring the simpler key without `stimtype`.

## 1-b. How are the data split into subjects?

i. Subjects are defined by mouse name (`mname`). The final `subjects` list is the sorted set of mouse names from processed sessions, and `subject_idx` maps each session to its subject.

ii. ```python
all_subjects = sorted(set(r['mname'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx.append(subject_to_idx[result['mname']])
```

iii. The notes repeatedly describe sessions as grouped by mouse and report 19 mice, so the subject split is taken directly from the session metadata rather than inferred any other way.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` tuple. `build_session_list()` collapses repeated appearances of the same physical session across experiment types into one session key, and the main loop processes `sorted(sessions_info.keys())`.

ii. ```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in sessions:
    sessions[key] = {
        'mname': ndb['mname'],
        'datexp': ndb['datexp'],
        'blk': ndb['blk'],
        'exp_types': [],
        'ndb_list': [],
    }
...
all_session_keys = sorted(sessions_info.keys())
```

iii. The notes say there are 142 index entries but 89 unique sessions, and the trajectory says the same physical session appears in multiple experiment types and should be loaded once.

## 1-d. How are the data split into trials?

i. Trials are enumerated with `for t in range(ntrials)`, but the per-trial frame window is not the full corridor traversal. The agent defines trial frames as those where `ft_trInd == trial`, `ft_CorrSpc` is true, and `ft_move > 0`, then drops any trial with fewer than 2 retained frames.

ii. ```python
def extract_trial_frames(beh, trial_idx, n_neural_frames):
    ...
    trial_mask = valid & (ft_trInd.astype(float) == trial_idx)
    corr_mask = ft_CorrSpc.astype(bool)
    move_mask = ft_move > 0
    frame_mask = trial_mask & corr_mask & move_mask
    return np.where(frame_mask)[0]

for t in range(ntrials):
    frame_idx = extract_trial_frames(beh, t, n_neural_frames)
    if len(frame_idx) < 2:
        continue
```

iii. `CONVERSION_NOTES.md` says the decoder should use only running frames in the texture corridor because the paper's analyses only considered running frames and the decoder outputs are corridor-defined. The trajectory repeats that choice explicitly: `ft_trInd == trial AND ft_CorrSpc AND ft_move > 0`.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies only minimal trial-quality filters: it skips trials with fewer than 2 retained frames after corridor-and-running filtering, and later skips any session with fewer than 2 surviving trials. It does not implement the reference solution's global 99th-percentile long-trial filter.

ii. ```python
for t in range(ntrials):
    frame_idx = extract_trial_frames(beh, t, n_neural_frames)
    if len(frame_idx) < 2:
        continue
    ...

if len(neural_trials) < 2:
    print(f"  WARNING: Session {session_key} has {len(neural_trials)} valid trials, skipping")
    return None
```

iii. The notes treat `ft_move > 0` plus corridor restriction as the main curation step and mention the decoder requirement that each session needs at least two trials. No separate long-trial outlier rule is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the concatenated `spks` arrays in the session's spike file. Brain-region assignments and neuron filtering come from `iarea` in the corresponding retinotopy file.

ii. ```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)
...
data = np.load(path, allow_pickle=True)
return data['iarea']
```

iii. The notes identify the spike files as Suite2p deconvolved fluorescence traces and the retinotopy files as the source of coarse visual-area labels.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, optionally clips spike and retinotopy arrays to the same neuron count if they mismatch, filters neurons to the visual-cortex mask, slices the selected trial frames, and stores each trial as `float32`. It does not compute dF/F or perform any temporal resampling.

ii. ```python
spk = load_neural_data(mname, datexp, blk)
...
if len(iarea) != n_neurons_total:
    min_n = min(len(iarea), n_neurons_total)
    iarea = iarea[:min_n]
    spk = spk[:min_n]
...
spk_filtered = spk[neuron_mask]
...
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. The notes say the paper already provides deconvolved fluorescence and that the native imaging frame should be used directly. The trajectory also frames the work as extracting the already processed calcium traces rather than reprocessing them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neuron filtering is based on retinotopy only: neurons with `iarea == -1` or `iarea == 7` are excluded, and the rest are assigned to one of four visual regions. If spike and retinotopy neuron counts disagree, the agent clips both arrays to their minimum length before filtering.

ii. ```python
mask = (iarea != -1) & (iarea != 7)
...
for iarea_val, region_name in AREA_MAP.items():
    region_idx[valid_iarea == iarea_val] = BRAIN_REGIONS.index(region_name)
...
if len(iarea) != n_neurons_total:
    min_n = min(len(iarea), n_neurons_total)
    iarea = iarea[:min_n]
    spk = spk[:min_n]
```

iii. The notes say the reference code excludes only neurons outside the four visual areas and does not add a further cell-quality filter. The mismatch clipping is justified in the notes as an edge-case safeguard.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata names the alignment event as corridor entry (trial start), but the actual per-trial neural arrays are built from the subset of corridor frames that also satisfy `ft_move > 0`. Each trial therefore begins at the first retained moving corridor frame for that trial, not necessarily at the first corridor frame after entry.

ii. ```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
...
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
'metadata': {
    'temporal_alignment_event': 'Corridor entry (trial start)',
    'off_start': 0.0,
    'off_end': None,
    ...
}
```

iii. The notes say the intended alignment is corridor entry, but also insist on keeping only running frames in the corridor. That combination is how the agent justified its final alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each retained imaging frame is one time bin. The agent estimates the per-session frame interval as the median difference in `ft`, then stores the dataset metadata `time_bin_size` as the mean frame interval across sessions, in milliseconds.

ii. ```python
frame_dt_days = np.nanmedian(np.diff(ft))
frame_dt_sec = frame_dt_days * 24 * 3600
...
frame_dts = [r['frame_dt_sec'] for r in session_results]
mean_frame_dt = np.mean(frame_dts)
...
'time_bin_size': mean_frame_dt * 1000,
```

iii. The notes report a native frame rate of about 3.17 Hz and explicitly state that one calcium-imaging frame is the target time bin.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `time_to_sound_cue` is derived from the trial's `SoundFr` value and the imaging-frame timing inferred from `ft`.

ii. ```python
ft = beh['ft']
frame_dt_days = np.nanmedian(np.diff(ft))
frame_dt_sec = frame_dt_days * 24 * 3600
...
sound_fr = beh['SoundFr'][t]
```

iii. The notes map this input to `SoundFr` together with the frame timing information from `ft`, describing it as a frame difference converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the agent computes `(frame_idx - sound_fr) * frame_dt_sec` and stores that time series as the first input channel. This uses a constant per-session frame duration rather than interpolating `SoundFr` onto an explicit timestamp axis.

ii. ```python
sound_fr = beh['SoundFr'][t]
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
...
inp[0, :] = time_to_cue.astype(np.float32)
```

iii. The notes say this variable should be a frame-difference-to-seconds conversion, and the code comment says it is represented with negative values before the cue and positive values after it.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same `frame_idx` array that is used to extract the neural trial, so its length matches the retained neural frames for that trial.

ii. ```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. The notes and trajectory both describe the behavioral inputs as being aligned by frame number to the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from the session date string `datexp`, grouped by mouse name `mname` via the unique session key.

ii. ```python
def compute_day_of_training(sessions_info):
    mouse_dates = defaultdict(list)
    for key, info in sessions_info.items():
        mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))
```

iii. The notes describe this variable as date-derived from the session names and grouped within each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent parses each session date, sorts sessions within each mouse, defines day 0 as the first session date, and stores the elapsed calendar days since that first date. That scalar is then broadcast across all retained frames of each trial.

ii. ```python
for mname, date_list in mouse_dates.items():
    date_list.sort(key=lambda x: x[1])
    first_date = date_list[0][1]
    for key, dt in date_list:
        day_map[key] = (dt - first_date).days
...
inp[1, :] = day_val
```

iii. `CONVERSION_NOTES.md` explicitly says `day 0 = first session date` and `subsequent sessions = days since first session`, which is the rationale the agent used instead of counting recording sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `time_since_trial_start` is derived from each trial's `StartFr` value together with the frame timing inferred from `ft`.

ii. ```python
ft = beh['ft']
frame_dt_days = np.nanmedian(np.diff(ft))
frame_dt_sec = frame_dt_days * 24 * 3600
...
start_fr = beh['StartFr'][t]
```

iii. The notes map this variable directly to `StartFr` and describe it as a frame-to-seconds conversion relative to trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the agent computes `(frame_idx - start_fr) * frame_dt_sec` and stores that as the third input channel. Like `time_to_sound_cue`, this uses a constant per-session frame duration rather than interpolated timestamps.

ii. ```python
start_fr = beh['StartFr'][t]
time_since_start = (frame_idx - start_fr) * frame_dt_sec
...
inp[2, :] = time_since_start.astype(np.float32)
```

iii. The notes justify this as a simple time difference to the floating-point trial-start frame, expressed in seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained `frame_idx` values used for the neural trial, so it is frame-for-frame aligned with the neural matrix.

ii. ```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. The notes state that all streams are aligned in frame space, and the code follows that pattern.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `reward_availability` is derived directly from the per-trial `isRew` flag.

ii. ```python
reward_avail = float(beh['isRew'][t])
```

iii. The notes describe this as a direct readout of whether the trial was in a rewarded corridor.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent casts the per-trial boolean to float and broadcasts that constant across all retained frames of the trial.

ii. ```python
reward_avail = float(beh['isRew'][t])
...
inp[3, :] = reward_avail
```

iii. The notes treat this variable as already available in the behavior data with no extra transformation required.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The visual-stimulus output is derived from the per-trial `WallName` strings.

ii. ```python
wall_names = beh['WallName']
...
stim_name = wall_names[t]
```

iii. The notes say to use `WallName` directly rather than other masked stimulus fields.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent does not collapse wall names to the four base textures. Instead, it collects all unique `WallName` strings across sessions, sorts them, assigns each one an index, and broadcasts that index across all time bins of the trial. In practice this yields 15 stimulus classes.

ii. ```python
output_trials.append((out, stim_name))
...
all_stimuli = set()
for result in session_results:
    for out_data, stim_name in result['output']:
        all_stimuli.add(stim_name)
all_stimuli = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
...
out_filled[0, :] = stim_to_idx[stim_name]
```

iii. `CONVERSION_NOTES.md` explicitly plans to `Use WallName directly` and to `Pool all unique stimuli across sessions`, and later reports `15 unique stimuli found`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind` in the behavior struct.

ii. ```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
```

iii. The notes identify these as the session's lick-frame list plus per-lick trial indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Within each trial, the agent selects licks whose `LickTrind` matches the trial, rounds the corresponding `LickFr` values to the nearest integer frame, turns those into a set, and emits a binary vector marking retained frames that appear in that set.

ii. ```python
trial_lick_mask = lick_trind == trial_idx
trial_lick_fr = lick_fr[trial_lick_mask]
...
lick_int_frames = np.round(trial_lick_fr).astype(int)
lick_set = set(lick_int_frames)
binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices],
                  dtype=np.float32)
```

iii. The notes say licking should be `1 if lick at this frame, 0 otherwise`, and later list `LickFr rounded to nearest integer` as an explicit edge-case choice.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is constructed over the same `frame_idx` array used to extract the neural trial, so it has the same length and frame alignment as the neural data.

ii. ```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
licking = make_lick_binary(beh, frame_idx, t)
```

iii. The notes describe all output streams as aligned by imaging frame to the neural arrays.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the framewise `ft_Pos` signal.

ii. ```python
positions = beh['ft_Pos'][:n_use]
trial_positions = positions[frame_idx[frame_idx < n_use]]
```

iii. The notes identify `ft_Pos` as the decimeter-scale position within the corridor for each imaging frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent clips positions to the texture range `[0, 40)` decimeters and discretizes them with edges `[0, 10, 20, 30, 40]`, producing four 1 m bins. If the behavior vector is shorter than the retained frame list, it pads with the last valid position.

ii. ```python
pos_clipped = np.clip(positions, 0, TEXTURE_LENGTH - 1e-6)
bins = np.digitize(pos_clipped, bin_edges) - 1
bins = np.clip(bins, 0, N_POS_BINS - 1)
...
if len(trial_positions) < len(frame_idx):
    trial_positions = np.concatenate([
        trial_positions,
        np.full(pad_len, trial_positions[-1] if len(trial_positions) > 0 else 0)
    ])
```

iii. The notes justify this as four equal 1 m spatial bins defined over the 4 m texture corridor, with padding added to survive small frame-count mismatches.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are fixed at 10 decimeters per category: `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`.

ii. ```python
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])
...
'output_values': [
    all_stimuli,
    ['not_licking', 'licking'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    speed_labels,
]
```

iii. The notes explicitly describe four equal-length 1 m spatial bins over the texture section.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained `frame_idx` values used for the neural trial; if the behavior signal is shorter than the neural frame list, the final valid position is repeated to preserve length matching.

ii. ```python
trial_positions = positions[frame_idx[frame_idx < n_use]]
...
pos_bins = discretize_position(trial_positions)
...
out[2, :] = pos_bins
```

iii. The notes say all streams should stay frame-aligned to the neural data, and they document padding as the fallback when frame counts differ slightly.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the framewise `ft_RunSpeed` signal.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
...
run_speeds = beh['ft_RunSpeed'][:n_use_speed]
```

iii. The notes map this output directly to the session's running-speed trace.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first collects running speeds from all sessions using frames with `ft_CorrSpc & (ft_move > 0)`, optionally subsamples each session to at most 10,000 values for speed, computes global 25/50/75 percentile thresholds, and then discretizes each trial's retained speeds with `np.digitize`. If the behavior vector is shorter than the retained frame list, it pads with the last valid speed.

ii. ```python
def collect_all_running_speeds(...):
    ...
    mask = ft_CorrSpc & (ft_move > 0)
    speeds = ft_RunSpeed[mask]
    if len(speeds) > 10000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(speeds), 10000, replace=False)
        speeds = speeds[idx]
    ...
    quartiles = np.percentile(all_speeds, [25, 50, 75])

def discretize_speed(speeds, quartile_edges):
    bins = np.digitize(speeds, quartile_edges)
    return bins.astype(np.int64)
```

iii. The notes say the goal was to compute quartiles across all sessions for consistent decoder binning and later report that the resulting bins are roughly equally populated.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The four categories are defined by three global percentile thresholds, producing bins `<q1`, `q1-q2`, `q2-q3`, and `>q3` as recorded in `output_values`.

ii. ```python
speed_labels = [
    f'<{speed_quartiles[0]:.1f}',
    f'{speed_quartiles[0]:.1f}-{speed_quartiles[1]:.1f}',
    f'{speed_quartiles[1]:.1f}-{speed_quartiles[2]:.1f}',
    f'>{speed_quartiles[2]:.1f}',
]
```

iii. The notes explicitly say to compute quartiles across all sessions and to label the bins from those numeric edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained `frame_idx` values used for the neural trial; if the speed vector is shorter than that frame list, the final valid value is repeated so the lengths still match.

ii. ```python
trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
...
speed_bins = discretize_speed(trial_speeds, speed_quartiles)
...
out[3, :] = speed_bins
```

iii. The notes frame running speed as another frame-aligned output stream and document padding as the edge-case behavior for short behavior vectors.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent adds several defensive fixes: if behavior cannot be found it skips the session; if retinotopy and spike neuron counts disagree it clips both to the minimum; it ignores non-finite `ft_trInd` values; it drops trials with fewer than 2 retained frames; if position or speed arrays are shorter than retained neural frames it pads with the last valid value; and if there are no licks it returns an all-zero lick vector.

ii. ```python
if beh is None:
    print(f"  WARNING: No behavior found for {session_key}, skipping")
    return None
...
if len(iarea) != n_neurons_total:
    min_n = min(len(iarea), n_neurons_total)
    iarea = iarea[:min_n]
    spk = spk[:min_n]
...
valid = np.isfinite(ft_trInd)
...
if len(frame_idx) < 2:
    continue
...
if len(trial_positions) < len(frame_idx):
    trial_positions = np.concatenate([...])
if len(trial_speeds) < len(frame_idx):
    trial_speeds = np.concatenate([...])
```

iii. The notes' edge-case section explicitly mentions handling spike/behavior off-by-one mismatches, short trials, and lick rounding, and the code generalizes those safeguards further.

## 12-a. What are the most time-consuming steps of the code?

i. From the agent's own notes, the expensive parts are the global speed-quartile pass and the per-session conversion over the very large spike files. In the code, the dominant costs are repeated large-file reads: spike files during session processing, and repeated behavior-file loads during speed collection and behavior lookup.

ii. ```python
def collect_all_running_speeds(...):
    for session_key in session_keys:
        beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
        ...

def process_session(...):
    spk = load_neural_data(mname, datexp, blk)
    ...
```

iii. `CONVERSION_NOTES.md` reports `Speed quartile collection: 7.6s`, `Processing: ~5 s/GB of spike data`, and a total spike-data size of hundreds of GB, which is the stated rationale for the runtime profile.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar/Python-level: the per-trial loop across every session, the per-trial frame extraction loop across all trials, the lick construction list comprehension over every retained frame, the all-session speed scan, and the output postprocessing loop that fills stimulus indices. None of these are vectorized away.

ii. ```python
for t in range(ntrials):
    frame_idx = extract_trial_frames(beh, t, n_neural_frames)
    ...

binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices],
                  dtype=np.float32)

for session_key in session_keys:
    ...

for result in session_results:
    for out_data, stim_name in result['output']:
        all_stimuli.add(stim_name)
```

iii. The trajectory only justifies one of these loops explicitly: it says the agent will `collect all speeds in a single pass` before discretizing. It does not provide a stronger optimization rationale for the other Python loops.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly reloads behavior files. `find_behavior_for_session()` opens behavior `.npy` files each time it is called, and it is called once per session during global speed collection, again during session processing, and again for plotting. The code also recomputes clipping/indexing logic for position and speed inside every trial loop.

ii. ```python
def find_behavior_for_session(session_key, sessions_info, exp_info):
    ...
    beh_data = np.load(beh_file, allow_pickle=True).item()
    ...

def collect_all_running_speeds(...):
    beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)

def process_session(...):
    beh, exp_type = find_behavior_for_session(session_key, sessions_info, exp_info)

def show_processing_plots(...):
    beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
```

iii. The notes justify the extra global pass because the agent wanted one set of speed quartiles for the whole dataset, but they do not justify the repeated reopening of behavior files.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes optional plotting machinery, carries `stim_name` strings alongside each output array only to replace them later with integer indices, writes placeholder `-1` stimulus values before a second pass fills them in, and computes an unused local `bin_edges` array inside `discretize_speed()`. These steps do not contribute directly to the final decoder inputs once the output dictionary is assembled.

ii. ```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
...
out = np.zeros((4, n_tp), dtype=np.int64)
out[0, :] = -1  # placeholder, will be filled with stimulus index
...
output_trials.append((out, stim_name))
...
def discretize_speed(speeds, quartile_edges):
    bin_edges = np.array([-np.inf, quartile_edges[0], quartile_edges[1], quartile_edges[2], np.inf])
    bins = np.digitize(speeds, quartile_edges)
```

iii. The notes emphasize validation plots and diagnostics, so the extra plotting code is intentional for debugging, but it is not needed by the final downstream analyses.
