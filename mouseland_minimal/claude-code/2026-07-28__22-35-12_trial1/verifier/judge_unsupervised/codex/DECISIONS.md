# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script first loads `/app/data/beh/Imaging_Exp_info.npy`, then iterates over experiment types and loads each matching behavior file `Beh_<exp_type>.npy`. It builds a behavior key from `(mname, datexp, blk[, stimtype])`, keeps only the first occurrence of each `(mname, datexp, blk)` recording, and stores the matched behavior struct. Neural data are then loaded per kept session from `spk/<mname>_<datexp>_<blk>_neural_data.npy`, and retinotopy is loaded from `retinotopy/<mname>_<datexp>_trans.npz`.

ii. ```python
def collect_all_sessions(exp_info, root):
    seen_recordings = set()
    ...
    for exp_type in exp_order:
        beh_data = np.load(os.path.join(root, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
        for ndb in db_list:
            rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
            if rec_key in seen_recordings:
                continue
            ...
            sessions.append({'beh': beh_data[beh_key], 'ndb': ndb, ...})

spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. The code comments say duplicate recordings are collapsed and that experiment types are ordered to “prefer more complete annotations”; `CONVERSION_NOTES.md` justifies this as turning 23 experiment types with overlap into 89 unique recordings.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from the kept sessions. The script creates a sorted unique subject list and stores, for each session, the index of its `mname` within that sorted list.

ii. ```python
all_subjects = sorted(set(s['mname'] for s in sessions))
...
all_subject_idx.append(all_subjects.index(result['mname']))
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=int),
```

iii. The justification is implicit: the raw metadata already encode mouse identity as `mname`, so the script uses that field directly.

## 1-c. How are the data split into sessions?

i. A session is defined as a unique `(mname, datexp, blk)` tuple. If the same tuple appears in multiple experiment types, only the first one encountered after a custom sort order is kept.

ii. ```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
...
exp_order = sorted(exp_info.keys(), key=lambda x: (
    0 if 'test' in x else 1,
    0 if 'sup_' in x else 1,
    0 if 'after' in x else 1,
    x
))
```

iii. `collect_all_sessions()` explicitly says “For recordings appearing in multiple experiment types, use the first one found.” The comments justify the sort order as preferring test sessions and “more complete annotations.”

## 1-d. How are the data split into trials?

i. Trial count comes from `beh['ntrials']`. After neural activity has been interpolated into an array of shape `(neurons, trials, 40)`, the script loops over `range(n_trials)` and slices one trial at a time from neural, input, and output arrays.

ii. ```python
n_trials = beh['ntrials']
...
interp_spk = position_interpolate_spk(..., n_trials, n_bins=N_POS_BINS)
...
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
    ...
    neural_trials.append(neural_trial)
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The justification is implicit in the reference interpolation code: the paper code also reshapes interpolated activity into `(neurons, trials, positions)`.

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no trial-level quality-control filter. All trials in `beh['ntrials']` are kept. The only explicit exclusions are session-level: sessions with fewer than 2 trials are skipped, and lick events outside valid trial/bin ranges are ignored.

ii. ```python
for tr in range(n_trials):
    ...

if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue

if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```

iii. Neither `CONVERSION_NOTES.md` nor the trajectory describes any dedicated trial-rejection rule beyond the minimum-trial requirement needed by the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is derived primarily from `spk_data['spks']`, which are concatenated across list elements into a single `(neurons, frames)` array. The later filtering/alignment steps additionally use retinotopy `iarea`, behavior `ft_move`, and behavior `ft_PosCum`.

ii. ```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
iarea = ret_data['iarea']
ft_move = beh['ft_move'][:n_frames]
ft_pos_cum = beh['ft_PosCum'][:n_frames]
```

iii. `CONVERSION_NOTES.md` states that the source neural signal is “deconvolved calcium traces from Suite2p,” matching the methods excerpt and the raw `spks` files.

## 2-b. How is the `neural` data processed?

i. The script concatenates raw `spks`, filters neurons to visual cortex, restricts frames to running periods (`ft_move > 0`), linearly interpolates each neuron over evenly spaced position bins across the full session, reshapes to `(neurons, trials, 60)`, and then crops to the first 40 texture bins. The final per-trial matrices are cast to `float32`.

ii. ```python
valid_spk = spk[valid_neurons][:, vr_moving]
interp_spk = position_interpolate_spk(
    valid_spk,
    ft_pos_cum[vr_moving],
    corridor_len,
    n_trials,
    n_bins=N_POS_BINS
)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
...
neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The docstring for `position_interpolate_spk()` says it matches `utils.spk_pos_interp` / `utils.get_interpPos_spk`, and `CONVERSION_NOTES.md` says `numpy.interp` was chosen as a faster equivalent to the paper’s `scipy.interpolate.interp1d` call.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, neurons are restricted to visual-cortex retinotopy labels by excluding `iarea == -1` and `iarea == 7`. Second, only running frames are kept by requiring `ft_move > 0`.

ii. ```python
valid_mask = (iarea != -1) & (iarea != 7)
...
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
...
valid_spk = spk[valid_neurons][:, vr_moving]
```

iii. Both choices are explicitly justified in `CONVERSION_NOTES.md`, which cites the paper code’s `neu_area_ID()` / `load_retino()` logic and the methods statement that only running timepoints were analyzed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script does not align neural data by frame timestamps such as `StartFr`. Instead, it position-interpolates the full running activity stream over cumulative corridor progress, reshapes the result into trial-by-position bins, then interprets the first position bin of each trial as corridor entry.

ii. ```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
norm_pos = accum_pos / corridor_len
...
return interp_spk.reshape(n_neurons, n_trials, n_bins)
...
neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. This follows the reference utility functions in the trajectory, where `spk_pos_interp()` and `get_interpPos_spk()` likewise interpolate on cumulative position and reshape by trial count.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script assigns each converted bin a duration of `1 / 6` s = 166.67 ms, based on the paper’s 60 cm/s fixed VR speed and 1 dm spatial bins. It does not rebin native frames in time directly; instead it converts framewise data into 60 position bins and keeps the first 40 bins.

ii. ```python
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
BIN_SIZE_MS = BIN_SIZE_SEC * 1000
...
N_POS_BINS = 60
N_TEXTURE_BINS = 40
...
'time_bin_size': BIN_SIZE_MS,
```

iii. `CONVERSION_NOTES.md` explicitly justifies the 166.67 ms bin as “1 dm / 6 dm/s,” and the methods excerpt says corridor motion was fixed at 60 cm/s during running.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundPos']`, not from `SoundTime`, `SoundFr`, or trial timestamps.

ii. ```python
sound_pos = beh['SoundPos']
...
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the quantity is computed from sound-cue position because the converted representation is position-binned rather than frame-timestamp-based.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the script subtracts the cue position in dm from each bin index `i` in `0..39` and multiplies by the assumed fixed-bin duration `BIN_SIZE_SEC`. Negative values are before the cue and positive values are after it.

ii. ```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` states the exact rule: “(position_bin - sound_cue_position) * bin_size_sec,” with negative-before / positive-after semantics.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same 40 per-trial position bins as the neural matrices. The cue-time signal is not aligned from frame timestamps; it is aligned on the shared interpolated position grid.

ii. ```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail], axis=0)
```

iii. The notes justify this by treating position bins as time bins under fixed VR speed, so cue distance in position becomes cue distance in time.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` stored in the metadata entry `ndb`, grouped by mouse `mname`.

ii. ```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
...
d = compute_day_of_training(s['datexp'])
...
days = (d - mouse_first_date[s['mname']]).days
```

iii. `CONVERSION_NOTES.md` says day of training is “Days since first recording for each mouse,” which is the agent’s chosen proxy for training day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script parses each session date, finds the earliest recording date for each mouse, computes integer calendar-day offsets from that date, converts them to float, and broadcasts the session value across all 40 bins of every trial.

ii. ```python
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
...
days = (d - mouse_first_date[s['mname']]).days
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The notes justify this as a simple continuous training-progress variable. No more detailed training-day field was used.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not taken from raw trial timestamps such as `Trial_start_time` or `StartFr`. It is derived from the converted position-bin index and the assumed fixed speed constant `VR_SPEED`.

ii. ```python
BIN_SIZE_SEC = 1.0 / VR_SPEED
...
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` justifies this as “Linear from 0 to ~6.5 seconds across 40 bins,” again using the fixed-speed mapping from position to time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the script creates a 40-element ramp `0, 1*BIN_SIZE_SEC, ..., 39*BIN_SIZE_SEC`, then stacks that as the third input row.

ii. ```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
...
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail], axis=0)
```

iii. The notes describe it as a linear time axis induced by the fixed 1 dm bins.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one with the neural bins: the first neural position bin gets time 0, and each later bin gets the next fixed 166.67 ms step.

ii. ```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The justification is the same fixed-speed position-to-time conversion used throughout the agent’s format.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `beh['isRew']`.

ii. ```python
is_rew = beh['isRew'].astype(float)
```

iii. `CONVERSION_NOTES.md` explicitly says reward availability comes from `beh['isRew']`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` flag is cast to float and broadcast across all 40 bins of the trial as a per-trial constant.

ii. ```python
is_rew = beh['isRew'].astype(float)
...
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The notes justify this as “1 if rewarded corridor, 0 otherwise,” matching the decoder task’s requested per-trial input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`, with the session’s `UniqWalls` used earlier only to collect the global label set.

ii. ```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. `CONVERSION_NOTES.md` says visual stimulus categories are taken from the trial stimulus names and mapped to integer labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script builds a globally sorted list of unique stimulus names across all sessions, converts each trial’s `WallName` to its index in that list, and broadcasts the resulting label across all 40 bins of the trial.

ii. ```python
all_stim_names = sorted(all_stim_set)
...
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
...
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this as a per-trial categorical output with 15 unique stimulus names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh['LickPos']` and `beh['LickTrind']`.

ii. ```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. The notes explicitly identify `LickPos` and `LickTrind` as the source variables.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script creates a `(trials, 40)` zero array, loops over lick events, floors each lick position to an integer dm bin, and sets that trial/bin entry to 1. Multiple licks in the same bin collapse to a single 1.

ii. ```python
lick_binary = np.zeros((n_trials, n_bins), dtype=int)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

iii. `CONVERSION_NOTES.md` says the output is binary and equals 1 if any lick occurred in that position bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned on the same 40 position bins used for the neural matrices. The code does not use `LickTime` or `LickFr`; alignment is by position within corridor.

ii. ```python
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
...
lick_tr = lick_binary[tr, :].astype(np.int64)
```

iii. The notes justify this as consistent with the position-binned neural representation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is not read from trialwise raw position variables such as `ft_Pos`, `VRpos`, or `ft_PosCum`. Instead, it is derived from the index of the already interpolated 40 texture bins, with corridor geometry coming indirectly from the fixed 4 m / 40 bin convention.

ii. ```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. `CONVERSION_NOTES.md` justifies this by stating that each output bin covers 1 m of the 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script creates a fixed vector of 40 labels, one per texture bin, and reuses that same vector for every trial.

ii. ```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
...
pos_tr = pos_bins.astype(np.int64)
```

iii. The notes describe this as four equal 1 m spatial bins over the corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 40 texture bins are grouped into four categories by integer division: bins `0-9`, `10-19`, `20-29`, and `30-39` become categories `0-3`.

ii. ```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The comments in code and the notes both state the same 1 m grouping rule.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Alignment is by construction: both neural activity and position labels use the same 40 per-trial interpolated corridor bins.

ii. ```python
neural_trial = interp_spk[:, tr, :]
pos_tr = pos_bins.astype(np.int64)
```

iii. The notes say both the neural representation and the position output operate on the same 1 dm spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, which the reference materials describe as running speed interpolated into a `(trials, positions)` matrix.

ii. ```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]
```

iii. `CONVERSION_NOTES.md` says quartiles are computed from `beh['run_pos']`, and the notebook snippet in the trajectory labels `run_pos` as running speed interpolated into trial-by-position space.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code crops `run_pos` to the first 40 texture bins and then uses `np.digitize` with globally precomputed quartile edges to turn each scalar speed into one of four bin indices.

ii. ```python
run_speed = run_pos[:, :N_TEXTURE_BINS]
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. `CONVERSION_NOTES.md` says the running-speed output uses four quartile bins computed over all 89 sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the 25th, 50th, and 75th percentiles of all `run_pos[:, :40]` values concatenated across sessions. `np.digitize` then maps speeds to categories `0-3`.

ii. ```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.digitize(run_speed, speed_quantile_edges)
```

iii. The notes explicitly describe these as global quartile edges and even record the learned edge values.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Alignment is direct because `run_pos` is already a trial-by-position array and the code takes the same first 40 bins used for the neural representation.

ii. ```python
run_pos = beh['run_pos']  # (n_trials, 60) - already position-interpolated
run_speed = run_pos[:, :N_TEXTURE_BINS]
...
speed_tr = speed_bins[tr, :].astype(np.int64)
```

iii. The justification is explicit in both the code comment and `CONVERSION_NOTES.md`: running speed is already position-interpolated before discretization.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is minimal. Missing behavior files or behavior keys cause warnings and skipping. Any session-level exception is caught and skipped. Sessions with fewer than 2 trials are skipped. Lick events outside valid ranges are ignored. There is no dedicated NaN-cleaning or imputation for the variables actually used.

ii. ```python
if not os.path.exists(beh_path):
    print(f"  Warning: behavior file not found for {exp_type}")
    continue
...
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue
...
except Exception as e:
    print(f"  ERROR processing session: {e}")
    ...
    continue
...
if result['n_trials'] < 2:
    ...
```

iii. The trajectory and notes emphasize sanity checks and validation output, but they do not describe any richer missing-data policy beyond skipping obviously broken cases.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are the per-neuron interpolation loop in `position_interpolate_spk()`, repeated over very large neuron counts, and the session-by-session / trial-by-trial materialization of all neural, input, and output arrays for 89 sessions and 38,110 trials.

ii. ```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
...
for i, s in enumerate(sessions):
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
...
for tr in range(n_trials):
    ...
```

iii. The code comments and notes justify `numpy.interp` as a speed optimization, which implies interpolation was identified by the agent as the main bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several explicit Python loops could have been vectorized: the per-neuron interpolation loop, the lick-event binning loop, the `pos_bins` construction loop, the per-trial construction loop, and the repeated `all_stim_names.index(...)` lookup list comprehension.

ii. ```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(...)
...
for i in range(len(lick_pos)):
    ...
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
for tr in range(n_trials):
    ...
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The trajectory shows the agent already optimized away the reference code’s chunked interpolation helper but kept these remaining Python loops.

## 12-c. What processing does the code repeat multiple times?

i. The code recomputes date parsing repeatedly, constructs a placeholder day-of-training row only to overwrite it later, rebuilds the same `pos_bins` array for every session, repeatedly does `list.index()` subject and stimulus lookups, and allocates fresh small arrays trial-by-trial.

ii. ```python
session_date = compute_day_of_training(datexp)
...
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
...
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. No explicit justification is given for these repetitions; they appear to be convenience choices in the implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is constructing deterministic outputs that add no trial-specific information: `position` is the same 40-label vector in every trial, and several per-trial quantities are first built in placeholder form and then overwritten. Broadcasting per-trial constants (`day_of_training`, `reward_availability`, `visual_stimulus`) across all 40 bins also inflates storage relative to their informational content.

ii. ```python
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
...
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
pos_tr = pos_bins.astype(np.int64)
```

iii. The trajectory’s later note that “position is deterministic (same for all trials)” is the clearest agent-authored acknowledgment that at least one output was mechanically generated rather than learned from trial-varying raw data.
