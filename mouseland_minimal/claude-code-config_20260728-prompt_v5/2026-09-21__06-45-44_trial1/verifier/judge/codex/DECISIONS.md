# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `beh/Imaging_Exp_info.npy` as the master index, deduplicates sessions by `(mname, datexp, blk)`, and keeps the first experiment-type entry for each unique recording. It loads behavior from the corresponding `Beh_<exp_type>.npy` file, and loads spikes and retinotopy per session.

ii. 
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
    allow_pickle=True
).item()

unique_sessions = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if key not in unique_sessions:
            unique_sessions[key] = (exp_type, ndb)
```
```python
if exp_type not in beh_cache:
    beh_cache[exp_type] = np.load(
        os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
        allow_pickle=True
    ).item()
```
```python
spk = load_spk(mname, datexp, blk)
iarea = load_iarea(mname, datexp)
```

iii. In the trajectory the AI explicitly decided to include “all 89 unique sessions across all experiment types” after initially considering a narrower subset. It also justified deduplication because the same recording can appear under multiple experiment types.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. The unique mouse names are sorted to form `subjects`, and each session stores the matching subject index.

ii. 
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_dates.setdefault(mname, []).append(date)

subjects = sorted(mouse_first_date.keys())
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```
```python
data_subject_idx.append(subject_to_idx[mname])
```

iii. The trajectory shows the AI recognized mouse identity as the natural subject split and reported 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is defined by the tuple `(mname, datexp, blk)`. If the same tuple appears in more than one experiment type, only the first occurrence is kept.

ii. 
```python
key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if key not in unique_sessions:
    unique_sessions[key] = (exp_type, ndb)

session_keys = sorted(unique_sessions.keys())
```

iii. The AI justified this in the trajectory by noting that several behavior files reference the same physical recording session and that double-counting should be avoided.

## 1-d. How are the data split into trials?

i. The AI does not use frame-wise trial windows such as `ft_trInd`, `StartFr`, `GrayFr`, or `EndFr`. Instead, it takes `beh['ntrials']`, filters to running frames, position-interpolates all running activity continuously, and reshapes the interpolated result into `(n_neurons, ntrials, 40)`. Each trial is therefore treated as a fixed 40-bin texture segment rather than a variable-length frame window from corridor entry.

ii. 
```python
ntrials = int(beh['ntrials'])
```
```python
VRmove = beh['ft_move'][:n_fr] > 0
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]
```
```python
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials,
                                   n_bins=N_BINS)
```
```python
for t in range(ntrials):
    neural_t = texture_spk[:, t, :]  # (n_neurons, 40)
```

iii. The trajectory shows the AI consciously moved away from variable-length frame trials because durations varied a lot. It argued that position interpolation gave a uniform 40-bin per-trial representation and treated position bins as an effective time axis.

## 1-e. How are trials filtered based on quality controls?

i. The AI skips sessions with missing behavior, fewer than two trials, or fewer than two running frames after masking. Within retained sessions it drops trials whose interpolated neural activity contains any `NaN` or `Inf`. It does not apply the reference trial-length outlier filter.

ii. 
```python
if beh is None:
    print(" -> SKIP: behavior not found")
    continue

ntrials = int(beh['ntrials'])
if ntrials < 2:
    print(f" -> SKIP: {ntrials} trials")
    continue
```
```python
if len(pos_cum_running) < 2:
    print(" -> SKIP: insufficient running frames")
    del spk_running
    continue
```
```python
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
```

iii. The trajectory justification focused on numerical validity after interpolation and on making the decoder workable, not on matching the reference trial-length QC. The docstring also states: “Trials with NaN/Inf in neural data after interpolation are excluded.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` primarily from `spks` in the session spike file, but also uses `iarea` from retinotopy for filtering, `ft_move` to keep only running frames, and `ft_PosCum` to define the interpolation coordinate.

ii. 
```python
def load_spk(mname, datexp, blk):
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    d = np.load(fn, allow_pickle=True).item()
    return np.concatenate(d['spks'], axis=0)
```
```python
def load_iarea(mname, datexp):
    fn = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    return np.load(fn, allow_pickle=True)['iarea']
```
```python
VRmove = beh['ft_move'][:n_fr] > 0
pos_cum = beh['ft_PosCum'][:n_fr]
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]
```

iii. The trajectory says the AI wanted to follow the paper’s running-only, position-interpolated analysis path, which is why it pulled in `ft_move` and cumulative position instead of using only the framewise spike matrix.

## 2-b. How is the `neural` data processed?

i. The spike planes are concatenated, filtered to visual-cortex neurons, masked to running frames only, and then interpolated from cumulative position onto a fixed 40-bin position grid covering the 4 m texture region. The result is stored as float32, one fixed-length array per trial.

ii. 
```python
spk = load_spk(mname, datexp, blk)
vc_mask = (iarea != -1) & (iarea != 7)
brain_region = get_brain_region_idx(iarea[vc_mask])
spk = spk[vc_mask]
```
```python
VRmove = beh['ft_move'][:n_fr] > 0
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]
```
```python
def position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS):
    bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
    trial_starts = np.arange(ntrials, dtype=np.float64)
    target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
    source = pos_cum_running / CORRIDOR_LENGTH
    for i in range(n_neurons):
        result[i] = np.interp(target, source, spk_running[i])
    return result.reshape(n_neurons, ntrials, n_bins)
```

iii. The AI repeatedly justified this choice in the trajectory by citing the paper’s running-only analyses and by arguing that position bins solved variable trial durations while still acting like a uniform time axis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy label. The AI excludes `iarea == -1` and `iarea == 7`, then maps the remaining neurons into four region groups.

ii. 
```python
vc_mask = (iarea != -1) & (iarea != 7)  # visual cortex mask
brain_region = get_brain_region_idx(iarea[vc_mask])
spk = spk[vc_mask]
```
```python
region[iarea == 8] = 0
for a in [0, 1, 2, 9]:
    region[iarea == a] = 1
for a in [5, 6]:
    region[iarea == a] = 2
for a in [3, 4]:
    region[iarea == a] = 3
```

iii. The trajectory says this was chosen to match the paper’s convention for “visual cortex” neurons and the `neu_area_ID`/`Get_density_map` logic in the source code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns neural data to the start of the texture corridor in position space, not to a framewise trial-start event on the native imaging grid. Every trial is represented by the same 40 interpolated position bins from 0 to 4 m.

ii. 
```python
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials,
                                   n_bins=N_BINS)
```
```python
'temporal_alignment_event': 'corridor entry (start of texture area)',
'off_start': 0.0,
'off_end': N_BINS * TIME_PER_BIN,
```

iii. In the trajectory the AI explicitly argued that because VR advances at constant speed while running, position bins could stand in for time bins and provide a uniform alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI treats each 1 dm position bin as `1/6` second, so the stored bin size is about `166.67 ms`. Yes, it rebins/resamples the native imaging data by position interpolation.

ii. 
```python
VR_SPEED_DM_S = 6.0
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S
TIME_PER_BIN_MS = TIME_PER_BIN * 1000
```
```python
'time_bin_size': TIME_PER_BIN_MS,
```
```python
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials,
                                   n_bins=N_BINS)
```

iii. The trajectory justification was that fixed 0.1 m bins were equivalent to fixed time bins during running and avoided variable trial lengths.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundPos`, not from frame timestamps or `SoundFr`.

ii. 
```python
sound_pos = beh['SoundPos'][t]  # in decimeters
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The trajectory states that sound cue position was in decimeters and that, under the AI’s position-bin interpretation, cue position could be converted into time-to-cue by multiplying the positional offset by the fixed seconds-per-bin value.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each of the 40 position bins, the AI subtracts the bin index from `SoundPos` and multiplies by `TIME_PER_BIN`, yielding a continuous 40-sample vector that is positive before the cue and negative after it.

ii. 
```python
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The trajectory justification was that position bins were being treated as an effective time axis, so cue distance in bins became cue time in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by sharing the same 40 position bins as the interpolated neural data.

ii. 
```python
neural_t = texture_spk[:, t, :]  # (n_neurons, 40)
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. The AI’s trajectory repeatedly framed all inputs and outputs as living on the same 40-bin position grid as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp`, parsed as a calendar date, and grouped by mouse `mname`.

ii. 
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_dates.setdefault(mname, []).append(date)
```
```python
session_date = datetime.strptime(datexp, '%Y_%m_%d')
day_of_training = float((session_date - mouse_first_date[mname]).days)
```

iii. The trajectory shows the AI debated using `sess#` or a session count, then chose “days from first recording for each mouse” as the simplest continuous measure.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI finds the earliest recording date for each mouse, subtracts it from each session date to get elapsed calendar days, converts that to float, and broadcasts the result across all 40 bins of each trial.

ii. 
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
```
```python
day_of_training = float((session_date - mouse_first_date[mname]).days)
day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
```

iii. The trajectory explicitly says the AI selected elapsed days from first recording rather than a simpler session-order count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from `StartFr` or frame timestamps. It is derived from the synthetic 40-bin position grid and the assumed constant seconds per bin.

ii. 
```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. The trajectory justification was the same position-is-time assumption used throughout the AI’s pipeline.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI precomputes a 40-element vector `[0, 1, ..., 39] * TIME_PER_BIN` and uses that same vector for every trial.

ii. 
```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```
```python
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. The trajectory says the AI wanted a uniform time axis tied to position bins rather than variable native frame times.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by sharing the same fixed 40 position bins as the neural data.

ii. 
```python
neural_t = texture_spk[:, t, :]
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. The trajectory rationale was to put all streams on one common 40-bin grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`.

ii. 
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The AI did not give a special justification beyond using the trial-level rewarded-corridor flag required by the task.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial boolean/numeric reward flag into float and broadcasts it across the 40 bins of the trial.

ii. 
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The trajectory treated this as a static per-trial input that needed to be repeated so all decoder inputs would share shape `(4, 40)`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
for wn in beh['UniqWalls']:
    all_stimuli.add(str(wn))
```
```python
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
```

iii. The trajectory says the AI intentionally built a global mapping over all unique `WallName` strings seen across sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps the fine-grained stimulus names as separate categories rather than collapsing them to four base textures. It sorts the global set of `WallName` values, maps each exact name to an integer, and repeats that integer across the 40 bins of the trial.

ii. 
```python
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
```
```python
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
```
```python
output_values = [
    stim_list,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
]
```

iii. The trajectory confirms this choice: the AI later described the decoder as operating on “15 classes” for visual stimulus.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickPos` and `LickTrind`, not from `LickFr`.

ii. 
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
```
```python
trial_mask = lick_trind == t
trial_lick_pos = lick_pos[trial_mask]
```

iii. The trajectory says the AI chose a position-binned representation and therefore used lick positions within each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial the AI takes lick positions in `[0, 40)`, floors them to integer bins, clips to `[0, 39]`, and marks those position bins as 1 in a binary 40-bin vector.

ii. 
```python
lick_binary = np.zeros(N_BINS, dtype=np.int32)
trial_mask = lick_trind == t
trial_lick_pos = lick_pos[trial_mask]
valid_licks = trial_lick_pos[(trial_lick_pos >= 0) & (trial_lick_pos < N_BINS)]
if len(valid_licks) > 0:
    lick_bins = np.clip(np.floor(valid_licks).astype(int), 0, N_BINS - 1)
    lick_binary[lick_bins] = 1
```

iii. The trajectory explicitly mentions “Licking binarized per position bin using LickPos and LickTrind.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned on the same 40 position bins as the interpolated neural data.

ii. 
```python
neural_t = texture_spk[:, t, :]
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)
```

iii. The AI’s general alignment justification in the trajectory was that all streams should live on the common position-bin grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is not derived from `ft_Pos` or another measured per-frame position variable. It is derived from the fixed 40-bin trial representation itself.

ii. 
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
```
```python
pos_arr = position_bins.copy()
```

iii. The trajectory acknowledges that under position interpolation the position output becomes deterministic from the bin index, and later notes that position bins are perfectly balanced because of this construction.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI precomputes a constant 40-sample vector of categories `[0]*10 + [1]*10 + [2]*10 + [3]*10` and reuses it for every trial.

ii. 
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
pos_arr = position_bins.copy()
```

iii. The trajectory justification was that the trial representation already lives in 0.1 m position bins, so 1 m categories can be assigned directly by bin index.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 40 bins are divided into four equal 10-bin chunks, corresponding to 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. 
```python
POSITION_N_CATS = 4
N_BINS = 40
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
```
```python
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The AI treated this as a direct consequence of using 40 equally spaced 0.1 m bins over a 4 m texture region.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is perfectly aligned by construction because the neural data itself was interpolated to those same 40 position bins.

ii. 
```python
neural_t = texture_spk[:, t, :]
pos_arr = position_bins.copy()
```

iii. The trajectory says position interpolation was selected specifically to place neural activity and behavioral outputs on the same grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `run_pos`, the precomputed speed-by-position array, not from framewise `ft_RunSpeed`.

ii. 
```python
speed = beh['run_pos'][:, :N_BINS]
valid = ~np.isnan(speed)
all_speed_values.append(speed[valid].ravel())
```
```python
speed = beh['run_pos'][t, :N_BINS].copy()
```

iii. The trajectory explicitly says the AI chose `run_pos` because it already matched the position-binned representation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. In a first pass the AI pools all non-NaN `run_pos` values from the first 40 bins of all sessions and computes global quartile thresholds. Per trial it takes the first 40 speed bins, fills `NaN` with zero, and digitizes each bin against those global thresholds.

ii. 
```python
speed = beh['run_pos'][:, :N_BINS]
valid = ~np.isnan(speed)
all_speed_values.append(speed[valid].ravel())
```
```python
all_speed_arr = np.concatenate(all_speed_values)
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
```
```python
speed = beh['run_pos'][t, :N_BINS].copy()
speed[np.isnan(speed)] = 0.0
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The trajectory says the AI wanted quartiles “across all sessions’ texture area” and believed a few outliers or NaNs would not matter much.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by three global percentile cut points at the 25th, 50th, and 75th percentiles of pooled `run_pos` values, then binned with `np.digitize`.

ii. 
```python
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
```
```python
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The trajectory explicitly records the decision “Running speed discretized into quartiles computed across all sessions’ texture area.”

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned on the same 40 position bins as the neural data because both use the texture-area position grid.

ii. 
```python
neural_t = texture_spk[:, t, :]
speed = beh['run_pos'][t, :N_BINS].copy()
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The trajectory frames `run_pos` as already being in the appropriate position-binned domain for the interpolated neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness by skipping sessions with missing behavior, skipping sessions with too few trials or too few running frames, skipping trials with interpolated `NaN`/`Inf` neural values, and replacing `NaN` running-speed bins with zero before discretization.

ii. 
```python
if beh is None:
    print(" -> SKIP: behavior not found")
    continue
```
```python
if len(pos_cum_running) < 2:
    print(" -> SKIP: insufficient running frames")
    del spk_running
    continue
```
```python
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
```
```python
speed = beh['run_pos'][t, :N_BINS].copy()
speed[np.isnan(speed)] = 0.0
```

iii. The trajectory justification focused on practical robustness for a very large conversion rather than on reproducing the reference’s frame-clipping behavior.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive operations are likely loading the large spike matrices for every session and the per-neuron interpolation of running-frame spikes onto 40 position bins for every trial.

ii. 
```python
spk = load_spk(mname, datexp, blk)
```
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```
```python
for sess_i, key in enumerate(session_keys):
    ...
    texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials,
                                       n_bins=N_BINS)
```

iii. The trajectory repeatedly discussed the cost of interpolating ~50k-70k neurons per session and the overall 273 GB output size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the explicit neuron loop inside `position_interpolate`. The per-trial loop that builds `stim_arr`, `lick_binary`, `pos_arr`, and `speed_cat` is another hotspot.

ii. 
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```
```python
for t in range(ntrials):
    ...
    lick_binary = np.zeros(N_BINS, dtype=np.int32)
    ...
    speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The trajectory does not spell this out explicitly, but it does acknowledge that the interpolation step dominates work and is being run at very large scale.

## 12-c. What processing does the code repeat multiple times?

i. The code makes a first pass over every session to gather `all_stimuli`, global speed values, and first recording dates, then a second pass to do the actual conversion. It also rereads behavior dictionaries via `beh_cache` in both passes and rebuilds per-trial constant arrays repeatedly.

ii. 
```python
# ---- First pass: collect global statistics ----
for key in session_keys:
    ...
    for wn in beh['UniqWalls']:
        all_stimuli.add(str(wn))
    speed = beh['run_pos'][:, :N_BINS]
    ...
```
```python
# ---- Second pass: process all sessions ----
for sess_i, key in enumerate(session_keys):
    ...
```
```python
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The trajectory explicitly describes the code as a two-pass process and treats that as a deliberate design to obtain global quartiles and stimulus mappings before conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does expensive full-neuron position interpolation even though the downstream decoder later reduces neural dimensionality heavily. It also constructs a deterministic per-trial `position` output from bin index alone rather than from measured framewise position, and it computes a 15-class exact-stimulus label set even though the task description only asked for broad stimulus category examples.

ii. 
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
pos_arr = position_bins.copy()
```
```python
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
```

iii. The trajectory does not explicitly call these unnecessary, but it does note that the decoder later projects to much lower dimensionality and that position becomes perfectly balanced because of uniform position binning.
