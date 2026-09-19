# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master index from `data/beh/Imaging_Exp_info.npy`, iterates experiment types in a custom priority order, reads each behavior file `Beh_<exp_type>.npy`, stores each unique `(mname, datexp, blk)` recording once, and later loads the matching spike file and retinotopy file for each session.

ii. 
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
```
```python
beh_path = os.path.join(root, 'beh', f'Beh_{exp_type}.npy')
beh_data = np.load(beh_path, allow_pickle=True).item()
```
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. In the trajectory, the agent said each unique recording should be included once, behavior should come from one experiment-type file for that recording, and spikes plus retinotopy should be loaded per recording (steps 34, 37, 40).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by mouse name `mname`. The final `subjects` list is the sorted unique set of `mname` values, and each kept session gets a `subject_idx` from that list.

ii. 
```python
all_subjects = sorted(set(s['mname'] for s in sessions))
```
```python
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The trajectory consistently treated each unique `(mname, datexp, blk)` as a session nested under one mouse, so `mname` was the subject identifier (steps 37, 40).

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. If the same recording appears under multiple experiment types, the first one encountered in the AI's custom `exp_order` is kept and later duplicates are skipped.

ii. 
```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

iii. The agent explicitly decided not to duplicate neural data across experiment types and to use each unique recording exactly once (steps 34, 37, 40).

## 1-d. How are the data split into trials?

i. The AI uses `beh['ntrials']` as the trial count and represents every trial as 40 fixed position bins from the texture corridor after whole-session position interpolation. It does not define per-trial frame windows from `ft_trInd` and `ft_CorrSpc`.

ii. 
```python
n_trials = beh['ntrials']
interp_spk = position_interpolate_spk(
    valid_spk,
    ft_pos_cum[vr_moving],
    corridor_len,
    n_trials,
    n_bins=N_POS_BINS
)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```
```python
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. In the trajectory, the agent reasoned that the paper used position-interpolated trials with 60 corridor bins and therefore chose 40 texture bins per trial for the decoder as well (steps 25, 37, 40).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter individual trials for quality. It keeps all `ntrials` from the behavior object and only skips an entire session if it ends up with fewer than two trials.

ii. 
```python
n_trials = beh['ntrials']
for tr in range(n_trials):
    ...
```
```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. The trajectory does not show a separate trial-QC policy; the agent focused on fixed-bin position interpolation and minimum-session-size checks instead (steps 40, 71).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from deconvolved traces in `spk_data['spks']`, concatenated across imaging planes, and from retinotopy labels `iarea` used to select neurons and assign regions.

ii. 
```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
iarea = ret_data['iarea']
```

iii. The trajectory states that neural data comes from Suite2p deconvolved traces and that `iarea` provides the brain-region assignments used for filtering and labeling (steps 25, 40).

## 2-b. How is the `neural` data processed?

i. The AI filters to running frames (`ft_move > 0`), position-interpolates those frames across the whole session into 60 bins per trial using cumulative position `ft_PosCum` and corridor length, keeps only the first 40 texture bins, and stores each trial as `float32`.

ii. 
```python
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
ft_pos_cum = beh['ft_PosCum'][:n_frames]
```
```python
interp_spk = position_interpolate_spk(
    valid_spk,
    ft_pos_cum[vr_moving],
    corridor_len,
    n_trials,
    n_bins=N_POS_BINS
)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The agent repeatedly justified this as matching the paper's position-interpolated analysis pipeline and later optimized the interpolation loop for speed because it became the bottleneck (steps 25, 37, 40, 71).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps neurons with `iarea != -1` and `iarea != 7`, then assigns region labels for V1, mHV, lHV, and aHV. Those kept neurons are the ones used in the output.

ii. 
```python
valid_mask = (iarea != -1) & (iarea != 7)
region_idx = np.full(len(iarea), -1, dtype=int)
region_idx[iarea == 8] = 0
region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
region_idx[(iarea == 5) | (iarea == 6)] = 2
region_idx[(iarea == 3) | (iarea == 4)] = 3
```
```python
valid_spk = spk[valid_neurons][:, vr_moving]
```

iii. The trajectory says the agent intended to include only visual cortex neurons, using the `iarea` conventions from the paper code and excluding `-1` and `7` (steps 25, 40).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says the data are aligned to trial start, but the actual alignment is to standardized position bins within the 4 m texture corridor: every trial is represented by the same 40 post-interpolation spatial bins rather than by the original imaging frames after corridor entry.

ii. 
```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
norm_pos = accum_pos / corridor_len
...
return interp_spk.reshape(n_neurons, n_trials, n_bins)
```
```python
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. In the trajectory, the agent explicitly chose position-interpolated corridor bins for alignment, even while acknowledging the downstream decoder prompt mentioned trial-start alignment (steps 25, 37, 40).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI sets the bin size to `1 / 6` seconds per bin, i.e. about 166.67 ms, by assuming 1 dm bins traversed at 6 dm/s. It therefore treats the data as temporally rebinned into fixed spatial bins.

ii. 
```python
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
BIN_SIZE_MS = BIN_SIZE_SEC * 1000
```
```python
'time_bin_size': BIN_SIZE_MS,
```

iii. The trajectory explains this choice as converting 1 dm position bins into time using the stated VR speed, rather than keeping the native 3.17 Hz imaging frame bins (steps 25, 37, 40).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives `time_to_sound_cue` from `SoundPos`, the cue position in the corridor, together with the synthetic position-bin index `i` for each of the 40 texture bins.

ii. 
```python
sound_pos = beh['SoundPos']
```
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. In the trajectory, the agent said the cue input should be based on the animal's position relative to the sound location, converted to seconds using the assumed corridor speed (steps 37, 40).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI subtracts the trial's cue position from each integer position bin index and multiplies by the fixed `BIN_SIZE_SEC`. This yields a deterministic 40-sample vector per trial based on spatial distance to the cue.

ii. 
```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The trajectory justification was that bin-to-cue distance could be converted to seconds by assuming approximately 0.167 s per decimeter bin (steps 37, 40).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same 40 position bins per trial that were used for the position-interpolated neural data.

ii. 
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                        axis=0)
```

iii. The trajectory treated all decoder variables as sharing the same position-bin grid once the neural data had been interpolated that way (steps 37, 40).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives day of training from each session's mouse name `mname` and date string `datexp`.

ii. 
```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
```
```python
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
```

iii. The trajectory described this feature as the training day for each mouse and reasoned about it in terms of recording dates rather than session order (step 40).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI parses each session date, finds the earliest recording date for each mouse, computes calendar days elapsed since that first date, and then broadcasts that scalar over all 40 bins of every trial in the session.

ii. 
```python
days = (d - mouse_first_date[s['mname']]).days
session_days[key] = float(days)
```
```python
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The trajectory justification was implicit: the agent framed this input as "days since first recording for each mouse" rather than as session count (step 40 and the continuation summary at step 111).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI does not derive this input from `StartFr` or timestamps. It derives it from the synthetic per-trial position-bin index `i` and the fixed `BIN_SIZE_SEC`.

ii. 
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The trajectory treated each position bin as a fixed-duration step from trial start, so elapsed time was computed from bin number rather than frame timestamps (steps 37, 40).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It creates the vector `[0, 1, 2, ..., 39] * BIN_SIZE_SEC` for every trial, so every trial gets the same monotonically increasing elapsed-time template.

ii. 
```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The trajectory justification was the same fixed-speed corridor assumption used for other time variables (steps 37, 40).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction to the same 40 interpolated position bins as the neural data.

ii. 
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_avail],
                        axis=0)
```

iii. The trajectory assumed all trial-varying decoder variables should share the common 40-bin corridor representation after interpolation (steps 37, 40).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `reward_availability` is derived from `beh['isRew']`.

ii. 
```python
is_rew = beh['isRew'].astype(float)
```

iii. The trajectory treated reward availability as the per-trial rewarded-corridor flag from behavior (step 40).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI casts the reward flag to float and broadcasts the trial's scalar reward value across all 40 bins in that trial.

ii. 
```python
is_rew = beh['isRew'].astype(float)
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. No further transformation was separately justified in the trajectory beyond making all decoder variables live on the same per-trial 40-bin grid (step 40).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The visual-stimulus output is derived from `beh['WallName']`, and the AI also collects `UniqWalls` across sessions to build a global category list.

ii. 
```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```
```python
for wn in s['beh']['UniqWalls']:
    all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
```

iii. The trajectory says the agent needed a consistent set of stimulus categories across sessions and viewed the wall texture names as the relevant source labels (step 40).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses every unique wall-name string as its own class, sorts those names globally, converts each trial's `WallName` to that global index, and broadcasts the resulting category across all 40 bins of the trial.

ii. 
```python
all_stim_names = sorted(all_stim_set)
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```
```python
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. The trajectory only states that the agent wanted "a consistent set of stimulus categories"; it did not describe mapping variants back to four base texture families, so the final code's choice of all unique names appears to be the operative decision (step 40).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The AI derives licking from `LickPos` and `LickTrind`, not from imaging-frame lick times.

ii. 
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
```

iii. The trajectory did not separately defend this variable choice, but it follows directly from the agent's decision to express all time-varying outputs on position bins rather than frame times.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI creates a `(n_trials, 40)` zero matrix, floors each lick position to an integer bin, uses `LickTrind` to choose the trial, and sets that position bin to `1`.

ii. 
```python
lick_binary = np.zeros((n_trials, n_bins), dtype=int)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

iii. The trajectory's broader justification was that decoder variables should live on the same position-bin grid as the interpolated neural data (steps 37, 40).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by putting lick events into the same 40 position bins used for the neural trial arrays.

ii. 
```python
lick_tr = lick_binary[tr, :].astype(np.int64)
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The trajectory did not describe a separate lick-alignment rule beyond the shared position-bin representation chosen for the converted dataset (steps 37, 40).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI's position output is not taken directly from a per-frame raw position variable. It is derived from the fixed 40-bin corridor template implied by `N_TEXTURE_BINS`, with the neural data itself created from `ft_PosCum` and `Corridor_Length`.

ii. 
```python
N_TEXTURE_BINS = 40
```
```python
ft_pos_cum = beh['ft_PosCum'][:n_frames]
...
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The trajectory shows the agent decided to represent each trial as standardized corridor position bins, so the decoded position categories were taken from that template rather than from the original framewise `ft_Pos` signal (steps 25, 37, 40).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI builds a fixed array of 40 texture-bin indices and groups every 10 adjacent bins into one of four 1 m categories.

ii. 
```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The trajectory justification was implicit in the position-bin design: 40 decimeter bins across the 4 m textured corridor naturally collapse into four 1 m bins (step 40).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded deterministically by integer division: bins `0-9` map to category `0`, `10-19` to `1`, `20-29` to `2`, and `30-39` to `3`.

ii. 
```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. No separate trajectory discussion appears; this follows directly from the AI's 1 dm spatial-bin representation of the 4 m texture corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position output is aligned by using the same 40 position bins per trial as the neural data.

ii. 
```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
pos_tr = pos_bins.astype(np.int64)
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The trajectory treated position as one more signal living on the common position-bin grid produced by neural interpolation (steps 37, 40).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI derives running speed from `beh['run_pos']`, which it assumes is already position-interpolated speed by trial and corridor bin.

ii. 
```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]
```

iii. The trajectory does not separately justify this source variable, but the code reflects the agent's broader preference for already position-binned behavioral variables once it committed to a position-aligned dataset.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI concatenates `run_pos[:, :40]` from all sessions, computes global 25th, 50th, and 75th percentile edges, then bins each session's 40 position-bin speeds with `np.digitize`.

ii. 
```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The trajectory shows the agent intentionally computed speed quartiles over the full dataset and even noted a sample-run mismatch when those global edges were applied to only a few sessions (step 55 and step 111 summary).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI thresholds running speed with three global percentile edges and maps the result to four categories `0-3`, labeled `Q1` through `Q4`.

ii. 
```python
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```
```python
['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The trajectory justification was that the task asked for quartile bins, so the agent used percentile thresholds from the pooled dataset (steps 40, 55).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by using the same 40 position bins per trial as the neural data.

ii. 
```python
speed_tr = speed_bins[tr, :].astype(np.int64)
output_trial = np.stack([stim_cat, lick_tr, pos_tr, speed_tr], axis=0)
```

iii. The trajectory consistently assumed that position-binned behavioral variables should be aligned to the same interpolated neural bins (steps 37, 40).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavior files or behavior keys with warnings, bounds-checks lick trial/bin assignments, and catches session-level exceptions so processing can continue. It does not explicitly trim behavior arrays to imaged frames or add other mismatch handling.

ii. 
```python
if not os.path.exists(beh_path):
    print(f"  Warning: behavior file not found for {exp_type}")
    continue
...
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue
```
```python
if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
    lick_binary[tr, bin_idx] = 1
```
```python
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    ...
    continue
```

iii. The trajectory mostly frames robustness in operational terms, especially continuing past failures during long conversions, rather than in terms of precise data-cleaning logic (step 71 and the later continuation summary).

## 12-a. What are the most time-consuming steps of the code?

i. The AI's own reasoning identifies whole-session position interpolation of large spike matrices as the major bottleneck, especially the per-neuron interpolation loop.

ii. 
```python
interp_spk = np.zeros((n_neurons, n_target), dtype=np.float32)
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

iii. In the trajectory, the agent explicitly said the conversion was taking too long because of position interpolation over 30k-90k neurons per session and that this, not file I/O, was the practical bottleneck of its implementation (step 71).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized loops are the per-neuron interpolation loop, the per-lick loop in `lick_to_position_bins`, the per-bin loop constructing `pos_bins`, and the per-trial assembly loop.

ii. 
```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```
```python
for i in range(len(lick_pos)):
    ...
```
```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The trajectory only explicitly calls out the interpolation loop as a performance problem and describes replacing `scipy` interpolation with `numpy.interp` for speed (step 71).

## 12-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of processing: it parses dates twice per session when computing training days, computes placeholder `day_of_training` arrays and then overwrites them later, and repeatedly uses `all_stim_names.index(wn)` inside per-session processing.

ii. 
```python
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    ...
for s in sessions:
    d = compute_day_of_training(s['datexp'])
```
```python
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
...
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The trajectory does not discuss these smaller redundancies; the explicit optimization attention went to the interpolation path (step 71).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI interpolates neural data into all 60 corridor bins and then discards the last 20 gray-space bins, computes `session_date` but does not use it downstream, and stores `n_neurons` from `valid_mask.sum()` even though only the trial arrays and `brain_region_idx` matter for later decoding.

ii. 
```python
interp_spk = position_interpolate_spk(..., n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```
```python
session_date = compute_day_of_training(datexp)
...
'session_date': session_date,
```

iii. The trajectory justifies the 60-bin interpolation as following the paper before trimming to the 40-bin textured corridor; it does not separately acknowledge the extra discarded work (steps 25, 40, 71).
