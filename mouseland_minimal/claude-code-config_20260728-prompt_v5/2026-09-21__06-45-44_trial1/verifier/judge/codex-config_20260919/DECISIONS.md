# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `beh/Imaging_Exp_info.npy` as a master index, deduplicates recordings by mouse/date/block, caches each experiment-type behavior file, and loads the corresponding spike and retinotopy files per session. It makes two passes: one for global stimulus/speed/date statistics and one for conversion.

ii. ```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
unique_sessions = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if key not in unique_sessions:
            unique_sessions[key] = (exp_type, ndb)
```

iii. The trajectory says the agent inspected the index and data layout, found 89 unique sessions with spike and retinotopy data, and chose to include all experiment types. It used caching to avoid repeatedly loading behavior files.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted mouse names collected from sessions with readable behavior during the first pass. Each retained session receives the corresponding index.

ii. ```python
mouse_dates.setdefault(mname, []).append(date)
subjects = sorted(mouse_first_date.keys())
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data_subject_idx.append(subject_to_idx[mname])
```

iii. The agent recognized `mname` as the explicit subject identifier and reported 19 subjects.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Duplicate appearances under experiment types are resolved by retaining the first occurrence.

ii. ```python
key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if key not in unique_sessions:
    unique_sessions[key] = (exp_type, ndb)
```

iii. The trajectory explicitly identified this tuple as a recording session and verified that it gives 89 unique sessions.

## 1-d. How are the data split into trials?

i. The AI uses `beh['ntrials']` and slices the position-interpolated tensor on its trial axis. Every trial is represented by 40 fixed 0.1-m bins covering the 4-m texture region, rather than native frame-defined trial windows.

ii. ```python
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
for t in range(ntrials):
    neural_t = texture_spk[:, t, :]
```

iii. The agent reasoned that position interpolation gives uniform trial lengths and argued that constant VR speed makes 0.1-m bins equivalent to 1/6-second bins.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than two declared trials are skipped; individual trials are skipped only if interpolated neural data contains NaN or Inf. A session is skipped if fewer than two valid trials remain. There is no long-trial filter.

ii. ```python
if ntrials < 2:
    continue
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
if len(session_neural) < 2:
    continue
```

iii. The script header calls NaN/Inf exclusion a processing decision. The trajectory focused on producing fixed position-binned trials and did not justify omission of the reference's extreme-duration control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from per-plane `spks` arrays in each session's neural file. `ft_move` and `ft_PosCum` select running samples and provide interpolation coordinates; `iarea` supplies neuron area assignments.

ii. ```python
d = np.load(fn, allow_pickle=True).item()
return np.concatenate(d['spks'], axis=0)
VRmove = beh['ft_move'][:n_fr] > 0
pos_cum = beh['ft_PosCum'][:n_fr]
```

iii. The agent inspected the paper code and chose deconvolved Suite2p spikes and the paper's position-interpolation variables.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, neurons filtered, nonmoving frames removed, and each neuron's activity linearly interpolated against cumulative position onto 40 spatial targets per trial. Results are float32.

ii. ```python
source = pos_cum_running / CORRIDOR_LENGTH
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
return result.reshape(n_neurons, ntrials, n_bins)
```

iii. The agent believed `get_interpPos_spk` was the paper's standard processing and preferred it to variable-duration native-frame trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes neurons with `iarea == -1` or `iarea == 7`, maps the remaining visual-area codes to four regions, and removes nonmoving frames before interpolation.

ii. ```python
vc_mask = (iarea != -1) & (iarea != 7)
brain_region = get_brain_region_idx(iarea[vc_mask])
spk = spk[vc_mask]
spk_running = spk[:, :n_fr][:, VRmove]
```

iii. The agent cited the paper convention `idx_neu = (arid!=-1) & (arid != 7)` and the statement that analyses considered running timepoints.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is placed on a fixed spatial grid beginning at corridor entry (position zero), not on the native temporal grid. Bin index is treated as elapsed time under an assumed constant 0.6 m/s VR speed.

ii. ```python
bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
```

iii. The trajectory acknowledges the task requests temporal alignment, but concludes that position interpolation is effectively temporal because VR moves at constant speed while running.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI reports 166.67 ms per bin, derived from 0.1 m divided by 0.6 m/s. It replaces native 3.17-Hz imaging frames with 40 position-interpolated bins.

ii. ```python
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S
TIME_PER_BIN_MS = TIME_PER_BIN * 1000
'time_bin_size': TIME_PER_BIN_MS,
```

iii. The agent justified the value by interpreting every decimeter of VR motion as 1/6 second.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundPos`, the spatial-bin indices, and the assumed constant VR speed.

ii. ```python
sound_pos = beh['SoundPos'][t]
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The agent chose the spatial cue position because all converted data had been position-interpolated.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The current bin position is subtracted from cue position and the distance is divided by 6 dm/s, producing positive values before and negative values after the cue.

ii. ```python
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The trajectory's justification is that position bins correspond to fixed time increments under constant VR motion.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has the same 40 position-bin indices as the interpolated neural matrix.

ii. ```python
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. The agent considered shared position-bin indices sufficient alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp` parsed as a calendar date and the earliest recording date observed for that mouse.

ii. ```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
```

iii. The agent inferred training day from session dates because no explicit training-day field was used.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It computes elapsed calendar days since the mouse's first recording and repeats that scalar over 40 bins.

ii. ```python
day_of_training = float((session_date - mouse_first_date[mname]).days)
day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
```

iii. The script documents this as “days from first recording for each mouse.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from trial timestamps or `StartFr`; it is constructed from spatial-bin index and assumed VR speed.

ii. ```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. The agent reasoned that constant-speed position bins can stand in for elapsed time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Bin indices 0–39 are multiplied by 1/6 second, yielding 0 through 6.5 seconds.

ii. ```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. The script states that a 1-dm bin equals 1/6 second.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The same static 40-element array is used for every position-interpolated trial.

ii. ```python
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. The agent treated corridor entry as bin zero and each following position bin as a fixed temporal offset.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `beh['isRew']` flag.

ii. ```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The agent identified `isRew` as the direct rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is cast to float and broadcast across all 40 bins of the trial.

ii. ```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. No transformation beyond broadcasting was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Categories are derived from `UniqWalls` to form a global vocabulary and from per-trial `WallName` to choose the label.

ii. ```python
for wn in beh['UniqWalls']:
    all_stimuli.add(str(wn))
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
```

iii. The agent chose all 15 wall variants as separate categories, as reflected in its reported 15-class decoder result.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Sorted unique wall names receive integer indices, and a trial's index is repeated across 40 bins. Crop/swap variants are not collapsed to four base textures.

ii. ```python
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
```

iii. The trajectory treats the 15 observed names as the requested stimulus categories and validates the resulting 15-way decoder.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses lick positions `LickPos` and trial indices `LickTrind`.

ii. ```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
trial_mask = lick_trind == t
```

iii. The agent selected position-domain lick variables to match its spatially interpolated trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Valid lick positions in [0,40) are floored to integer spatial bins; any bin containing at least one lick is set to one.

ii. ```python
valid_licks = trial_lick_pos[(trial_lick_pos >= 0) & (trial_lick_pos < N_BINS)]
lick_bins = np.clip(np.floor(valid_licks).astype(int), 0, N_BINS - 1)
lick_binary[lick_bins] = 1
```

iii. The agent intended a binary lick label per converted position bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the same 40 spatial bins as position-interpolated neural activity, not matched by imaging frame.

ii. ```python
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)
```

iii. The agent viewed position-domain binning as the appropriate alignment after neural interpolation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is implicit in the fixed converted-bin index rather than read per trial from `ft_Pos`.

ii. ```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS)
```

iii. Since every trial was interpolated to uniform corridor positions, the agent considered position deterministic.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A static vector of four labels, each repeated for ten 0.1-m bins, is copied for every trial.

ii. ```python
pos_arr = position_bins.copy()
```

iii. This directly implements four equal 1-m segments over the 4-m texture corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Converted indices 0–9, 10–19, 20–29, and 30–39 map to categories 0–3.

ii. ```python
position_bins = np.repeat(np.arange(4), 40 // 4).astype(np.int32)
```

iii. The agent used the four equal-length categories explicitly requested.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position labels are defined by the same target grid used to interpolate neural activity.

ii. ```python
neural_t = texture_spk[:, t, :]
pos_arr = position_bins.copy()
```

iii. The common position-bin index provides exact spatial alignment by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, the trial-by-position speed matrix.

ii. ```python
speed = beh['run_pos'][:, :N_BINS]
```

iii. The agent chose the already position-binned speed stream to match its neural grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global 25th, 50th, and 75th percentile thresholds are computed from all non-NaN values in all sessions. Per-trial NaNs are replaced with zero before categorization.

ii. ```python
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
speed[np.isnan(speed)] = 0.0
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The script states that speed is discretized into quartiles across all sessions to meet the four-25%-bins requirement.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` applies the three global percentile thresholds, returning categories 0–3. Ties are not rank-split.

ii. ```python
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The agent reported nearly balanced global quartiles and considered that successful discretization.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The first 40 columns of each `run_pos` trial are assumed to correspond directly to the neural interpolation's 40 spatial targets.

ii. ```python
speed = beh['run_pos'][t, :N_BINS].copy()
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)
```

iii. Both streams are in the position domain, which the agent used as its alignment basis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior keys, sessions with too few trials/running frames, trials with nonfinite interpolated neural data, and sessions with fewer than two surviving trials are skipped. Spike/behavior lengths are truncated to their minimum. Speed NaNs are changed to zero. Other load errors are not caught.

ii. ```python
beh = beh_cache[exp_type].get(beh_key)
if beh is None: continue
n_fr = min(n_frames_spk, len(beh['ft_move']))
if len(pos_cum_running) < 2: continue
speed[np.isnan(speed)] = 0.0
```

iii. The agent added defensive checks discovered during inspection, including concern about irregular frame counts and invalid interpolation output; it did not document evidence for zero-imputing speed.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating enormous spike files, per-neuron interpolation in Python, retaining the full 273.76-GB converted dataset, and garbage collection are the major conversion costs. The subsequent decoder training was also extremely costly, though outside conversion.

ii. ```python
return np.concatenate(d['spks'], axis=0)
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
gc.collect()
```

iii. The trajectory repeatedly highlights the scale (roughly 46k neurons/session and a 273.76-GB output) and the long end-to-end runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit loop calling `np.interp` once per neuron is the clearest vectorization/compiled-batching candidate. Session and trial loops reflect heterogeneous files/records and are less directly vectorizable.

ii. ```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```

iii. The agent did not explicitly discuss vectorizing this loop; it instead focused on memory and dataset scale.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded in both passes; every session's speed data is visited once for global thresholds and again for conversion; session metadata and behavior keys are reconstructed in both passes. Behavior-file caching prevents repetition within a pass.

ii. ```python
beh_cache = {}
for key in session_keys:  # first pass
    ...
beh_cache = {}
for sess_i, key in enumerate(session_keys):  # second pass
    ...
```

iii. The agent deliberately adopted a two-pass design to obtain global stimulus and speed statistics before encoding outputs.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It gathers mouse dates and unique stimuli from every session in a separate pass, builds a full 60-bin conceptual target although only 40 bins are retained, and repeatedly forces garbage collection. More importantly, position interpolation and running-frame filtering discard native timing/stopping information required by the requested temporal variables.

ii. ```python
all_stimuli = set()
all_speed_values = []
mouse_dates = {}
gc.collect()
```

iii. The agent regarded these operations as necessary for its global encoding and memory management; it did not identify discarded work itself.
