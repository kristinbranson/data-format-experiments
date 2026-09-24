# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from three subfolders of `/app/data`: `beh` (behavior, one file per experiment type plus the master index `Imaging_Exp_info.npy`), `spk` (deconvolved traces, one file per recording) and `retinotopy` (the visual-area label of each neuron). `Imaging_Exp_info.npy` is read first and is the index of every recording, grouped by experiment type. Each entry gives `mname`, `datexp`, `blk` (and `stimtype` for swap sessions), from which the session id and the behavior key are built. The behavior files are loaded lazily into a dictionary `beh_cache` keyed by experiment type (and kept there for the rest of the pass), the spike file and the retinotopy file are loaded once per session. The whole dataset is traversed **twice**: a first pass over behavior only (to pool running speeds, collect the stimulus names and the recording dates), and a second pass that loads behavior + spikes + retinotopy and writes the converted trials.

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

beh_key = f'{mname}_{datexp}_{blk}'
if 'stimtype' in ndb:
    beh_key += f'_{ndb["stimtype"]}'
beh = beh_cache[exp_type].get(beh_key)
```
```python
def load_spk(mname, datexp, blk):
    """Load and concatenate spike data across planes, matching utils.load_spk."""
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    d = np.load(fn, allow_pickle=True).item()
    return np.concatenate(d['spks'], axis=0)

def load_iarea(mname, datexp):
    """Load area assignments from retinotopy data."""
    fn = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    return np.load(fn, allow_pickle=True)['iarea']
```

iii. The AI explicitly copied the loading pattern of the paper's own `data_process_script.ipynb` and `utils.load_spk`: "I'm mapping out the experiment type categories... I'll pick the first experiment type encountered for each unique `(mname, datexp, blk)` combination and load behavior data using that type consistently, using the stimtype-specific behavior key when available." It decided to include every experiment type ("My plan is to include all experiment types to maximize data"), after first considering restricting to the supervised cohort and rejecting that: "I lean toward including all sessions since it maximizes diversity for decoding other variables like stimulus, position, and speed, and the reward input just ends up constant for those sessions." The two-pass design was chosen because the speed quartiles and the global stimulus vocabulary have to be known before any trial can be written.

## 1-b. How are the data split into subjects (mice)?

i. The mouse is the `mname` field of the index entry, i.e. the first component of the session key `(mname, datexp, blk)`. The unique mouse names are sorted and become `subjects`; `subject_to_idx` maps a name to its position and `subject_idx` records one index per written session. The result is 19 subjects over 89 sessions.

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
subjects = sorted(mouse_first_date.keys())
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```
```python
data_subject_idx.append(subject_to_idx[mname])
```
```python
'subjects': subjects,
'subject_idx': np.array(data_subject_idx, dtype=np.int64),
```

iii. No justification is given beyond the obvious: the index already names the mouse of every recording, so nothing has to be inferred. The AI confirmed the count in its log ("Subjects (19): ['DR10', 'DR15', ...]").

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse, one date, one block, keyed by the tuple `(mname, datexp, blk)`. Because the same recording is listed under several experiment types in `Imaging_Exp_info.npy`, only the **first** experiment type that lists it is kept; that experiment type also decides which `Beh_*.npy` file and which behavior key are used. This yields 89 unique sessions, all of which are written out.

ii.
```python
unique_sessions = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = (ndb['mname'], ndb['datexp'], ndb['blk'])
        if key not in unique_sessions:
            unique_sessions[key] = (exp_type, ndb)

session_keys = sorted(unique_sessions.keys())
print(f"Found {len(session_keys)} unique sessions")
```

iii. From the trajectory: "For sessions with stimtype variants, since the same neural data can be referenced by multiple behavior entries, I'll pick the first experiment type encountered for each unique `(mname, datexp, blk)` combination." The session id is also exactly the name of the spike file, so the tuple is the natural session unit. The AI double-checked "whether all 89 unique sessions are truly distinct, since some might be duplicated across different experiment types."

## 1-d. How are the data split into trials?

i. Trials are **not** cut out with the per-frame trial label `ft_trInd`. Instead the AI reproduces the paper's own preprocessing (`utils.spk_pos_interp` / `utils.get_interpPos_spk`): the imaging frames in which the virtual reality was moving (`ft_move > 0`) are placed on the cumulative-position axis `ft_PosCum`, and the activity is linearly interpolated onto a regular grid of `trial + k/60` (corridor length = 60 dm), so trial *t* is by construction the segment of cumulative position `[60t, 60t+60)`. Only the first 40 bins of each trial (the 4 m texture area, `Texture_Length = 40`) are computed and kept; the 2 m grey space is dropped. Every trial therefore has exactly 40 bins, and `beh['ntrials']` sets the number of trials.

ii.
```python
def position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS):
    # Target: first n_bins positions within each corridor
    # Matches the paper's linPos = np.arange(0, ntrials, 1/corridor_length)
    bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
    trial_starts = np.arange(ntrials, dtype=np.float64)
    target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
    source = pos_cum_running / CORRIDOR_LENGTH
    ...
    for i in range(n_neurons):
        result[i] = np.interp(target, source, spk_running[i])
    return result.reshape(n_neurons, ntrials, n_bins)
```
```python
n_fr = min(n_frames_spk, len(beh['ft_move']))
VRmove = beh['ft_move'][:n_fr] > 0
pos_cum = beh['ft_PosCum'][:n_fr]
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]
...
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
...
for t in range(ntrials):
    neural_t = texture_spk[:, t, :]  # (n_neurons, 40)
```

iii. The AI states this is the paper's standard processing: "Position interpolation: 60 bins per full corridor (6m), then only the first 40 bins (texture area, 4m) are kept. This matches the paper's standard processing and gives uniform trial lengths." It adopted it to solve the variable-trial-length problem: "Since corridors move at a fixed 60 cm/s while running, binning by position is actually equivalent to binning by constant-speed time, which resolves the variable-length issue elegantly." The running-frame restriction is taken from the Methods: "We only considered timepoints during running for analysis."

## 1-e. How are trials filtered based on quality controls?

i. Almost no trial-level quality control is applied. Every one of the `ntrials` trials of every session is written, except trials whose interpolated neural matrix contains a NaN or an Inf. A session is skipped if it has fewer than 2 trials, if its behavior key is missing, if it has fewer than 2 running frames, or if fewer than 2 trials survive the NaN/Inf check. No session was actually dropped: all 89 sessions and all 38,110 trials in the dataset were written. Note that there is no length/outlier filter (it is largely unnecessary in this representation, because stationary frames are removed before interpolation, so a mouse that parks in the corridor contributes no extra bins), and there is no check that a trial was actually imaged — for a trial past the end of the imaging, `np.interp` clamps to the last frame's values instead of producing NaN, so such a trial is silently kept as a flat 40-bin trace.

ii.
```python
ntrials = int(beh['ntrials'])
if ntrials < 2:
    print(f" -> SKIP: {ntrials} trials")
    continue
...
if len(pos_cum_running) < 2:
    print(" -> SKIP: insufficient running frames")
```
```python
for t in range(ntrials):
    neural_t = texture_spk[:, t, :]  # (n_neurons, 40)

    # Skip trials with NaN/Inf
    if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
        continue
```
```python
if len(session_neural) < 2:
    print(f" -> SKIP: {len(session_neural)} valid trials")
    continue
```

iii. The docstring lists only "6. Trials with NaN/Inf in neural data after interpolation are excluded." In the trajectory the AI planned to handle "edge cases like trials with NaN start/gray frames (skip them) and sessions with too few valid trials (skip if under 2)", the second of which is required by the instructions ("There needs to be at least two trials within each session"). It never revisited trial-length outliers, because in the position-interpolated representation a stopped animal simply contributes no frames rather than hundreds of stationary bins.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy` — a list of (neurons x frames) arrays, one per imaging plane, concatenated along the neuron axis exactly as `utils.load_spk` does. The per-neuron visual area comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`. The behavior variables `ft_move` and `ft_PosCum` are used to build the axis the traces are interpolated onto.

ii.
```python
spk = load_spk(mname, datexp, blk)          # np.concatenate(d['spks'], axis=0)
n_neurons_total, n_frames_spk = spk.shape
iarea = load_iarea(mname, datexp)
```

iii. These are the deconvolved Suite2p traces the paper analyses throughout; the AI describes them in the metadata as "Position-interpolated deconvolved spikes (Suite2p with 0.75s decay)" and in the docstring as "Neural data: Deconvolved spikes from Suite2p".

## 2-b. How is the `neural` data processed?

i. Three steps. (1) Neurons outside the visual areas are dropped (see 2-c). (2) Frames are restricted to those where the VR was moving (`ft_move > 0`), after truncating the behavior to the number of imaged frames. (3) The remaining traces are linearly interpolated from the cumulative-position axis onto a uniform grid of 1-dm position bins, 60 per corridor, of which only the first 40 (the texture area) are computed — a direct re-implementation of `utils.spk_pos_interp`. No dF/F, deconvolution, smoothing, z-scoring or normalisation is applied. The result is stored as `float32` (one 40-column slice per trial, ~46k neurons per session, giving a 274 GB pickle).

ii.
```python
VRmove = beh['ft_move'][:n_fr] > 0
spk_running = spk[:, :n_fr][:, VRmove]
pos_cum_running = pos_cum[VRmove]
texture_spk = position_interpolate(spk_running, pos_cum_running, ntrials, n_bins=N_BINS)
```
```python
result = np.empty((n_neurons, len(target)), dtype=np.float32)
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
return result.reshape(n_neurons, ntrials, n_bins)
```

iii. "Only frames where the VR is moving (`ft_move > 0`) are used, matching the paper: 'We only considered timepoints during running for analysis'" and "Position interpolation ... matches the paper's standard processing and gives uniform trial lengths". The AI justified equating position bins with time bins by the Methods statement that "we fixed the speed of the virtual reality when mice ran faster than a speed threshold, and kept the virtual reality stationary otherwise": "Since corridors move at a fixed 60 cm/s while running, binning by position is actually equivalent to binning by constant-speed time." It deliberately kept `float32` rather than `float16`: "since the paper's underlying computations use float64 and the decoder itself converts to float32, I decide to stick with float32 for consistency and correctness rather than introducing a storage optimization that isn't warranted." Notably, the AI had earlier identified the main downside of this representation — "position becomes deterministic under position-interpolation, so predicting position from neural activity would be trivial and defeats the decoder's purpose. I need to revert to raw frame-based data" — but then reversed itself and kept position interpolation without resolving that objection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is the retinotopic area: neurons with `iarea == -1` (unassigned) or `iarea == 7` (non-visual) are dropped; everything else is kept and mapped onto the four coarse areas of `utils.neu_area_ID` (V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4). No SNR, firing-rate or variance criterion is applied. This keeps 4,105,393 neurons (V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180), exactly the same counts as the expert solution.

ii.
```python
def get_brain_region_idx(iarea):
    """Map iarea values to brain region indices, following neu_area_ID in utils.py."""
    region = np.empty(len(iarea), dtype=np.int32)
    region[:] = -1  # temporary
    region[iarea == 8] = 0  # V1
    for a in [0, 1, 2, 9]:
        region[iarea == a] = 1  # mHV
    for a in [5, 6]:
        region[iarea == a] = 2  # lHV
    for a in [3, 4]:
        region[iarea == a] = 3  # aHV
    return region
```
```python
iarea = load_iarea(mname, datexp)
vc_mask = (iarea != -1) & (iarea != 7)  # visual cortex mask
brain_region = get_brain_region_idx(iarea[vc_mask])
spk = spk[vc_mask]
```

iii. "Neuron filtering: Neurons with `iarea == -1` (unassigned) or `iarea == 7` (non-visual) are excluded, matching the paper's convention (e.g. `idx_neu = (arid!=-1) & (arid != 7)` in `Get_density_map`)." In the trajectory: "the paper consistently limits its analyses to classified visual cortex neurons, and unclassified ones (`iarea == -1`) aren't reliably part of that region anyway — so filtering to visual cortex matches the paper's methodology and also keeps the dataset smaller." The cells themselves were already curated by the authors with the Suite2p classifier, so no further quality filter was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry. Bin 0 of every trial is the cumulative position at which the mouse entered that corridor (`ft_PosCum = 60 * t`) and bin 39 is 3.9 m into the texture; all trials are therefore exactly 40 bins long and start at the alignment event, with `off_start = 0.0` and `off_end = 40/6 = 6.67 s`. The alignment is a *distance* alignment that is treated as a time alignment under the constant-VR-speed assumption; wall-clock duration of a traversal is not preserved (stationary periods are excised and the remaining time per bin is only approximately constant).

ii.
```python
bin_offsets = np.arange(n_bins, dtype=np.float64) / CORRIDOR_LENGTH
trial_starts = np.arange(ntrials, dtype=np.float64)
target = (trial_starts[:, None] + bin_offsets[None, :]).ravel()
source = pos_cum_running / CORRIDOR_LENGTH
```
```python
'temporal_alignment_event': 'corridor entry (start of texture area)',
'off_start': 0.0,
'off_end': N_BINS * TIME_PER_BIN,
```

iii. "Temporally aligned based on trial start (corridor entry)" is the instruction; the AI satisfies it via the position grid, arguing that "position-interpolated bins effectively serve as a consistent time axis too — 60 bins per trial across the 6 m corridor at 0.1 m resolution gives uniform trial lengths, which resolves the variable-duration issue from stop/start running", and that fixed-length trials avoid padding or truncation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each bin is one decimetre of virtual corridor, declared as 166.67 ms (`1 dm / 60 cm s^-1`), and `metadata['time_bin_size'] = 166.666...`. Rebinning is applied and it is the central transformation of the conversion: the native imaging grid (3.17 Hz, 315 ms bins, ~22 running frames per traversal) is replaced by a 40-bin position grid, i.e. the data is *upsampled* by roughly a factor of two by linear interpolation. The nominal 166.67 ms is an approximation: the VR advances at most 2 dm per frame (60 cm/s) but averages ~1.75 dm per frame while "moving", so a bin is really ~180 ms on average, and the number of running frames per traversal still varies (median 22, 95th percentile 29 in the sessions checked), so the real duration of a 40-bin trial varies by ~40% even after stationary time is excised.

ii.
```python
CORRIDOR_LENGTH = 60    # decimeters (6m total corridor)
TEXTURE_LENGTH = 40     # decimeters (4m texture area)
VR_SPEED_DM_S = 6.0     # VR speed in dm/s (60 cm/s)
TIME_PER_BIN = 1.0 / VR_SPEED_DM_S  # seconds per position bin (1/6 s)
TIME_PER_BIN_MS = TIME_PER_BIN * 1000  # ~166.67 ms
```
```python
'time_bin_size': TIME_PER_BIN_MS,
'calcium_imaging_frame_rate_hz': 3.17,
```

iii. "Time bin interpretation: VR moves at constant 60 cm/s when running, so each 1-dm position bin = 1/6 second = 166.67 ms." The AI reasoned that this reconciles the paper's position-based processing with the instruction's requirement of one constant bin size across all trials and sessions.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundPos'][t]`, the position (in decimetres) at which the sound cue was played on that trial, together with the bin index. The frame-based variables `SoundFr` / `SoundTime` / `ft` are not used.

ii.
```python
sound_pos = beh['SoundPos'][t]  # in decimeters
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
```

iii. The AI verified the units before using it: "Checking the unsup_test1 session, SoundPos ranges close to [4.2, 36.0] dm, which lines up well with the 5–35 dm spec for the uniform distribution, confirming positions are stored in decimeters." Because the neural axis is the position axis, the position of the cue is the natural source.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The signed distance in decimetres from the current bin to the cue position is converted to seconds by the constant VR speed: `(SoundPos - bin) / 6`. It is positive before the cue and negative after, matching the name. It is a time-varying input, one value per bin, and the resulting range is ±6.5 s (versus ±73 s for the expert's wall-clock version, because stationary time is excised and the VR time base is compressed).

ii.
```python
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. Same constant-speed argument as 2-e: distance to the cue and time to the cue are the same variable up to the factor 1/6 as long as the VR is moving. The AI did not represent the cue as a binary event series because it was asked for a "time to" input.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on exactly the same 40 position bins as the neural matrix of that trial, so it is aligned by construction: `input` has shape (4, 40) and `neural` has shape (n_neurons, 40).

ii.
```python
neural_t = texture_spk[:, t, :]        # (n_neurons, 40)
...
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)  # (4, 40)
```

iii. All streams in this conversion share the single position grid, so no separate alignment step is needed.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date `datexp` in the index entry of every session of that mouse; the earliest date for a mouse is its reference day 0.

ii.
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_dates.setdefault(mname, []).append(date)
...
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
```

iii. The dates are the only field that orders a mouse's sessions; the AI noted the ambiguity of the variable across cohorts ("The 'day of training' variable has a similar ambiguity, since it means something different depending on whether mice are supervised, unsupervised, or naive") and settled on a purely date-based definition that is available for every cohort.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed between the session's date and the mouse's first recorded session: `(session_date - first_date).days`, as a float, broadcast across the 40 bins of every trial of the session. The values run from 0 to 92 (the expert's per-mouse session ordinal runs 0–7 instead).

ii.
```python
session_date = datetime.strptime(datexp, '%Y_%m_%d')
day_of_training = float((session_date - mouse_first_date[mname]).days)
...
day_arr = np.full(N_BINS, day_of_training, dtype=np.float32)
```

iii. "Day of training: computed as days from first recording for each mouse." The AI kept the input time-varying-shaped for uniformity: "day_of_training and reward_availability stay constant across bins but repeated per trial", after checking that the decoder also accepts 1-D per-trial inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From no raw variable at all: it is the bin index of the position grid (which itself derives from `ft_PosCum`), scaled by the constant bin duration. `StartFr`, `Trial_start_time` and `ft` are not used.

ii.
```python
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
```

iii. From the trajectory: "time since trial start is just bin index scaled by bin duration" — under the constant-VR-speed reading of the position axis, elapsed time since corridor entry is a deterministic function of the distance travelled. The AI had also noticed that `StartFr` is sometimes -1 in the raw data, which this construction sidesteps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `bin_index / 6` seconds, i.e. a fixed ramp from 0 to 6.5 s. Because it is precomputed once outside the session loop and never modified, it is **identical for every trial of every session**, so it carries no across-trial information and it does not reflect the real elapsed time of a traversal (real running time per traversal has a median of ~6.9 s but a 95th percentile of ~9 s, and the total wall-clock duration including stops can be tens of seconds).

ii.
```python
# Pre-compute static arrays
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)
...
input_t = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. Same constant-speed justification; the AI treated the position grid as a uniform time grid, in which case time since corridor entry is exactly the bin index times the bin size.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined on the same 40-bin grid as the neural matrix, so it is aligned by construction with bin 0 at corridor entry.

ii.
```python
neural_t = texture_spk[:, t, :]     # (n_neurons, 40)
input_t  = np.stack([...], axis=0)  # (4, 40), row 2 = time_since_start
```

iii. As for all streams here, alignment is inherited from the shared position grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew'][t]`, the boolean flag marking trials run in the rewarded corridor.

ii.
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. "Reward availability comes from the `isRew` field in behavior data." The AI was aware that this is always 0 for the naive/unsupervised/grating cohorts and accepted that: "the reward input just ends up constant for those sessions."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and a broadcast across the 40 bins of the trial, giving values in {0.0, 1.0}.

ii.
```python
reward_arr = np.full(N_BINS, float(beh['isRew'][t]), dtype=np.float32)
```

iii. It is already the binary per-trial variable the instructions ask for ("1 if in rewarded corridor, 0 if not").

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName'][t]`, the name of the texture on the walls of that corridor. The vocabulary is collected globally in the first pass from `beh['UniqWalls']` of every session, so that one consistent label set is used across sessions. `TrialStim`, `WallType` and `stim_id` are not used.

ii.
```python
for wn in beh['UniqWalls']:
    all_stimuli.add(str(wn))
...
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
```
```python
stim_name = str(beh['WallName'][t])
stim_cat = stim_to_idx[stim_name]
```

iii. "Since sessions likely use different sets of stimulus names (circle1, leaf1, rock, etc.), I need to build a single global mapping from WallName strings to integers before writing the conversion script", and it chose to collect the names programmatically rather than hard-code them: "I should collect all unique WallName values programmatically rather than hardcode a list." (`WallType` is explicitly flagged as unusable in the authors' own notebook documentation.)

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The raw wall names are used as the categories, sorted alphabetically, giving **15 classes**: `['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5']`. They are *not* collapsed into the four base textures (circle / leaf / rock / wood), so different crops of the same photograph, and the spatially shuffled controls (`leaf1_swap1/2`), are separate classes. The class index is broadcast across the 40 bins of the trial, and `output_values[0]` is the 15-name list. Chance level for this output becomes 6.7% instead of 25%.

ii.
```python
stim_list = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
...
stim_cat = stim_to_idx[stim_name]
stim_arr = np.full(N_BINS, stim_cat, dtype=np.int32)
```
```python
output_values = [
    stim_list,                           # visual_stimulus categories
    ...
]
```

iii. "Since each specific texture counts as its own category, I'll organize this as a two-pass process: first pass gathers unique stimuli..." — the AI treated each distinct wall pattern as its own category, arguing that the different crops and shuffles are physically different images. It reported the consequence explicitly in its summary: "visual_stimulus (15 classes) 58.9% vs chance 6.7%".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickPos']` (the corridor position of each lick, in decimetres) and `beh['LickTrind']` (the 0-based trial index of each lick). `LickFr` / `LickTime` are not used.

ii.
```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
...
trial_mask = lick_trind == t
trial_lick_pos = lick_pos[trial_mask]
```

iii. "I'm defining licking per position bin using `LickPos`"; because the whole dataset is on a position axis, the positional stamp of each lick is the matching variable. The AI checked the units of `LickPos`: "Licking positions are in decimeters (0–40 range for texture area)".

## 8-b. What processing is involved in computing `output` *Licking*?

i. A 40-long binary vector per trial. Licks of that trial with position in [0, 40) are floored to their decimetre bin and that bin is set to 1; licks in the grey space (position >= 40, roughly 5–15% of licks depending on session) are discarded together with the grey space itself. Multiple licks in one bin collapse to a single 1. The resulting overall lick rate is 1.9% of bins (the expert's frame-based version gives 4.1%).

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

iii. "For lick timing, I need to check what units LickPos uses — likely decimeters or meters — so I can properly determine which position bin each lick falls into"; the instruction asks for a binary time series (0 = not licking, 1 = licking), which this produces directly on the trial's bin grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Via the shared 40 position bins: a lick at position p decimetres is assigned to bin floor(p), which is the same bin as the interpolated neural activity at that position in that trial.

ii.
```python
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)   # (4, 40)
session_neural.append(neural_t)                                            # (n_neurons, 40)
session_output.append(output_t)
```

iii. The position stamp is the common reference for all streams in this conversion. (One consequence the AI does not discuss: licks emitted while the animal was stationary — e.g. stopping to collect water — are still assigned to their position bin even though the corresponding imaging frames were removed from the neural data.)

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Indirectly from `ft_PosCum`, through the interpolation grid: because the neural data is resampled onto regular 1-dm bins, the position of bin k is exactly k decimetres, so the output is computed from the bin index alone. `ft_Pos` is never read.

ii.
```python
# Pre-compute static arrays
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
...
# Position: 4 bins of 1m each (deterministic from position index)
pos_arr = position_bins.copy()
```

iii. The code comment states it: "deterministic from position index". Position is a property of the axis in this representation.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 bins are divided into 4 blocks of 10 bins (10 dm = 1 m), giving the fixed label vector `[0]*10 + [1]*10 + [2]*10 + [3]*10`, which is copied unchanged into every trial of every session. The output therefore has no trial-to-trial variability at all and is exactly 25% per class by construction.

ii.
```python
POSITION_N_CATS = 4      # 4 spatial bins of 1m each
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
pos_arr = position_bins.copy()
```
```python
['0-1m', '1-2m', '2-3m', '3-4m'],      # position categories
```

iii. "position in the corridor as four bins spanning the 4-meter track". The AI recognised the consequence at one point — "position becomes deterministic under position-interpolation, so predicting position from neural activity would be trivial and defeats the decoder's purpose" — and in the final summary it reported it as expected: "Position bins are perfectly balanced at 25% each (expected since we used uniform position binning)."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1 m bins over the 4 m texture area: bins 0–9 → class 0 (0–1 m), 10–19 → class 1, 20–29 → class 2, 30–39 → class 3. Exactly the discretisation the instructions ask for, and the same boundaries as the expert's `ft_Pos // 10`. The grey space (4–6 m) is not represented.

ii.
```python
position_bins = np.repeat(np.arange(POSITION_N_CATS), N_BINS // POSITION_N_CATS).astype(np.int32)
```

iii. "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins" is the instruction; 10 one-decimetre bins make one metre.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Perfectly and trivially: the label of bin k is the position of bin k of the same trial's neural matrix.

ii.
```python
neural_t = texture_spk[:, t, :]                                      # (n_neurons, 40)
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], 0)  # (4, 40)
```

iii. Alignment is inherent to the position-binned representation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['run_pos']`, the authors' running speed already interpolated into a (trials x 60 positions) matrix; only the first 40 columns (texture area) are used. The per-frame `ft_RunSpeed` (used by the expert) is not used.

ii.
```python
speed = beh['run_pos'][:, :N_BINS]     # first pass, for the quartile thresholds
...
speed = beh['run_pos'][t, :N_BINS].copy()   # second pass, per trial
```

iii. "For speed quartiles, `run_pos` is the right choice since it's already interpolated to match position bins, unlike raw frame-by-frame speed." This keeps the speed variable on exactly the same axis as the neural data. (Note the VR speed is capped, so `run_pos` is the animal's true running speed on the ball and does vary, unlike the VR advance rate.)

## 10-b. What processing is involved in computing `output` *Running speed*?

i. In the first pass all non-NaN values of `run_pos[:, :40]` from all 89 sessions are pooled and the 25th/50th/75th percentiles are computed once globally (16.58, 28.72, 43.29). In the second pass each trial's 40 speeds are taken, NaNs are replaced by 0.0, and `np.digitize` assigns each value to one of 4 classes using those global thresholds. The result is exactly 25% of all bins per class, and the thresholds are stored in the metadata.

ii.
```python
all_speed_arr = np.concatenate(all_speed_values)
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
print(f"Speed quartiles: {speed_quartiles}")
```
```python
speed = beh['run_pos'][t, :N_BINS].copy()
speed[np.isnan(speed)] = 0.0
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```
```python
'speed_quartile_thresholds': speed_quartiles.tolist(),
```

iii. "Running speed discretized into quartiles computed across all sessions' texture area" (docstring) and "I'll pool speed values across all sessions to compute quartile thresholds for discretization", which matches the instruction "Running speed discretized into 4 bins, each corresponding to 25% of the data". The AI checked units and decided they did not matter ("since I'm computing quartiles for discretization, actual units don't matter as long as I use relative bins"), and noticed but dismissed an outlier: "run_pos[0,0] is unusually high at 553.2 ... quartile discretization is fairly robust to a few outliers."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global thresholds (16.58 / 28.72 / 43.29 speed units) applied with `np.digitize`, producing classes 0–3 labelled `['Q1','Q2','Q3','Q4']`; values below the 25th percentile (including the NaN→0 substitutions and the negative, backwards-running values) fall in Q1. Thresholds are global rather than per session, so a session's own distribution may be skewed away from 25% per class even though the pooled distribution is exactly uniform.

ii.
```python
speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```
```python
['Q1', 'Q2', 'Q3', 'Q4'],              # running speed quartiles
```

iii. Global pooling was chosen so that the four classes each hold 25% of the whole dataset, which is the literal reading of the instruction.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos[t, k]` is the speed at position bin k of trial t, exactly the bin of the neural matrix, so the two are aligned by construction with no resampling.

ii.
```python
speed = beh['run_pos'][t, :N_BINS].copy()
...
output_t = np.stack([stim_arr, lick_binary, pos_arr, speed_cat], axis=0)   # (4, 40)
```

iii. "`run_pos` is already interpolated to match position bins" — the authors computed it on the same 60-bin corridor grid used for the neural interpolation.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards: the behavior is truncated to the number of imaged frames (`n_fr = min(n_frames_spk, len(ft_move))`); a missing behavior key prints a warning and the session is skipped; a session with fewer than 2 trials, fewer than 2 running frames, or fewer than 2 surviving trials is skipped; trials whose interpolated neural data contains NaN or Inf are dropped; NaN running speeds are replaced by 0; licks outside [0, 40) decimetres are discarded; `gc.collect()` and `del` are used to release large temporaries. Two gaps: a trial that extends past the end of the imaging is not detected, because `np.interp` clamps to the last available frame instead of producing NaN (the expert drops one such trial); and no exception handler wraps a session, so an unexpected failure would abort the whole 2-hour run.

ii.
```python
n_fr = min(n_frames_spk, len(beh['ft_move']))
```
```python
beh = beh_cache[exp_type].get(beh_key)
if beh is None:
    print(" -> SKIP: behavior not found")
    continue
```
```python
if np.any(np.isnan(neural_t)) or np.any(np.isinf(neural_t)):
    continue
```
```python
speed[np.isnan(speed)] = 0.0
```

iii. The truncation follows the reference notebook, which does `beh['ft_move'][:nfr]` and `beh['ft_PosCum'][:nfr]` with `nfr = spk.shape[1]`. The AI planned these guards in advance: "handling edge cases like trials with NaN start/gray frames (skip them) and sessions with too few valid trials (skip if under 2)" and "handling NaN speed values by filling before discretizing." It had also spotted that "`StartFr` = -1 for unsup_test1 session — need to handle this", which the position-based construction makes moot.

## 12-a. What are the most time-consuming steps of the code?

i. Three steps dominate. (1) Reading the spike files: 405 GB of `.npy` across 89 sessions, plus the `np.concatenate` of the per-plane arrays, which doubles the peak. (2) The interpolation itself: a Python loop calling `np.interp` once per neuron, measured by the AI at ~18 s per session for 55k neurons (~30 min in total). (3) Writing the 274 GB pickle. The whole conversion took about 2 hours; downstream, the file size also made the decoder run out of GPU memory and fall back to a multi-hour CPU run.

ii.
```python
d = np.load(fn, allow_pickle=True).item()
return np.concatenate(d['spks'], axis=0)
```
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```
```python
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The AI measured the interpolation ("The interpolation takes ~18 seconds per session with 55K neurons. Total estimated time: ~27 minutes for 89 sessions") and reduced it by computing 40 instead of 60 bins per trial: "Let me optimize the interpolation to only compute texture area bins." It also estimated the pickle write time ("about 12 minutes to write the pickle at typical disk speeds") and judged the total acceptable.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron interpolation loop is the main one: `np.interp` is called ~46,000 times per session on the same target/source grid. Since the grid is shared, the bin indices and interpolation weights could be computed once with `np.searchsorted` and applied to all neurons in two vectorised gathers (or one sparse matrix product), removing ~4 million Python-level calls across the dataset. The per-trial loop is also mostly vectorisable: `time_to_cue`, `stim_arr`, `reward_arr` and `speed_cat` could be computed for all trials at once (`np.digitize(beh['run_pos'][:, :40], ...)` on the whole matrix), and the lick vector could be built for all trials with a single 2-D scatter using `LickTrind` and `floor(LickPos)`. The two full passes over the behavior files could also be merged into one.

ii.
```python
for i in range(n_neurons):
    result[i] = np.interp(target, source, spk_running[i])
```
```python
for t in range(ntrials):
    ...
    time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
    ...
    trial_mask = lick_trind == t
    ...
    speed_cat = np.digitize(speed, speed_quartiles).astype(np.int32)
```

iii. The AI gives no justification for the loop; it copied the structure of the paper's `utils.spk_pos_interp`, which loops over neurons in the same way ("for s in range(raw_spk.shape[0]): # loop through neurons"). It did optimise the loop's inner cost by computing only the 40 texture bins instead of all 60.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded and iterated twice, once in the first (statistics) pass and once in the second (conversion) pass — `beh_cache` is rebuilt from scratch after being deleted. Within the trial loop, `np.arange(N_BINS)` is rebuilt for every trial (for `time_to_cue`), `position_bins.copy()` is made for every trial although the array is always the same, and `lick_trind == t` rescans the entire lick array once per trial. `np.percentile` is computed over the pooled speed array of the whole dataset (hundreds of millions of values) in one shot, which is fine, but the values were gathered by re-reading every behavior file.

ii.
```python
for key in session_keys:            # first pass
    ...
    speed = beh['run_pos'][:, :N_BINS]
    ...
del beh_cache
...
beh_cache = {}
for sess_i, key in enumerate(session_keys):    # second pass, reloads every Beh_*.npy
```
```python
time_to_cue = ((sound_pos - np.arange(N_BINS)) * TIME_PER_BIN).astype(np.float32)
pos_arr = position_bins.copy()
trial_mask = lick_trind == t
```

iii. The two-pass design is justified — the global speed quartiles and the global stimulus vocabulary must be known before the first trial is written — and the AI checked that the behavior files are small enough to reread ("each Beh file likely runs around 100 MB across sessions, so with ~23 experiment types the total should stay around 2.3 GB, which is manageable"). The repeated small allocations inside the trial loop are not discussed.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in the size of what is produced. (1) The interpolation writes 40 bins per trial from a median of ~22 real imaging frames, so roughly half of the 274 GB of neural data is interpolated filler that carries no additional information, and it is stored in `float32` rather than `float16` — the AI considered `float16` and reverted. The 274 GB file then forced the decoder onto CPU and a multi-hour run, when the decoder random-projects each session to at most 2,000 neurons and then takes 100 PCs anyway (the AI noted this: "`svd_max_neurons` parameter defaults to 2000 ... so having many neurons is fine"). (2) The position output row is the same constant vector for all 38,110 trials, i.e. ~1.5 M stored integers with zero information. (3) `time_since_start` is likewise identical in every trial. (4) The first pass pools every speed value of the dataset into memory (`all_speed_values`) only to take three percentiles. (5) Per-trial `neural_t` slices are views into `texture_spk`, so `del texture_spk; gc.collect()` does not actually release the session array.

ii.
```python
result = np.empty((n_neurons, len(target)), dtype=np.float32)
```
```python
pos_arr = position_bins.copy()          # identical in every trial
time_since_start = (np.arange(N_BINS, dtype=np.float32) * TIME_PER_BIN)  # identical in every trial
```
```python
all_speed_values.append(speed[valid].ravel())
...
all_speed_arr = np.concatenate(all_speed_values)
speed_quartiles = np.percentile(all_speed_arr, [25, 50, 75])
```
```python
neural_t = texture_spk[:, t, :]   # view; keeps the whole session array alive
...
del texture_spk
gc.collect()
```

iii. The AI decided against `float16` on the grounds of numerical fidelity ("the paper's underlying computations use float64 and the decoder itself converts to float32, I decide to stick with float32 for consistency and correctness rather than introducing a storage optimization that isn't warranted"), and against subsampling neurons on the grounds that the instructions ask to "include all relevant data from the source". It checked that the machine could hold the result ("Peak memory would be roughly 373 GB ... comfortably within the 960 GB ceiling") and proceeded.
