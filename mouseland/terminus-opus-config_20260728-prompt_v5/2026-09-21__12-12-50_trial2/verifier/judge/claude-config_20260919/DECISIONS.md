# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from `/app/data`, which is hard-coded as `DATA_ROOT`. `beh/Imaging_Exp_info.npy` is loaded first as the master index; it is a dict keyed by experiment type (23 types), each value a list of recording entries carrying `mname`, `datexp`, `blk`, `sess#`, `exptype`, `rewType`, `stim_id`. `get_unique_sessions` flattens that index into one record per unique `mname_datexp_blk` key (89 records), accumulating the list of experiment types each recording appears under. For each session the script then reads three things: the deconvolved traces from `spk/<mname>_<datexp>_<blk>_neural_data.npy` (a dict whose `spks` entry is one array per imaging plane, concatenated along the neuron axis), the visual-area labels from `retinotopy/<mname>_<datexp>_trans.npz` (`iarea`), and the behavior from `beh/Beh_<exp_type>.npy`. Behavior files hold many sessions each, so `load_beh_for_session` loads a file once and caches every session it contains in the module-level `_beh_cache`; a session whose key carries a stimulus-type suffix (e.g. `TX124_2023_12_24_1_swap1`) is found by a `startswith` match. Two extra full passes over all behavior files are made before conversion begins, one to collect the global list of wall names and one to compute global speed quartiles.

ii.
```python
DATA_ROOT = '/app/data'

def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()

def get_unique_sessions(exp_info):
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in sessions:
                sessions[key] = {... 'exp_types': [], 'sess_num': s.get('sess#', 0), ...}
            sessions[key]['exp_types'].append(exp_type)
    return sessions

def load_spk(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
    return np.concatenate([nspk for nspk in spk_data['spks']], 0)

def load_retino(mname, datexp):
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz'),
                     allow_pickle=True)
    return dtrans['iarea']
```
```python
def load_beh_for_session(session_key, exp_info):
    if session_key in _beh_cache:
        return _beh_cache[session_key]
    sessions = get_unique_sessions(exp_info)
    sess = sessions[session_key]
    for exp_type in sess['exp_types']:
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if os.path.exists(beh_file):
            beh = np.load(beh_file, allow_pickle=True).item()
            for k, v in beh.items():
                if k not in _beh_cache:
                    _beh_cache[k] = v
            if session_key in _beh_cache:
                return _beh_cache[session_key]
            for k in beh.keys():
                if k.startswith(session_key):
                    _beh_cache[session_key] = beh[k]
                    return beh[k]
    raise ValueError(f"Could not find behavioral data for session {session_key}")
```

iii. CONVERSION_NOTES Step 1 records `load_exp_beh`, `load_spk`, `load_retino` from the reference `utils.py` as the loading functions, and Step 10 Check 6(a) states "Data loading: Same as load_spk (concatenate planes) and load_retino. MATCH." The caching is justified in Step 6 as "Cached behavioral data loading to avoid reloading large files".

## 1-b. How are the data split into subjects (mice)?

i. The subject is `mname` from the experiment-info entry, carried on every session record. Sessions are iterated in sorted session-key order and each new `mname` is appended to `unique_subjects` with its index stored in `subject_map`; `subject_idx` records that index per session. This yields 19 mice, with 1–8 sessions each, matching the paper's "89 recordings in 19 mice".

ii.
```python
for s_idx, session_key in enumerate(session_keys):
    sess = sessions[session_key]
    mname = sess['mname']
    if mname not in subject_map:
        subject_map[mname] = len(unique_subjects)
        unique_subjects.append(mname)
    ...
    subject_idx_list.append(subject_map[mname])
```
```python
'subjects': unique_subjects,
'subject_idx': np.array(subject_idx_list),
```

iii. No separate justification is given beyond CONVERSION_NOTES Step 2/Step 3, which report 19 subjects from the data and 19 mice from the paper and mark them consistent ("Mice | 19 unique | 19 mnames | 19 mice | Consistent").

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block: the key `mname_datexp_blk`. Because the master index lists the same recording under several experiment types, `get_unique_sessions` keeps only the first occurrence of each key (`if key not in sessions`) and appends further experiment types to that record's `exp_types` list. Session keys are then sorted, giving 89 sessions, which equals the number of spike files and the paper's 89 recordings. No session is dropped for any reason.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in sessions:
    sessions[key] = {...}
sessions[key]['exp_types'].append(exp_type)
...
session_keys = sorted(sessions.keys())
```

iii. CONVERSION_NOTES Step 2 notes "Spike files 89 / Unique sessions 89" and Step 5 Key Decision 2: "All 89 sessions included: Each unique recording once." The agent's reasoning (trajectory step 315) records that when a session appears under multiple experiment types the behavioral data is identical and only the `stim_id` focus differs, so a single copy is kept.

## 1-d. How are the data split into trials?

i. Trials are taken exactly as the behavior declares them: `beh['ntrials']` trials per session, no more and no less. The split is realised through the reference position-interpolation: only frames where the virtual reality is moving (`ft_move > 0`) are kept, their cumulative VR position `ft_PosCum` is divided by `Corridor_Length` (= 60) so that whole numbers mark trial boundaries, and the neural traces are interpolated onto the regular grid `np.arange(0, n_trials, 1/60)`. The result is reshaped to `(n_neurons, n_trials, 60)`, so every trial is exactly 60 position bins covering the 4 m textured corridor (bins 0–39) plus the 2 m grey space (bins 40–59). Every trial in every session therefore has the same length. Total = 38,110 trials across 89 sessions.

ii.
```python
ft_move  = beh['ft_move'][:n_frames]
ft_pos_cum = beh['ft_PosCum'][:n_frames]
vr_moving = ft_move > 0

interp_spk = position_interpolate_spk(
    spk[:, vr_moving], ft_pos_cum[vr_moving],
    corridor_len, n_trials, N_POS_BINS)
```
```python
def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    src_pos = accum_pos / corridor_len
    ...
            interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
    return interp_spk.reshape(n_neurons, n_trials, n_bins)
```
```python
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```

iii. CONVERSION_NOTES Step 5 Key Decisions 1 and 5: "Position-based binning (60 bins): Matches reference code exactly" and "Grey space included: Full 60 bins per trial"; Step 10 Check 6(c)/(d): "Temporal alignment: Position interpolation using cumulative position, same as spk_pos_interp. MATCH" and "Binning: 60 bins per trial, same as data_process_script cell 9. MATCH". In the trajectory the agent weighed excluding the grey space and decided to "keep all 60 bins for the trial" because "the reference code includes grey space (60 bins total)" and "a trial includes both corridor and grey space".

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. Every one of the `ntrials` trials of every one of the 89 sessions is written out (38,110 trials). There is no minimum or maximum trial-length rule, no check that a trial was actually imaged, no check that a session retains at least two trials, and no session is dropped. The only data-level exclusion anywhere in the script is at the frame level (non-moving frames are excluded from the interpolation source) and at the neuron level (non-visual-cortex neurons).

ii. There is no filtering code. The trial list is simply `range(n_trials)`:
```python
n_trials = beh['ntrials']
...
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
...
for t in range(n_trials):
    inp = np.stack([...])
    input_trials.append(inp)
```

iii. CONVERSION_NOTES Step 3 lists as the only trial curation rule: "Trial curation: Only frames where VR is moving (ft_move > 0) used in position interpolation". No further justification for the absence of trial filtering appears in the notes or in the trajectory; the agent never raised trial exclusion as a topic. Implicitly, warping every traversal onto a fixed 60-bin position grid makes trial duration irrelevant, so slow or stopped traversals do not produce long trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along axis 0. The area label of each neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`. The alignment of the traces to trials uses the behavior variables `ft_move`, `ft_PosCum` and `Corridor_Length`.

ii.
```python
spk = load_spk(mname, datexp, blk)          # np.concatenate(spk_data['spks'], 0)
n_neurons_total, n_frames = spk.shape
iarea = load_retino(mname, datexp)
assert len(iarea) == n_neurons_total, \
    f"Retinotopy mismatch: {len(iarea)} vs {n_neurons_total} neurons"
```

iii. CONVERSION_NOTES Step 3 quotes the methods, "All analyses were based on deconvolved fluorescence traces", and Step 1 notes "Neural data: deconvolved fluorescence traces from Suite2p", so the file contents are used directly. The neuron-count assertion is the agent's own guard that the retinotopy vector and the concatenated planes are in the same order.

## 2-b. How is the `neural` data processed?

i. No dF/F and no deconvolution is applied — the files already contain Suite2p deconvolved traces. The one processing step is the reference position interpolation: frames where the VR is stationary are discarded, and each neuron's trace over the remaining frames is linearly interpolated (`np.interp`) from the cumulative-position axis `ft_PosCum/60` onto a uniform grid of 60 bins per trial. Neurons are processed in batches of 2000 rows, one `np.interp` call per neuron, into a preallocated `(n_neurons, n_trials*60)` `float32` array that is then reshaped. No smoothing, normalisation, z-scoring or PCA is applied, and the result is stored as `float32` (yielding a 410.69 GB pickle).

ii.
```python
def position_interpolate_spk(raw_spk, accum_pos, corridor_len, n_trials, n_bins=60):
    lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
    src_pos = accum_pos / corridor_len
    n_neurons = raw_spk.shape[0]
    interp_spk = np.zeros((n_neurons, n_trials * n_bins), dtype=np.float32)
    batch_size = 2000
    for i in range(0, n_neurons, batch_size):
        end_i = min(i + batch_size, n_neurons)
        batch = raw_spk[i:end_i]
        for s in range(batch.shape[0]):
            interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
    return interp_spk.reshape(n_neurons, n_trials, n_bins)
```

iii. CONVERSION_NOTES Step 1 identifies `spk_pos_interp` / `get_interpPos_spk` as the reference processing functions and Step 10 Check 6(c)/(d) claims an exact match with them. Step 6 states "Optimized interpolation using np.interp (4x speedup over scipy interp1d)". The agent explicitly considered `float16` to shrink the output ("Option: Use float16 instead of float32") but kept `float32` after reading the decoder and concluding it "handles large neural data".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is anatomical: `iarea` is mapped to one of V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4); any neuron whose `iarea` is outside these sets (i.e. −1 or 7) is dropped. No firing-rate, SNR or variance criterion is applied. This keeps 17,363–78,815 neurons per session (mean 46,128) and 4,105,393 neurons in total, split V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180.

ii.
```python
AREA_MAPPING = {'V1': [8], 'mHV': [0, 1, 2, 9], 'lHV': [5, 6], 'aHV': [3, 4]}
EXCLUDED_AREAS = [-1, 7]  # outside visual cortex

def get_brain_region_idx(iarea):
    region_idx = np.full(len(iarea), -1, dtype=int)
    for r_idx, (region, areas) in enumerate(AREA_MAPPING.items()):
        for area_val in areas:
            region_idx[iarea == area_val] = r_idx
    valid_mask = region_idx >= 0
    return valid_mask, region_idx
```
```python
valid_mask, region_idx = get_brain_region_idx(iarea)
spk = spk[valid_mask]
region_idx = region_idx[valid_mask]
```

iii. CONVERSION_NOTES Step 3 Curation: "Neuron curation: Exclude neurons outside visual cortex (iarea == -1 or 7)", and Step 10 Check 7: "Brain region mapping: V1=8, mHV=0|1|2|9, lHV=5|6, aHV=3|4 matches neu_area_ID exactly. PASS." The reference `utils.neu_area_ID` applies no further quality filter, so none is added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Because the interpolation grid is `np.arange(0, n_trials, 1/60)` on the axis `ft_PosCum/Corridor_Length`, integer values of that axis are exactly corridor entries, so bin 0 of every trial is corridor entry by construction. Every trial then runs for the same 60 bins (4 m of texture + 2 m of grey space) up to the next corridor entry. `metadata['temporal_alignment_event'] = 'Trial start (corridor entry)'`, `off_start = 0.0`, `off_end = 10.0` s. Nothing is padded and nothing is truncated: each traversal is warped onto the common grid.

ii.
```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)   # integer values == corridor entries
src_pos = accum_pos / corridor_len
...
interp_spk = interp_spk.reshape(n_neurons, n_trials, n_bins)
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': TOTAL_LENGTH_M / VR_SPEED,  # ~10s for full corridor + grey
```

iii. CONVERSION_NOTES Step 3 Processing Details: "Trial alignment: Corridor entry (trial start)". In the trajectory the agent argues the position interpolation is equivalent to temporal alignment: "The reference code uses position-based interpolation which effectively does this since VR moves at constant speed."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the data are rebinned. Native acquisition is 3.17 Hz (315 ms per imaging frame). The conversion resamples each trial onto 60 uniform position bins of 1 decimetre. Because the methods state the corridor advances at a fixed 60 cm/s whenever the mouse is running, the agent converts 0.1 m to 1/6 s and reports `time_bin_size = 166.67 ms`, `off_end = 10.0` s per trial. In practice a traversal contains only ~23–34 moving imaging frames, so the 60-bin grid is an ~1.9× *upsampling* of the native frame rate, and the bins are uniform in VR position (and in "VR-moving" time) rather than in wall-clock time, since frames with the VR stationary are removed before interpolation.

ii.
```python
FS = 3.17            # Hz, calcium imaging frame rate
VR_SPEED = 0.6       # m/s = 60 cm/s constant VR speed
N_POS_BINS = 60
BIN_SIZE_M = TOTAL_LENGTH_M / N_POS_BINS   # 0.1 m per bin
TIME_PER_BIN = BIN_SIZE_M / VR_SPEED       # ~0.1667 s per bin
TIME_BIN_MS = TIME_PER_BIN * 1000          # ~166.67 ms
```
```python
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FS,
'position_bin_size_m': BIN_SIZE_M,
```

iii. CONVERSION_NOTES Step 3 quotes "bins of position: 60, each bin is 1 decimeter" and "the virtual corridors always moved at a constant speed (60 cm s−1)", and Step 5 Key Decision 4 sets "Time bin size: ~166.67 ms (0.1m / 0.6 m/s)". The agent's stated rationale (trajectory) is that constant VR speed makes a position bin equivalent to a fixed time bin, so a position grid satisfies the "same bin size for all trials and sessions" requirement while matching the reference pipeline.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundPos']`, the position (in decimetres along the 60-unit corridor+grey axis) at which the sound cue was played on each trial. No timing variable (`SoundTime`, `SoundFr`, `ft`) is used; the cue position is converted to a time offset using the constant VR speed.

ii.
```python
sound_pos = beh['SoundPos']  # position units
time_to_cue = make_time_to_sound_cue(sound_pos, N_POS_BINS)
```

iii. CONVERSION_NOTES Step 5 mapping table: "SoundPos | input[0]: time_to_sound_cue | Convert position distance to time (seconds) | Continuous, time-varying". Because the neural data live on a position axis, the cue's position is the natural anchor. Step 10 Check 3 sanity-checks the range: "SoundPos range [4.2, 36.0] ≈ expected [5, 35]".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every trial the signed distance in position bins from each bin centre (`bin + 0.5`) to the cue position is computed, then multiplied by 0.1 m/bin and divided by 0.6 m/s to give seconds. The convention is positive before the cue (cue still ahead) and negative after, matching the name "time **to** sound cue". Realised range over the full dataset is [−9.7, +6.4] s, consistent with cue positions of 0.4–3.6 m in a 10 s trial. The value is time-varying within a trial and stored as `float32`.

ii.
```python
def make_time_to_sound_cue(sound_pos, n_bins=60):
    positions = np.arange(n_bins) + 0.5           # center of each bin
    dist = sound_pos[:, np.newaxis] - positions[np.newaxis, :]   # positive = cue ahead
    time_to_cue = (dist * BIN_SIZE_M / VR_SPEED).astype(np.float32)
    return time_to_cue
```

iii. Trajectory: "Time to sound cue: At each position bin, compute distance to sound cue position, convert to time using VR speed (60 cm/s)." The constant-speed conversion is justified from the methods statement that the VR advances at a fixed 60 cm/s.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same 60 position bins per trial that the neural array uses, so bin *b* of `input[0]` and column *b* of `neural` refer to the same point in the traversal. It is stacked into the per-trial `(4, 60)` input array.

ii.
```python
for t in range(n_trials):
    inp = np.stack([
        time_to_cue[t],                                            # (n_bins,)
        np.full(N_POS_BINS, day_of_training[t], dtype=np.float32),
        time_since_start[t],
        np.full(N_POS_BINS, reward_avail[t], dtype=np.float32),
    ], axis=0)  # (4, n_bins)
    input_trials.append(inp)
```

iii. Implicit: all streams are placed on the common position-bin grid, which the agent treats as the common time base (CONVERSION_NOTES Step 5, Key Decision 1).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `sess#` field of the experiment-info entry, stored as `sess_num` when the session record is first created in `get_unique_sessions`. No date information is used, even though `datexp` is available in the same record and in the session key.

ii.
```python
sessions[key] = {
    'mname': s['mname'], 'datexp': s['datexp'], 'blk': s['blk'],
    ...
    'sess_num': s.get('sess#', 0),
    ...
}
```
```python
day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: "sess# | input[1]: day_of_training | Direct mapping from exp_info | Continuous, per-trial". The agent's own reasoning acknowledges the mismatch: "The sess# field in exp_info gives the session number (0=before learning, 1=after learning). But 'day of training' is more nuanced. I'll use sess# as a proxy." Later (Step 10) it notes "Day of training goes up to 12... Need to verify" and records only "sess#=0 for unsup_train1_before_learning - correct".

## 4-b. What processing is involved in computing `input` *Day of training*?

i. None. The `sess#` integer is broadcast unchanged across all 60 bins of every trial of the session and stored as `float32`. Because a recording is kept only under the first experiment type it is encountered in, the value taken is that of that first experiment type. Across the 89 sessions the realised values are 0 (21 sessions), 1 (61), 2 (3), 3 (1), 6 (1), 10 (1), 12 (1), giving an input range of [0, 12]. Within a mouse the sequence is not monotonic in date — e.g. DR10's six sessions (2022-07-12, 07-19, 07-21, 07-28, 07-29, 07-30) receive 0, 1, 1, 0, 1, 1.

ii.
```python
day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
...
np.full(N_POS_BINS, day_of_training[t], dtype=np.float32),
```

iii. As above: `sess#` is used as a proxy for day of training with no further processing, and CONVERSION_NOTES Step 11/README describe it as "Session number (0 = before learning, 1 = after learning, etc.)".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No raw variable at all. It is derived purely from the position-bin index and the two constants `BIN_SIZE_M = 0.1` m and `VR_SPEED = 0.6` m/s, so the vector is byte-identical for every trial of every session.

ii.
```python
def make_time_since_trial_start(n_trials, n_bins=60):
    time_per_bin = BIN_SIZE_M / VR_SPEED
    times = (np.arange(n_bins) * time_per_bin).astype(np.float32)
    return np.tile(times, (n_trials, 1))
```

iii. CONVERSION_NOTES Step 5 mapping table: "Position bin index | input[2]: time_since_trial_start | bin_idx * time_per_bin | Continuous, time-varying". The justification is again constant VR speed: on the position grid, elapsed VR time is a fixed function of the bin index.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `bin_index × 0.16667 s`, tiled over trials, stored as `float32`. Range is [0.0, 9.8] s in every session (bin 59 → 9.833 s). It measures elapsed *VR* time from corridor entry, i.e. it excludes any wall-clock time during which the mouse stopped and the VR was frozen, and it has zero variance across trials.

ii.
```python
time_since_start = make_time_since_trial_start(n_trials, N_POS_BINS)
...
time_since_start[t],
```

iii. Same as 5-a: the agent's rationale is that with a constant-speed corridor the position axis is a linear time axis, so the time since trial start is fully determined by bin index.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined on the same 60 position bins per trial as the neural array, with bin 0 = corridor entry = 0 s, so it is aligned by construction.

ii.
```python
inp = np.stack([
    time_to_cue[t],
    np.full(N_POS_BINS, day_of_training[t], dtype=np.float32),
    time_since_start[t],
    np.full(N_POS_BINS, reward_avail[t], dtype=np.float32),
], axis=0)
```

iii. Implicit in the position-grid design; `off_start = 0.0` and `off_end = 10.0` in the metadata record the same alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, the boolean per-trial flag marking trials run in the rewarded corridor.

ii.
```python
reward_avail = beh['isRew'].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: "isRew | input[3]: reward_availability | Direct boolean->float | Discrete, per-trial". Step 10 Check verifies it is all-False for unsupervised mice (DR10) and ≈0.44 for a task mouse (TX108).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Only a cast from bool to `float32` and a broadcast across the trial's 60 bins. Realised range [0, 1]; it is identically 0 for the unsupervised, naive and grating sessions, and mixed 0/1 for the task sessions.

ii.
```python
np.full(N_POS_BINS, reward_avail[t], dtype=np.float32),
```

iii. No processing needed — the flag is directly available (CONVERSION_NOTES Step 9 consistency table: "Reward availability | Present for task mice | [0, 1] | Yes").

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the wall texture of each trial) and `beh['UniqWalls']` (the textures used in that session). A global label set is built beforehand by `collect_all_stim_names`, which reads every `Beh_*.npy` file and takes the sorted union of all `UniqWalls` entries — 15 names: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

ii.
```python
def collect_all_stim_names(exp_info):
    all_stim = set()
    ...
            for session_key, dat in beh.items():
                for wn in dat['UniqWalls']:
                    all_stim.add(str(wn))
    return sorted(list(all_stim))
```
```python
def get_stimulus_category(wall_name, uniq_walls):
    category_names = list(uniq_walls)
    categories = np.array([category_names.index(wn) for wn in wall_name], dtype=np.int64)
    return categories, category_names
```

iii. CONVERSION_NOTES Step 5 mapping table: "WallName | output[0]: visual_stimulus | Map to global category index | 15 categories", and Key Decision 7: "15 stimulus categories: All unique wall names across all sessions". `WallName` is preferred over `stim_id`/`WallType` (the reference notebook explicitly says to ignore `WallType`).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's wall name is first indexed against the session's own `UniqWalls`, then remapped in `main()` to the index into the 15-name global list, and broadcast as a constant across all 60 bins of the trial as `int64`. No grouping of crops or spatial shuffles into base textures is done, so `circle1`, `circle2`, `circle3` are three distinct classes and `leaf1`, `leaf1_swap1`, `leaf1_swap2` are three more. Since each session uses only 2–4 of the 15 names, most classes are absent within any given session; chance level for the decoder becomes 1/15 = 0.0667.

ii.
```python
stim_categories, stim_names = get_stimulus_category(wall_name, uniq_walls)
...
np.full(N_POS_BINS, stim_categories[t], dtype=np.int64),
```
```python
local_stim_names = result['stim_names']
global_stim_map = {name: all_stim_names.index(name) for name in local_stim_names}
for t in range(result['n_trials']):
    local_cat = int(result['output'][t][0, 0])
    local_name = local_stim_names[local_cat]
    result['output'][t][0, :] = global_stim_map[local_name]
```

iii. CONVERSION_NOTES Step 9 lists "Stimuli | circle, leaf, rock, brick + variants | 15 categories | Yes". The agent noted in the trajectory that "the chance level for visual_stimulus is 0.0667 = 1/15 because we have 15 global stimulus categories, but this session only uses 2... The balanced accuracy accounts for this."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickPos']` (position stamp of each lick) and `beh['LickTrind']` (trial stamp of each lick), together with `beh['Corridor_Length']` for the bin edges. The frame-indexed alternative `beh['LickFr']` is not used.

ii.
```python
lick_raster = make_lick_raster(
    beh['LickPos'], beh['LickTrind'], n_trials, N_POS_BINS, corridor_len)
```

iii. CONVERSION_NOTES Step 5 mapping table: "LickPos/LickTrind | output[1]: licking | Binary raster at position bins | Binary, time-varying". Position-stamped licks are the natural choice given that the target grid is a position grid; the reference `utils.get_lick_raster` also builds its raster from `LickPos`/`LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A `(n_trials, 60)` zero array is filled with 1 wherever at least one lick falls, by looping over every lick event, reading its trial index and position, locating the position bin with `np.searchsorted` on `np.linspace(0, 60, 61)`, clipping the bin to [0, 59], and setting that cell to 1. Licks whose trial index is out of range are skipped. The result is cast to `int64`. Across the full dataset 1.5% of bins are marked as licking; sessions of unsupervised and naive mice are all-zero, task sessions run 2–26% per session.

ii.
```python
def make_lick_raster(lick_pos, lick_trind, n_trials, n_bins=60, corridor_len=60.0):
    lick_raster = np.zeros((n_trials, n_bins), dtype=np.float32)
    bin_edges = np.linspace(0, corridor_len, n_bins + 1)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        if tr < 0 or tr >= n_trials:
            continue
        bin_idx = np.searchsorted(bin_edges, pos, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        lick_raster[tr, bin_idx] = 1.0
    return lick_raster
```

iii. CONVERSION_NOTES Step 10 Check 4: "Licking check: 0 licks for unsupervised mouse (DR10) - expected. Task mice (TX108) have 3589 lick events. PASS." Step 10 Issues: "Licking very sparse (1.5%) - expected for dataset with many unsupervised/naive mice."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick raster is built directly on the same 60 position bins used for the neural array (bin edges span 0–`Corridor_Length` = 0–60 decimetres, i.e. corridor entry to the end of the grey space), and row *t* corresponds to trial *t* of the neural array, so it is aligned by construction. Note that a lick emitted while the VR was stationary is still assigned to the position bin the mouse was in, whereas the neural data for such frames were excluded from the interpolation source.

ii.
```python
out = np.stack([
    np.full(N_POS_BINS, stim_categories[t], dtype=np.int64),
    lick_raster[t].astype(np.int64),
    pos_output[t].astype(np.int64),
    speed_output[t].astype(np.int64),
], axis=0)  # (4, n_bins)
output_trials.append(out)
```

iii. Implicit in the common position-grid design.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. No raw variable is read. Position is taken to be the position-bin index itself, which is exact by construction because the neural data were warped onto the position axis (`ft_PosCum/Corridor_Length`). `beh['ft_Pos']` is never used.

ii.
```python
pos_output = make_position_output(n_trials, N_POS_BINS)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Position bins | output[2]: position | 4 bins of 1m each in corridor | Categorical, time-varying". The reasoning is that after position interpolation the bin index *is* the VR position, so no separate lookup is needed.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A `(n_trials, 60)` array is filled by bin index only, identically for every trial and every session: bins 0–9 → 0, 10–19 → 1, 20–29 → 2, 30–59 → 3. Cast to `int64`. Because it is a pure function of the bin index, the per-session distribution is exactly (0.167, 0.167, 0.167, 0.500) in all 89 sessions and the variable carries no trial-to-trial information.

ii.
```python
def make_position_output(n_trials, n_bins=60):
    pos_output = np.zeros((n_trials, n_bins), dtype=np.float32)
    for b in range(n_bins):
        if b < 10:
            pos_output[:, b] = 0
        elif b < 20:
            pos_output[:, b] = 1
        elif b < 30:
            pos_output[:, b] = 2
        else:  # bins 30-59 (3-4m corridor + grey space)
            pos_output[:, b] = 3
    return pos_output
```

iii. Docstring and CONVERSION_NOTES Step 5: 4 equal 1 m bins over the 4 m corridor, as required by the Decoder Task.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Ten position bins (1 m) per category for the first three categories; the fourth category absorbs bins 30–59, i.e. the last 1 m of textured corridor *plus* the entire 2 m grey space. Category labels are `['0-1m', '1-2m', '2-3m', '3-4m']`. The consequence is that 20/60 = 33% of all samples are grey-space samples labelled "3-4m", and class 3 holds 50% of the data against 16.7% for each of the others.

ii.
```python
        else:  # bins 30-59 (3-4m corridor + grey space)
            pos_output[:, b] = 3
```
```python
'output_values': [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
```

iii. The agent identified the problem explicitly and chose to keep it: CONVERSION_NOTES Step 10 Issues Found: "Position output imbalance (50% in bin 3 due to grey space) - inherent to including grey space in trial", and Step 12 lists position accuracy (0.372) as "Moderate - position is deterministic so could be higher". The trajectory records it weighed excluding the grey space or giving it a fifth category and concluded "grey space (bins 40-60) gets assigned to the last position bin since the mouse is past the corridor. This is what I already do, but it creates imbalance... Let me keep all 60 bins but document the position imbalance. The decoder should handle this."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Trivially aligned: the label of bin *b* is a function of *b*, and the neural column *b* is by construction the activity at that VR position, for the same trial index.

ii.
```python
pos_output[t].astype(np.int64),
```

iii. Implicit in the position-grid design.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['run_pos']`, the authors' pre-computed running speed already interpolated into a (trials × 60 positions) array. `beh['ft_RunSpeed']` (the per-frame speed) is not used.

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60)
speed_output = make_speed_output(run_pos, speed_bin_edges)
```

iii. CONVERSION_NOTES Step 5 mapping table: "run_pos | output[3]: running_speed | Quartile bins (global) | Categorical, time-varying". `run_pos` is already on the same trials × position-bin grid as the converted neural data, so no resampling is needed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A pre-pass (`compute_global_speed_quartiles`) reads every `Beh_*.npy` file once, concatenates every session's `run_pos.ravel()` (99 behaviour entries, i.e. including the duplicate swap keys), keeps only finite and strictly positive values, and takes the 25th/50th/75th percentiles of that pooled distribution. The resulting edges `[0, 18.00, 30.65, 45.38, inf]` are then used for every session. Per-trial values are assigned with `np.digitize` against the three interior edges; values ≤ 0 (backward ball movement) fall in bin 0. Cast to `int64`.

ii.
```python
def compute_global_speed_quartiles(exp_info):
    all_speeds = []
    ...
            if 'run_pos' in dat:
                all_speeds.append(dat['run_pos'].ravel())
    all_speeds = np.concatenate(all_speeds)
    valid = np.isfinite(all_speeds) & (all_speeds > 0)
    all_speeds = all_speeds[valid]
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    bin_edges = np.array([0, quartiles[0], quartiles[1], quartiles[2], np.inf])
    return bin_edges
```
```python
def make_speed_output(run_pos, speed_bin_edges):
    speed_output = np.digitize(run_pos, speed_bin_edges[1:-1]).astype(np.float32)
    return speed_output
```

iii. Trajectory: "the task says 'Running speed discretized into 4 bins, each corresponding to 25% of the data'. This means quartiles across ALL data, which requires a global pass." CONVERSION_NOTES Step 5 Key Decision 6: "Speed quartiles: Computed globally across all sessions." Step 10 Check 5 notes the negative values and declares them "handled by quartile binning".

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three fixed global thresholds — 18.00, 30.65, 45.38 (units of `run_pos`) — give categories labelled `['Q1','Q2','Q3','Q4']`. Pooled over the whole dataset the realised fractions are 0.271 / 0.254 / 0.241 / 0.233, close to but not exactly 25% each because zero/negative samples were excluded when the percentiles were estimated but are then assigned to Q1. Per session the distribution is far from uniform, because `run_pos` scale differs greatly between sessions (median 0.02 for DR10_2022_07_12_1 versus 51.7 for TX123_2023_12_21_1): one session lands 99.7% of its samples in Q1, and several land 60–69% in Q4.

ii.
```python
print(f"  Speed quartile edges: {bin_edges}")   # [0. 18.00140878 30.64536835 45.37690418 inf]
speed_output = np.digitize(run_pos, speed_bin_edges[1:-1]).astype(np.float32)
```
```python
'speed_bin_edges': speed_bin_edges.tolist(),
```

iii. CONVERSION_NOTES Step 9 consistency table records "Running speed | Quartiles | Q1-Q4 (25% each globally) | Yes". The agent saw the per-session skew in the sample run ("Session 1 has 99.6% in Q1... the global quartiles don't fit well") but did not change the approach.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is supplied by the authors already interpolated onto the same trials × 60-position-bin grid that the neural interpolation produces, so row *t*, column *b* of the speed output corresponds to trial *t*, column *b* of the neural array with no further alignment work.

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60)
speed_output = make_speed_output(run_pos, speed_bin_edges)
...
speed_output[t].astype(np.int64),
```

iii. Implicit in the position-grid design; the agent's Step 10 Check 6(c) claims the temporal alignment matches the reference `spk_pos_interp`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four guards exist. (1) Behaviour can run past the imaging, so `ft_move` and `ft_PosCum` are truncated to the number of imaged frames (`beh[...][:n_frames]`), exactly as the reference notebook does. (2) An assertion checks that the retinotopy vector has the same length as the concatenated spike matrix. (3) Lick events with an out-of-range trial index are skipped and lick bin indices are clipped to [0, 59]. (4) Non-finite and non-positive speeds are excluded when the global speed percentiles are estimated. Beyond that, nothing: trials that fall outside the imaged range are not detected (`np.interp` silently clamps to the edge value, producing a constant trial), NaNs in `run_pos` would be silently binned as Q4 by `np.digitize`, `ft_trInd` NaNs are never consulted, and there is no fallback if a behaviour key or file is missing other than raising `ValueError`.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
ft_pos_cum = beh['ft_PosCum'][:n_frames]
vr_moving = ft_move > 0
```
```python
assert len(iarea) == n_neurons_total, \
    f"Retinotopy mismatch: {len(iarea)} vs {n_neurons_total} neurons"
```
```python
        if tr < 0 or tr >= n_trials:
            continue
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
```
```python
valid = np.isfinite(all_speeds) & (all_speeds > 0)
```

iii. CONVERSION_NOTES Step 10 Check 5 addresses the negative running speeds ("backward movement - handled by quartile binning. PASS") and Check 3 the sound-cue range ("[4.2, 36.0] ≈ expected [5, 35]. Slight variation expected"). The truncation to `n_frames` is copied from the reference code. The agent's Step 10 conclusion is that the dataset is clean and no further handling is needed.

## 12-a. What are the most time-consuming steps of the code?

i. Three things dominate. (1) Reading the spike files — 405 GB across 89 files, 1.5–9 GB each, plus the `np.concatenate` of the per-plane arrays, which doubles peak memory for the session. (2) The position interpolation, timed explicitly at 6–39 s per session (~40–70% of the per-session time reported by the script). (3) Writing the 410.69 GB pickle at the end. The two pre-passes over all behaviour files (stimulus names, speed quartiles) add a fixed startup cost. Reported total: 3189 s ≈ 53 min for the full run, over the 15-minute target in the instructions.

ii.
```python
t_interp = time.time()
interp_spk = position_interpolate_spk(
    spk[:, vr_moving], ft_pos_cum[vr_moving], corridor_len, n_trials, N_POS_BINS)
print(f"    Interpolation took {time.time()-t_interp:.1f}s")
...
elapsed = time.time() - t0
print(f"    Session processed in {elapsed:.1f}s")
```
```python
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. CONVERSION_NOTES Step 7 estimates "~50s for 2 sessions, estimated ~40 min for full 89 sessions"; Step 6 records "Optimized interpolation using np.interp (4x speedup over scipy interp1d)". The agent accepted the ~40 min estimate and proceeded without further optimisation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The dominant one is the per-neuron `np.interp` loop inside `position_interpolate_spk` — up to ~79,000 scalar `np.interp` calls per session; the whole batch shares one source axis, so a single `np.searchsorted` on `src_pos` plus one gather-and-blend would interpolate all neurons at once. The `for i in range(len(lick_pos))` loop in `make_lick_raster` could be replaced by one `np.searchsorted` on the whole lick vector plus a fancy-index assignment. The `for b in range(n_bins)` loop in `make_position_output` builds a constant pattern that could be one `np.clip(np.arange(60)//10, 0, 3)` broadcast. The two `for t in range(n_trials)` loops that build the per-trial input and output stacks, and the per-trial stimulus remapping loop in `main`, all re-slice arrays that already exist in stacked form. The outer batching loop (`batch_size = 2000`) exists only to bound memory and does not itself help.

ii.
```python
    for i in range(0, n_neurons, batch_size):
        end_i = min(i + batch_size, n_neurons)
        batch = raw_spk[i:end_i]
        for s in range(batch.shape[0]):
            interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
```
```python
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        ...
        lick_raster[tr, bin_idx] = 1.0
```
```python
    for b in range(n_bins):
        if b < 10:
            pos_output[:, b] = 0
        ...
```
```python
    for t in range(result['n_trials']):
        local_cat = int(result['output'][t][0, 0])
        local_name = local_stim_names[local_cat]
        result['output'][t][0, :] = global_stim_map[local_name]
```

iii. The agent documented only "Optimized interpolation using np.interp (4x speedup over scipy interp1d)" (CONVERSION_NOTES Step 6) and did not identify the remaining scalar loops; the reference `utils.spk_pos_interp` also loops per neuron, so the agent treated the loop as matching the reference.

## 12-c. What processing does the code repeat multiple times?

i. `get_unique_sessions(exp_info)` is re-executed from scratch on every call to `load_beh_for_session`, i.e. once per session in addition to the call in `main`. The behaviour files are read three separate times over the run: once by `collect_all_stim_names`, once by `compute_global_speed_quartiles`, and once during conversion (the third pass is cached, the first two are not shared). `compute_global_speed_quartiles` also pools 99 behaviour entries, so the duplicated swap keys of a session contribute their `run_pos` twice to the quartile estimate. Inside `process_session`, `np.full(N_POS_BINS, ...)` rebuilds the constant `day_of_training` and `reward_availability` vectors once per trial, and `make_time_since_trial_start` tiles the same 60-value vector `n_trials` times even though it never varies. The stimulus label is written once by `get_stimulus_category` and then overwritten a second time in `main` during the local→global remap.

ii.
```python
def load_beh_for_session(session_key, exp_info):
    if session_key in _beh_cache:
        return _beh_cache[session_key]
    sessions = get_unique_sessions(exp_info)     # recomputed for every session
```
```python
all_stim_names = collect_all_stim_names(exp_info)     # pass 1 over all beh files
speed_bin_edges = compute_global_speed_quartiles(exp_info)  # pass 2 over all beh files
```
```python
  Loaded 99 sessions for speed computation        # 99 > 89: swap keys counted twice
```
```python
    return np.tile(times, (n_trials, 1))
```

iii. CONVERSION_NOTES Step 6 claims the opposite ("Efficient stimulus name and speed quartile collection (load each beh file once)"), which is true within each helper but not across them; the repetition across the three passes and the per-session `get_unique_sessions` call are not documented.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item is precision and extent of the neural array. All ~46,000 neurons per session × 60 bins × 38,110 trials are stored as `float32`, producing a 410.69 GB pickle; `float16` would halve it, and the decoder projects the population down to at most ~2,000 components/neurons anyway, so the stored precision and full width are never used at that resolution. The agent evaluated `float16` and dropped the idea. Second, 20 of the 60 bins per trial are grey space, which carries no visual texture and receives a position label that is not a corridor position — a third of the interpolated neural volume (~137 GB) exists only to be mislabelled. Third, several arrays are built that are constant by construction and could be a single shared vector: `make_time_since_trial_start` tiles an identical row for every trial in every session, and `make_position_output` materialises an identical (n_trials, 60) label matrix for every session. Fourth, `interp_value` (the scipy `interp1d` wrapper copied from the reference `utils.py`) is defined but never called, and `EXCLUDED_AREAS` is defined but never used. Fifth, `region_names` is built inside `get_brain_region_idx` and discarded. Sixth, the per-trial stimulus remap loop rewrites the whole 60-element row when only a scalar changed.

ii.
```python
interp_spk = np.zeros((n_neurons, n_trials * n_bins), dtype=np.float32)
...
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```
```python
Saved /app/converted_data.pkl (410.69 GB)
```
```python
def interp_value(v, vind, tind):          # never called
    Model_ = interpolate.interp1d(vind, v, fill_value='extrapolate')
    return Model_(tind)

EXCLUDED_AREAS = [-1, 7]                  # never referenced
```
```python
def get_brain_region_idx(iarea):
    region_names = ['V1', 'mHV', 'lHV', 'aHV']   # unused local
```

iii. Not documented as waste. CONVERSION_NOTES Step 9 simply records "converted_data.pkl: 410.69 GB" and Step 10 accepts the grey-space position imbalance as "inherent to including grey space in trial". The trajectory shows the agent computed the projected size (~427 GB), considered `float16` and neuron subsampling, then decided the decoder "handles large neural data" and kept `float32`.
