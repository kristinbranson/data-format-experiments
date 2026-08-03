# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `Imaging_Exp_info.npy` is read first to enumerate all sessions. Behavior files (`Beh_<exp_type>.npy`) are loaded per session (not grouped by behavior file). Neural data and retinotopy are loaded per session.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# Per session:
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
beh = load_behavior_for_session(session_key, exp_type, ndb)
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. The AI loads data similarly to the reference, reading from the same three directories and the same master index file. The main difference is that behavior files are reloaded per session rather than grouped and loaded once per behavior file.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in each exp_info entry. Subjects are the sorted unique mouse names across all sessions.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. Same approach as reference — subjects are derived from the `mname` field in the experiment info.

## 1-c. How are the data split into sessions?

i. A session is uniquely identified by `mname_datexp_blk`. The AI deduplicates sessions that appear under multiple experiment types by keeping the first occurrence. This yields 89 unique sessions.

ii.
```python
session_map = {}
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
        if key not in session_map:
            session_map[key] = (exp_type, ndb)
```

iii. Same deduplication logic as reference — each physical recording is used only once.

## 1-d. How are the data split into trials?

i. The AI defines trials using the frame range from `StartFr` (corridor entry) to `GrayFr` (grey space entry), rounding these fractional frame numbers to the nearest integer. This produces variable-length trials. The reference instead uses `ft_trInd` and `ft_CorrSpc` to find corridor frames per trial, then takes the first `N_FRAMES=32` frames, padding shorter trials.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
# ...
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The AI chose to use the explicit trial boundary markers (StartFr, GrayFr) rather than the per-frame trial labels (ft_trInd, ft_CorrSpc). This is a reasonable but different approach that results in variable-length trials rather than fixed-length trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI skips trials where `StartFr < 0`, `GrayFr > n_frames`, `StartFr >= GrayFr`, or where the trial has fewer than 2 frames. The reference drops trials that have no frames remaining after intersecting `ft_trInd` and `ft_CorrSpc` within the imaging range.

ii.
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. The AI's filtering is slightly different — it validates frame range bounds directly rather than relying on the data's own labeling. The CONVERSION_NOTES report 1 trial skipped across all sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of per-plane arrays concatenated into a single neurons-by-frames matrix. The visual area of each neuron comes from `iarea` in retinotopy files.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = np.load(os.path.join(root, fn), allow_pickle=True)['iarea']
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed (no dF/F, no deconvolution needed — already deconvolved). The AI extracts the frame range `StartFr:GrayFr` for each trial and stores as float16. Trials have variable length. The reference instead uses `ft_CorrSpc` frames and pads to a fixed length of 32.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The AI noted the data is already deconvolved fluorescence traces, consistent with the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL neurons, including those outside the four visual areas (iarea -1 and 7, mapped to "other"). The reference drops neurons outside V1, mHV, lHV, and aHV.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
# ... maps specific iarea values to 0-3 ...
# No filtering step — all neurons kept
```

iii. The AI's CONVERSION_NOTES state "No neuron filtering/curation in the reference code - all Suite2p-detected neurons used", but this is incorrect — the reference code does filter by brain region. The AI includes a 5th "other" region instead of dropping those neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). The AI uses `StartFr` (rounded to int) as the start of the trial window. The reference uses `ft_CorrSpc` and `ft_trInd` to find corridor frames, which effectively aligns to the first frame labeled as being in the corridor for that trial.

ii.
```python
sfr = start_frs[trial_idx]  # np.round(beh['StartFr']).astype(int)
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr]
```

iii. Both approaches align to corridor entry, but via different mechanisms.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The imaging frames are the bins at ~3.17 Hz (~315 ms per frame). The AI computes the bin size from the median of frame timestamp differences.

ii.
```python
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
# ...
'time_bin_size': float(median_dt * 1000),  # in ms
```

iii. Same as reference — no rebinning applied.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame at which the sound cue was played) and the frame indices of the trial.

ii.
```python
sound_frs = beh['SoundFr']
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Same source variable as reference (`SoundFr`), but the computation method differs.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes the time difference as `(SoundFr - frame_index) * dt_sec`, treating frame indices as integers and multiplying by a single median dt_sec. The reference interpolates SoundFr onto the actual frame time axis using `np.interp`, then computes `cue_time - frame_time`.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The AI uses a simpler arithmetic approach. The reference's interpolation approach is more precise since frame times aren't perfectly uniform, but the difference is small.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`sfr:gfr`) used for the neural data extraction, so it is inherently aligned.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Same alignment principle as reference — both use the same frames for all data streams.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session entry, which contains the date string (e.g., `2023_01_05`). The AI parses these into datetime objects.

ii.
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_sessions[mname].append((sess_key, date))
```

iii. The reference uses the sorted session order to count recording days (0, 1, 2, ...), not calendar dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes calendar day differences from each mouse's first recording date. So if a mouse was recorded on days 1, 8, 15, the values would be 0, 7, 14. The reference counts session order: 0, 1, 2, etc.

ii.
```python
for mname, sessions in mouse_sessions.items():
    sessions.sort(key=lambda x: x[1])
    first_date = sessions[0][1]
    for sess_key, date in sessions:
        day_map[sess_key] = (date - first_date).days
```

iii. The AI's CONVERSION_NOTES say "calendar day difference from mouse's first recording date". This is a different interpretation than the reference's session-count approach. The CONVERSION_NOTES report a range of 0-92 days, while the reference would produce 0-7 at most.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (corridor entry frame) and the frame indices of the trial.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Same source variable as reference.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_index - StartFr) * dt_sec` using a single median dt value. The reference interpolates StartFr onto the frame time axis using `np.interp`, then computes `frame_time - start_time`.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Simpler arithmetic vs. interpolation. Small numerical differences expected.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame indices as the neural data extraction.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
```

iii. Same alignment principle as all other streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks trials in the rewarded corridor.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Same as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float32: 1.0 if rewarded, 0.0 if not. Broadcast to all frames.

ii.
```python
np.full(n_trial_frames, rew_val, dtype=np.float32)
```

iii. Same processing as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name on each trial's corridor walls.

ii.
```python
stim_name = wall_names[trial_idx]
# Later:
stim_names, stim_to_idx = build_stimulus_mapping(session_results)
```

iii. Same source variable as reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses all 15 unique wall names as separate stimulus categories (e.g., circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, etc.). The reference maps all 15 names to 4 base textures (circle, leaf, rock, wood) using a lookup table.

ii.
```python
stim_names = sorted(all_stim_names)  # all 15 unique WallName values
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```

iii. The AI does not group stimulus variants. The instructions say "Visual stimulus category, e.g. circle1, leaf2, etc." which could be interpreted either way, but the reference groups them into 4 base categories. The paper describes 4 texture categories (circle, leaf, rock, wood).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick in the session, and `LickTrind` (referenced but not strictly needed).

ii.
```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. Same source variable as reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are rounded to the nearest integer (AI uses `np.round`). A binary array is created where 1 indicates at least one lick at that frame. The reference truncates with `.astype(int)` (floor) rather than rounding.

ii.
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
frame_offsets = lick_frs_int[mask] - start_fr
lick_binary[frame_offsets] = 1.0
```

iii. Minor difference: round vs truncate for fractional lick frame numbers. Also, the AI filters licks by the trial's frame range (StartFr to GrayFr) rather than building a session-wide licking array and indexing into it.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are filtered to the same `StartFr:GrayFr` range used for the neural data, so alignment is inherent.

ii.
```python
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. Same alignment principle — licking is indexed by the same frames as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[sfr:gfr]
```

iii. Same source variable as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 4 bins using `np.digitize` with bin edges at `[0, 10, 20, 30, 40]` dm. The reference uses `ft_Pos // 10` clipped to 0-3.

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```

iii. Both approaches divide the 40 dm corridor into 4 equal 10 dm bins. The AI uses `np.digitize` while the reference uses integer division. Results should be very similar but may differ at bin boundaries.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter (10 dm) bins: 0-10, 10-20, 20-30, 30-40 dm. Values outside are clipped to [0, 3].

ii.
```python
bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)  # [0, 10, 20, 30, 40]
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. Same 4-bin scheme as reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one position per imaging frame, extracted with the same frame range `sfr:gfr` as the neural data.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
```

iii. Same alignment principle as other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. Same source variable as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes speed quartiles globally across all corridor frames in the dataset using `np.percentile`, then applies `np.digitize` to bin speeds. The reference computes rank-based quartiles per session using argsort, ensuring exactly 25% of frames fall in each bin within each session.

ii.
```python
def compute_speed_bin_edges(all_speeds):
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
    return quartiles

def discretize_speed(speed, quartiles):
    binned = np.digitize(speed, quartiles, right=True)
    binned = np.clip(binned, 0, 3)
    return binned
```

iii. Two major differences: (1) global vs per-session quartiles, and (2) percentile-based vs rank-based binning. The rank-based approach handles ties (e.g., many frames at speed=0) better. The AI's CONVERSION_NOTES acknowledge a skewed distribution issue (Q1=30.2%) that they tried to fix with `right=True`.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile bins based on global percentile thresholds. The reference uses rank-based quartiles per session.

ii.
```python
quartiles = np.percentile(valid, [25, 50, 75])
binned = np.digitize(speed, quartiles, right=True)
```

iii. The AI's approach resulted in uneven bin distributions (30.2%, 19.8%, 24.9%, 25.1%) while the reference guarantees 25% per bin per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one speed per imaging frame, extracted with the same frame range `sfr:gfr`.

ii.
```python
trial_speed = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles)
```

iii. Same alignment principle.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses `min(n_frames_neural, n_frames_beh)` to handle behavior running past imaging. Trials with invalid frame ranges or fewer than 2 frames are skipped. Licks outside the frame range are ignored.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
# ...
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
```

iii. Similar to the reference, which also cuts to the number of imaged frames. The AI reports 1 trial skipped across the full dataset.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (large neural data files) and the initial speed quartile computation (which loads all behavior files).

ii.
```python
# Speed collection requires loading all behavior files:
def collect_all_corridor_speeds(session_map, max_sessions=None):
    for sess_key, (exp_type, ndb) in sessions:
        beh = load_behavior_for_session(sess_key, exp_type, ndb)
        # ...
```

iii. The AI's code has an additional bottleneck: loading all behavior files twice — once for speed quartile computation and once during session processing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_session` constructs arrays individually for each trial. The speed quartile collection iterates over all sessions loading behavior files.

ii.
```python
for trial_idx in range(ntrials):
    # ... per-trial processing ...
```

iii. Similar to reference — the per-trial loop is the main candidate for vectorization but is minor compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The AI loads behavior files twice: once in `collect_all_corridor_speeds` to compute global speed quartiles, and once in the main processing loop. The reference does not have this duplication since it computes speed quartiles per-session.

ii.
```python
# First pass:
all_speeds = collect_all_corridor_speeds(session_map)
# Second pass:
for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(...)
```

iii. This doubles the behavior file I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes neurons from the "other" brain region (iarea -1, 7) which are outside visual cortex. The reference drops these neurons. Including them increases data size and may not contribute meaningfully to decoding visual/behavioral variables. The AI also computes global speed quartiles across all sessions when per-session quartiles would be simpler and avoid the extra data loading pass.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
```

iii. The "other" neurons add to data size (the AI's converted file is 148 GB vs the reference's ~110 GB) without clear benefit for decoding visual stimuli and behavioral variables.
