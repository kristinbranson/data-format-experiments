# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads experiment metadata from `Imaging_Exp_info.npy`, builds a unique session map from that metadata, caches behavior dictionaries by experiment type, and then loads per-session neural and retinotopy files while iterating over the unique sessions.

ii.
```python
exp_info = np.load(
    os.path.join(data_root, 'beh', 'Imaging_Exp_info.npy'),
    allow_pickle=True
).item()

session_map = get_session_beh_mapping(exp_info, os.path.join(data_root, 'beh'))
all_sessions = sorted(session_map.keys())

if exp_type not in beh_cache:
    beh_path = os.path.join(data_root, 'beh', f'Beh_{exp_type}.npy')
    beh_cache[exp_type] = np.load(beh_path, allow_pickle=True).item()

spk = load_spk(mname, datexp, blk, root=os.path.join(data_root, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(data_root, 'retinotopy'))
```

iii. In `CONVERSION_NOTES.md`, the agent says the raw data are split across behavior, spike, retinotopy, and experiment-info files, and that each unique recording should be used once. The trajectory also shows the agent explored all four sources before writing the loader.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse-name prefix of each session key, and `subject_idx` is built from a sorted list of unique mouse IDs.

ii.
```python
all_mice = sorted(set(k.split('_')[0] for k in all_sessions))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
...
subject_idx_list.append(mouse_to_idx[mname])
```

iii. The notes state there are 89 recordings in 19 mice and that sessions are keyed as `mouse_date_block`. The trajectory shows the agent explicitly counted 19 unique mouse names from those keys.

## 1-c. How are the data split into sessions?

i. One unique recording key `mname_datexp_blk` is treated as one session. If the same recording appears in multiple experiment-type lists, the code keeps only one entry, preferring the metadata row with the greatest number of non-NaN `stim_id` values.

ii.
```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
...
if key not in session_map:
    session_map[key] = (exp_type, beh_key, ndb)
else:
    old_ndb = session_map[key][2]
    old_nstim = np.sum(~np.isnan(old_ndb['stim_id'].astype(float)))
    new_nstim = np.sum(~np.isnan(ndb['stim_id'].astype(float)))
    if new_nstim > old_nstim:
        session_map[key] = (exp_type, beh_key, ndb)
```

iii. `CONVERSION_NOTES.md` says “Each unique recording = one session” and that duplicate metadata entries share the same underlying recording. The trajectory shows the agent discovered 33 recordings appearing in multiple experiment types and chose a deduplication rule based on “more stimuli”.

## 1-d. How are the data split into trials?

i. Trials are taken from `beh['ntrials']`. Neural data are interpolated into an array shaped `(n_neurons, ntrials, 60)`, then each trial is extracted by indexing the second axis. Inputs and outputs are also constructed by iterating `for trial in range(ntrials)`.

ii.
```python
ntrials = beh['ntrials']
...
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
...
for trial in range(ntrials):
    neural_trial = interp_spk[:, trial, :].astype(np.float32)
```

iii. The notes say the agent followed the reference spatial-interpolation helper that returns neurons x trials x positions. The trajectory shows the agent read the `get_interpPos_spk` helper from the reference `utils.py`.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. Trials are only excluded indirectly if a whole session is skipped, and sessions are skipped if they have fewer than 2 trials, missing neural/retinotopy files, or fewer than 10 valid neurons.

ii.
```python
if ntrials < 2:
    print(f"    Skipping: only {ntrials} trials")
    skipped_sessions.append(sess_key)
    continue
...
if n_valid < 10:
    print(f"    Skipping: only {n_valid} valid neurons")
    skipped_sessions.append(sess_key)
    continue
```

iii. The notes emphasize session-level sanity checks such as “All sessions have >= 2 trials” and valid retinotopy matches. The trajectory does not show the agent identifying any additional trial-level curation rule in the paper or code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the per-session `spks` arrays in the neural `.npy` file, plus `ft_move` and `ft_PosCum` from behavior for selecting running frames and aligning/interpolating them, plus `iarea` from retinotopy for neuron filtering and region labels.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
...
valid_neuron_mask = (iarea != -1) & (iarea != 7)
...
VRmove = beh['ft_move'][:nfr] > 0
ft_AcumPos = beh['ft_PosCum'][:nfr]
```

iii. The notes explicitly say the neural source is deconvolved fluorescence from `spks`, filtered by retinotopy and interpolated using running-frame behavior variables. The trajectory shows the agent inspected the shapes of `spks` and `iarea` and the behavior keys.

## 2-b. How is the `neural` data processed?

i. The code concatenates imaging planes, filters neurons by retinotopy, keeps only running frames, interpolates each neuron across cumulative position into 60 spatial bins per trial, casts each trial to `float32`, and replaces any NaN/Inf values with zero.

ii.
```python
spk_filtered = spk[valid_neuron_mask]
...
interp_spk = get_interpPos_spk(
    spk_filtered[:, VRmove],
    ft_AcumPos[VRmove],
    ntrials,
    n_bins=int(CL),
    lengths=CL
)
...
neural_trial = interp_spk[:, trial, :].astype(np.float32)
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. `CONVERSION_NOTES.md` says the neural processing follows the reference `utils.py` helpers for concatenation and position interpolation and cites the paper statement that only running timepoints were analyzed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by `iarea`, excluding `-1` and `7`. Sessions are skipped if they have fewer than 10 remaining neurons. Running-frame selection `ft_move > 0` also removes non-running frames before interpolation.

ii.
```python
valid_neuron_mask = (iarea != -1) & (iarea != 7)
n_valid = valid_neuron_mask.sum()
...
if n_valid < 10:
    print(f"    Skipping: only {n_valid} valid neurons")
    skipped_sessions.append(sess_key)
    continue
...
VRmove = beh['ft_move'][:nfr] > 0
```

iii. The notes justify the `iarea` exclusion as “outside visual cortex” and cite the paper’s “only considered timepoints during running” sentence. The `<10 valid neurons` session cutoff appears to be the agent’s own safety threshold rather than something cited from the reference pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start / corridor entry by interpolating along cumulative position and reshaping into trial x position bins, so each trial starts at corridor position 0 and runs through the full 60 decimeter bins.

ii.
```python
linPos = np.arange(0, new_shape[0], 1 / new_shape[1])
...
interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos)
...
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
```

iii. The script header and metadata both say “trial start (corridor entry)”. The notes say alignment is position 0 at corridor entry, and the trajectory shows the agent relying on the paper/instructions wording “Temporally aligned based on trial start (corridor entry).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script treats each 1-dm spatial bin as `1/6` s, so the effective bin size is 166.67 ms. No frame-based temporal rebinning is done; instead, neural data are spatially interpolated into 60 bins.

ii.
```python
n_bins = 60  # spatial bins per trial (40 corridor + 20 gray)
time_bin_sec = 1.0 / 6.0  # 1 dm at 6 dm/s = 166.67 ms
...
'time_bin_size': time_bin_sec * 1000,
```

iii. The notes justify this by the constant 60 cm/s VR speed during running and explicitly call the output “position-based binning” rather than direct time binning. The trajectory shows the agent deciding to use spatial rather than frame-time bins because trial durations vary.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundPos']` plus the synthetic 0-59 spatial-bin positions that define the aligned trial axis.

ii.
```python
SoundPos = beh['SoundPos']  # sound cue position in dm
...
positions = np.arange(n_bins)
sound_pos = SoundPos[trial]
time_to_cue = (sound_pos - positions) * time_bin_sec
```

iii. The notes state that `SoundPos` is the cue position per trial and that the paper says cue positions are uniformly drawn between 0.5 m and 3.5 m.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every trial and every aligned bin, the code computes signed distance from the current bin to the cue location and converts that distance to seconds by multiplying by `1/6`. Positive values mean before cue, negative values after cue.

ii.
```python
time_to_cue = (sound_pos - positions) * time_bin_sec  # shape (60,)
```

iii. The notes explain the same formula and the sign convention. The trajectory shows the agent reasoning that fixed VR speed makes position-to-time conversion possible after spatial interpolation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same 60-bin per-trial position axis as the neural data, so there is one cue-time value for each neural bin.

ii.
```python
input_trial = np.stack([
    time_to_cue.astype(np.float32),
    day_array,
    time_since_start.astype(np.float32),
    reward_avail
], axis=0)  # shape (4, 60)
```

iii. The agent’s notes say all inputs are expressed on the same 60 spatial bins used for the neural interpolation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` fields in `Imaging_Exp_info.npy`. The code parses the session date string and groups dates by mouse.

ii.
```python
key = f'{ndb["mname"]}_{ndb["datexp"]}_{ndb["blk"]}'
date = datetime.strptime(ndb['datexp'], '%Y_%m_%d')
mouse_dates[ndb['mname']].append(date)
session_dates[key] = (ndb['mname'], date)
```

iii. The trajectory shows the agent considering explicit day/session fields, but the final notes say it used “days since the mouse’s first recording session” computed from metadata dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script finds the earliest recording date and subtracts it from each session date to get a calendar-day offset, then repeats that scalar across all bins of every trial in the session.

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
...
session_days[key] = (date - mouse_first_date[mname]).days
...
day_array = np.full(n_bins, day, dtype=np.float32)
```

iii. The notes justify this as a session-level constant reflecting training day. In the trajectory, the agent debated other sources of “day” but ended up using date differences.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the aligned spatial-bin index rather than directly from timestamps. The relevant raw assumptions are corridor length and constant VR speed during running.

ii.
```python
positions = np.arange(n_bins)
time_since_start = positions * time_bin_sec
```

iii. The notes say this is computed from position because the neural data were already re-expressed on a position grid and each bin corresponds to `1/6` s during running.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code multiplies the bin number `0..59` by `1/6` s to create a monotonically increasing per-bin time axis from the aligned trial start.

ii.
```python
time_since_start = positions * time_bin_sec  # shape (60,)
```

iii. The notes explicitly state the result ranges from 0 to about 9.83 s.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction because the same 60 bins used for neural interpolation are used to define `positions`.

ii.
```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
...
time_since_start = positions * time_bin_sec
```

iii. The notes describe the full dataset as sharing one common 60-bin trial grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `beh['isRew']`, the trial-level boolean for rewarded versus unrewarded corridor.

ii.
```python
isRew = beh['isRew']
...
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The notes say reward availability is “from `beh['isRew']`” and explain that unsupervised mice therefore get all zeros.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial-level boolean is converted to float and broadcast to all 60 bins of the trial as a constant signal.

ii.
```python
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
```

iii. The notes justify this as matching the decoder spec: a discrete per-trial indicator of rewarded corridor identity.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`, not from `stim_id`.

ii.
```python
WallName = beh['WallName']
...
stim_idx = stim_to_idx[str(WallName[trial])]
```

iii. The notes say stimulus categories are identified by their names such as `circle1`, `leaf2`, and `wood1_swap2`. The trajectory shows the agent examining both `stim_id` and `WallName`, then choosing names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script builds a global sorted vocabulary of all stimulus names across all behavior files, maps each trial’s `WallName` to an integer index, and repeats that category across all 60 bins of the trial.

ii.
```python
all_stim_names_set = set()
...
for name in np.unique(beh_data[beh_key_inner]['WallName']):
    all_stim_names_set.add(str(name))
all_stim_names = sorted(all_stim_names_set)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
```

iii. The notes justify this as creating the “full stimulus list across all sessions.” The trajectory shows the agent fixing an earlier bug so sample runs would still use the global stimulus vocabulary.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh['LickPos']` and `beh['LickTrind']`.

ii.
```python
lick_spatial = make_lick_spatial_bins(
    beh['LickPos'], beh['LickTrind'], ntrials, n_bins=int(CL)
)
```

iii. The notes say licking is mapped from `LickPos` and `LickTrind` into spatial bins. The trajectory shows the agent inspecting those arrays directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code allocates an `(ntrials, 60)` zero array, loops over licks, floors each lick position to a decimeter bin, clips to `[0, 59]`, and sets that bin to 1 for the corresponding trial.

ii.
```python
lick_arr = np.zeros((ntrials, n_bins), dtype=np.int64)
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(np.clip(np.floor(pos), 0, n_bins - 1))
    if 0 <= tr < ntrials:
        lick_arr[tr, bin_idx] = 1
```

iii. The notes justify the binarization because the decoder output is supposed to be binary licking, not lick counts.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the same 60 spatial bins per trial as the neural data.

ii.
```python
lick_trial = lick_spatial[trial, :].astype(np.int64)  # shape (60,)
output_trial = np.stack([
    stim_category,
    lick_trial,
    pos_trial,
    speed_trial
], axis=0)
```

iii. The notes explicitly describe licking as “binary, time-varying” on the same spatial-bin axis as the neural trials.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the synthetic aligned position-bin index `0..59`, together with the corridor structure assumed by `n_bins=60`.

ii.
```python
n_bins = 60
positions = np.arange(n_bins)
```

iii. The notes describe the data as 60 bins per trial, with 40 corridor bins and 20 gray-space bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code digitizes the 60 aligned positions using cut points at 10, 20, 30, and 40 decimeters, then copies that same category vector into every trial.

ii.
```python
position_bins_edges = [10, 20, 30, 40]
...
pos_category = np.digitize(positions, position_bins_edges).astype(np.int64)
...
pos_trial = pos_category.copy()
```

iii. The notes justify this as four 1 m corridor segments plus an additional gray-space segment.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into five categories: `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)`, and `[40,60)` decimeters, with the last category labeled `gray_space`.

ii.
```python
# Bins: [0-10) -> 0, [10-20) -> 1, [20-30) -> 2, [30-40) -> 3, [40-60) -> 4 (gray)
position_bins_edges = [10, 20, 30, 40]
position_values = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray_space']
```

iii. `CONVERSION_NOTES.md` explicitly states that the agent added a fifth gray-space category beyond the four 1 m corridor bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by using the same 60-bin trial grid that the neural data use after interpolation.

ii.
```python
neural_trial = interp_spk[:, trial, :].astype(np.float32)
...
pos_trial = pos_category.copy()  # shape (60,)
```

iii. The notes repeatedly describe all modalities as living on the same 60-bin per-trial representation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['run_pos']`, the `(ntrials, 60)` running-speed array already expressed on position bins.

ii.
```python
run_pos = beh['run_pos']    # (ntrials, 60) running speed at each position
```

iii. The notes say the running-speed output comes from `beh['run_pos']`. The trajectory shows the agent inspecting `run_pos` statistics before finalizing this choice.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first concatenates all `run_pos` values from all sessions, computes the 25th/50th/75th percentiles globally, then applies `np.digitize` per session to turn each 60-bin speed trace into quartile labels.

ii.
```python
all_speeds = np.concatenate([rp.ravel() for rp in run_pos_all])
all_speeds = all_speeds[~np.isnan(all_speeds)]
edges = np.percentile(all_speeds, [25, 50, 75])
...
result = np.digitize(run_pos, edges)
```

iii. The notes justify global quartiles by the decoder specification “4 bins, each corresponding to 25% of the data.” The trajectory shows the agent correcting an earlier sample-run bug so the quartiles used all sessions, not just the sampled ones.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into four bins using global quartile cutpoints, producing integer categories `0, 1, 2, 3`.

ii.
```python
def speed_to_bins(run_pos, edges):
    result = np.digitize(run_pos, edges)  # 0, 1, 2, 3
    return result.astype(np.int64)
```

iii. The notes say each bin should contain about 25% of the global speed data and the labels are built from the estimated percentile edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The code assumes `run_pos` is already on the same `(ntrials, 60)` position grid used by the neural interpolation, so it can be stacked directly as a trial-aligned output.

ii.
```python
run_pos = beh['run_pos']    # (ntrials, 60) running speed at each position
speed_bins = speed_to_bins(run_pos, speed_edges)  # (ntrials, 60)
...
speed_trial = speed_bins[trial, :].astype(np.int64)
```

iii. The notes describe `run_pos` as “running speed at each spatial position,” which is why the agent considered it already aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles a few cases defensively: it skips sessions with missing neural or retinotopy files, skips sessions with too few trials or too few valid neurons, removes NaNs before estimating speed quartiles, and replaces NaN/Inf neural values with zeros on a per-trial basis.

ii.
```python
except Exception as e:
    print(f"    Skipping: could not load neural data: {e}")
...
all_speeds = all_speeds[~np.isnan(all_speeds)]
...
if np.any(np.isnan(neural_trial)) or np.any(np.isinf(neural_trial)):
    neural_trial = np.nan_to_num(neural_trial, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes say “No NaN or Inf values in final neural arrays (cleaned during conversion)” and emphasize data-integrity checks. The trajectory shows the agent also checked `run_pos` for NaNs before finalizing the speed discretization.

## 12-a. What are the most time-consuming steps of the code?

i. The heavy steps are loading the huge spike matrices, neuron-by-neuron interpolation inside `spk_pos_interp`, and the full first pass over all sessions just to collect `run_pos` for speed quartiles.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
...
for s in range(raw_spk.shape[0]):
    spk_resh.append(np.reshape(
        interp_value(raw_spk[s, :], accum_pos / corridorLen, linPos),
        (int(new_shape[0]), int(new_shape[1]))
    ))
...
for sess_key in all_sessions_full:
    ...
    run_pos_all.append(beh['run_pos'])
```

iii. The notes call out the very large dataset size and the trajectory explicitly says the full conversion is dominated by interpolation of 89 sessions with roughly 50K neurons each.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the neuron loop in `spk_pos_interp`, the lick-event loop in `make_lick_spatial_bins`, the trial loop that repeatedly stacks arrays, and the repeated Python loops used to gather global stimulus names and speed arrays.

ii.
```python
for s in range(raw_spk.shape[0]):
    ...

for i in range(len(lick_pos)):
    ...

for trial in range(ntrials):
    ...
```

iii. The trajectory shows the agent recognized the interpolation cost but did not rewrite the reference-style helper. The code stays close to the original helper rather than optimizing it.

## 12-c. What processing does the code repeat multiple times?

i. It makes two passes over the sessions, loads behavior files once for `beh_cache` and again when collecting all stimulus names, rebuilds constant per-trial arrays inside the trial loop, and duplicates a subset of the full dataset again when saving `sample_data.pkl`.

ii.
```python
for sess_key in all_sessions_full:
    ...

for exp_type_key in exp_info.keys():
    beh_path = os.path.join(data_root, 'beh', f'Beh_{exp_type_key}.npy')
    ...

day_array = np.full(n_bins, day, dtype=np.float32)
reward_avail = np.full(n_bins, float(isRew[trial]), dtype=np.float32)
stim_category = np.full(n_bins, stim_idx, dtype=np.int64)
...
sample_data = {
    'neural': neural_all[:n_sample],
    ...
}
```

iii. The notes discuss separate “first pass” and “second pass” processing, and the trajectory shows the agent later added an additional global stimulus-vocabulary pass after discovering the sample-run issue.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `n_position_categories` but never uses it, prints extensive summary statistics and region counts that are not consumed by the decoder, stores extra metadata fields that the decoder does not use, and builds/saves `sample_data.pkl`, which is not part of the full downstream training run.

ii.
```python
n_position_categories = 5  # 4 corridor bins + gray space
...
for region_name in brain_regions:
    region_idx = brain_region_map[region_name]
    count = sum(np.sum(br == region_idx) for br in brain_region_idx_all)
    print(f"  {region_name}: {count} neurons total")
...
if sample_file is not None:
    sample_data = {
        ...
    }
```

iii. The trajectory shows the agent was trying to satisfy the deliverable list and add validation outputs, so it kept extra bookkeeping and reporting beyond what `train_decoder.py` needs for the final full-data analysis.
