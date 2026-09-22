# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, deduplicates recordings by `mname_datexp_blk`, then processes each unique recording by loading one behavior dictionary file, one spike file, and optionally one retinotopy file. Behavior files are cached by filename, but sessions are still handled one recording at a time.

ii. 
```python
def get_all_unique_recordings():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    seen = set()
    recordings = []
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                recordings.append((key, s, exp_type))
    return recordings
```
```python
beh = load_beh_for_recording(key, exp_type)
spk = load_spk(mname, datexp, blk)
iarea = load_retino(mname, datexp)
```

iii. In `CONVERSION_NOTES.md`, the AI states that there are 23 experiment types, 142 total entries, and 89 unique recordings, and describes the behavior files as shared across experiment types while spike and retinotopy are session-specific.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The output `subjects` list is built in first-seen order while sessions are assigned `subject_idx` from that list.

ii. 
```python
mname = result['mname']
if mname not in subject_to_idx:
    subject_to_idx[mname] = len(subjects)
    subjects.append(mname)
...
all_subject_idx.append(subject_to_idx[mname])
```

iii. The AI's notes explicitly say the dataset contains 19 mice and treat `mname` as the subject identifier throughout.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` recording. Duplicate appearances across experiment types are removed by keeping the first occurrence of a session key.

ii. 
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in seen:
    seen.add(key)
    recordings.append((key, s, exp_type))
```

iii. `CONVERSION_NOTES.md` says there are 89 unique recordings and that behavioral data may repeat across experiment types, so the AI chose to deduplicate at the session-key level.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd == trial`, but only frames that are both inside the textured corridor (`ft_CorrSpc`) and classified as running (`ft_move > 0`) are kept. Each trial therefore becomes a variable-length set of running corridor frames rather than all corridor-entry-to-exit frames.

ii. 
```python
corridor_mask = ft_CorrSpc.copy()
running_mask = beh['ft_move'][:n_use] > 0

for trial in range(ntrials):
    trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
    trial_frames = np.where(trial_mask)[0]
```

iii. The notes and trajectory justify this as matching the paper's statement that analyses used only running timepoints and only the 4 m textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if fewer than 2 kept frames survive the running-plus-corridor filter. Sessions with fewer than 2 surviving trials are skipped entirely. The AI does not implement the reference's global long-trial outlier filter.

ii. 
```python
trial_frames = np.where(trial_mask)[0]

if len(trial_frames) < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The only explicit rationale in the artifacts is practical decoder compatibility: the target format requires at least two trials per session, and the code implicitly requires at least two timepoints per trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from `spks` in each session's spike file. Brain-region labels are derived from `iarea` in the retinotopy file.

ii. 
```python
data = np.load(fn, allow_pickle=True).item()
return np.concatenate(data['spks'], 0)
```
```python
return np.load(fn, allow_pickle=True)['iarea']
```

iii. The notes say the neural data are Suite2p deconvolved fluorescence traces and that retinotopy is used for brain-area assignment.

## 2-b. How is the `neural` data processed?

i. The AI concatenates imaging planes across neurons, selects only running corridor frames for each trial, and stores the resulting per-trial matrices as `float32`. It does not compute dF/F or do additional temporal processing.

ii. 
```python
return np.concatenate(data['spks'], 0)
```
```python
neural = spk[:, trial_frames].astype(np.float32)
```

iii. In the notes, the AI says no dF/F is needed because the reference data already contain deconvolved traces, and it keeps `float32` "for decoder compatibility."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not drop neurons outside the four named visual areas. Instead it assigns those neurons to an extra `"unassigned"` category and keeps them all; if retinotopy is missing, every neuron is labeled `"unassigned"`.

ii. 
```python
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
...
region_idx = np.full(len(iarea), 4, dtype=np.int64)
for ia_code, reg_idx in IAREA_TO_REGION.items():
    region_idx[iarea == ia_code] = reg_idx
```
```python
brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
```

iii. The notes justify this with "All neurons included - no d-prime filtering," arguing the decoder should use all available information and that the paper's selectivity filtering was analysis-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata says trials are aligned to corridor entry, but the actual retained neural samples begin at the first running frame inside the corridor for each trial. Alignment is therefore implicit through the selected frame indices, not by explicitly preserving all frames from `StartFr`.

ii. 
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
```
```python
trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
trial_frames = np.where(trial_mask)[0]
neural = spk[:, trial_frames].astype(np.float32)
```

iii. The notes repeatedly describe the alignment event as corridor entry, but the trajectory shows the AI deliberately chose running-only corridor frames to match the paper's running-timepoint analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame as the time bin, with size estimated from the median `ft` difference per session. No temporal rebinning is applied.

ii. 
```python
dt = float(np.median(np.diff(ft)) * 86400)
...
median_dt_ms = float(np.median(dt_values) * 1000)
...
'time_bin_size': median_dt_ms,
```

iii. The notes state the frame rate is about 3.18 Hz, or about 315 ms per frame, and list "Native frame rate: ~3.18 Hz, no temporal rebinning" as a key decision.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and from the frame timing implied by `ft` through a single per-session frame duration estimate `dt`.

ii. 
```python
sound_fr = beh['SoundFr']
dt = float(np.median(np.diff(ft)) * 86400)
```
```python
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. The notes map `SoundFr` to `time_to_sound_cue` in seconds and treat the native imaging frame interval as the time base.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every kept frame in a trial, the AI subtracts the cue frame index from the retained frame index and multiplies by `dt`. This produces a continuous signed offset in seconds, negative before the cue and positive after it.

ii. 
```python
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. The trajectory discusses representing the cue relative to frame time in seconds; the implemented choice is frame-index arithmetic using the median frame interval.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed directly on the same `trial_frames` indices used to slice the neural matrix, so it has exactly the same timepoints as the per-trial neural data.

ii. 
```python
neural = spk[:, trial_frames].astype(np.float32)
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. The AI consistently uses the kept frame indices as the common alignment grid for neural, input, and output variables.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata fields in `Imaging_Exp_info.npy`, preferring `days` when present and otherwise using `sess#`.

ii. 
```python
def get_training_day(db_info):
    if 'days' in db_info:
        return float(db_info['days'])
    if 'sess#' in db_info:
        return float(db_info['sess#'])
    return 0.0
```

iii. The trajectory shows the AI inspected `days` and `sess#` across experiment types and chose to use those fields directly rather than reconstructing day order from sorted session IDs.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The chosen scalar session value is broadcast across all kept time bins in each trial. There is no per-mouse reindexing or recomputation from dates.

ii. 
```python
training_day = get_training_day(db_info)
...
day_arr = np.full(n_t, training_day, dtype=np.float32)
```

iii. The notes explicitly map "`sess#/days` -> day_of_training" and treat it as a per-trial input array repeated over timepoints.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the implemented code it is derived from the retained frame indices themselves, specifically each kept frame's index relative to the first kept frame in the trial, scaled by `dt`. It does not use `StartFr`.

ii. 
```python
dt = float(np.median(np.diff(ft)) * 86400)
...
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. The notes describe this input as "frame - start" in seconds, but the code operationalizes "start" as the first retained running corridor frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI subtracts `trial_frames[0]` from every retained frame index and multiplies by `dt`, yielding a nonnegative time axis starting at zero for the first retained sample.

ii. 
```python
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. The trajectory framed this as continuous time since trial onset, but the implemented shortcut avoids interpolating the fractional `StartFr` values.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same `trial_frames` used for neural slicing, so it has the same number of time bins and is aligned one-to-one with the neural columns.

ii. 
```python
neural = spk[:, trial_frames].astype(np.float32)
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. The AI uses a shared frame-index grid for each trial's neural and non-neural time-varying signals.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`, the per-trial rewarded-corridor indicator.

ii. 
```python
is_rew = beh['isRew']
...
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. The notes map `isRew` directly to `reward_availability` with no extra transformation beyond numeric conversion.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial boolean into a float and broadcasts it across all kept time bins in the trial.

ii. 
```python
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. The decision is justified in the notes as a straightforward binary per-trial contextual variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, the per-trial corridor wall texture label.

ii. 
```python
wall_name = beh['WallName']
...
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
```

iii. The trajectory shows the AI enumerated all unique `WallName` values across the dataset before defining a global mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI treats all 15 observed wall names as separate categories, maps them with a global lookup table, and broadcasts the resulting per-trial code across all time bins of the trial.

ii. 
```python
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2',
                      'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap1',
                      'wood1_swap2', 'wood2', 'wood5'])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
```
```python
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` lists "15 stimulus categories mapped globally" as a key decision.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`. `LickTrind` groups licks by trial; `LickFr` provides the lick frame positions.

ii. 
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
...
for li in range(len(lick_fr)):
    lick_by_trial[int(lick_trind[li])].append(lick_fr[li])
```

iii. The notes identify `LickFr` as the core source, while the implementation adds `LickTrind` to relocate each lick within the trial-specific retained frame set.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each lick in the current trial, the AI finds the nearest retained frame in `trial_frames`; if the lick is within one frame of that retained sample, it sets that time bin to 1. All other bins are 0.

ii. 
```python
lick_binary = np.zeros(n_t, dtype=np.int64)
if trial in lick_by_trial:
    for lf in lick_by_trial[trial]:
        diffs = np.abs(trial_frames - lf)
        closest = np.argmin(diffs)
        if diffs[closest] < 1.0:
            lick_binary[closest] = 1
```

iii. The trajectory shows the AI was concerned that running-only filtering would otherwise drop many licks, so it explicitly snapped lick times to the nearest kept frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the neural data through the same `trial_frames` index set used for slicing neural activity. Alignment is by nearest retained frame, not by directly flagging the original frame index on the full imaging grid.

ii. 
```python
neural = spk[:, trial_frames].astype(np.float32)
...
diffs = np.abs(trial_frames - lf)
closest = np.argmin(diffs)
if diffs[closest] < 1.0:
    lick_binary[closest] = 1
```

iii. The AI's justification follows from its running-only trial representation: once non-running frames are removed, licks have to be reassigned to the surviving frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the framewise position trace.

ii. 
```python
ft_Pos = beh['ft_Pos'][:n_use]
...
pos = ft_Pos[trial_frames]
```

iii. The notes describe `ft_Pos` as corridor position in decimeters and map it directly to the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides position by 10, floors it, clips the result to `[0, 3]`, and uses those four integer bins as the position categories.

ii. 
```python
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes justify this as converting the 4 m textured corridor into four 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed at 0, 10, 20, 30, and 40 decimeters, implemented as four 1 m bins after division by 10 and clipping.

ii. 
```python
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes explicitly say "`ft_Pos / 10 -> position | 4 bins of 1m`."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed with the same `trial_frames` used for the neural data, so its timepoints match the retained neural samples exactly.

ii. 
```python
neural = spk[:, trial_frames].astype(np.float32)
pos = ft_Pos[trial_frames]
```

iii. The AI uses one common retained-frame grid for all trial-varying streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_use]
...
speed = ft_RunSpeed[trial_frames]
```

iii. The notes map `ft_RunSpeed` directly to the running-speed output before discretization.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes three global percentile boundaries over all running corridor frames in all recordings. Then each kept trial frame is assigned to one of four bins using those global thresholds.

ii. 
```python
running_corridor = beh['ft_CorrSpc'][:n] & (beh['ft_move'][:n] > 0)
corridor_speeds = beh['ft_RunSpeed'][:n][running_corridor]
...
bins = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed = ft_RunSpeed[trial_frames]
speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
```

iii. The notes say this was done on running corridor frames "to ensure 25% per bin."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are defined by the global 25th, 50th, and 75th percentiles of running speed across the retained running corridor frames, then applied with `np.digitize`.

ii. 
```python
bins = np.percentile(all_speeds, [25, 50, 75])
...
speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
```

iii. The trajectory shows the AI explicitly debugged this thresholding logic and decided to use percentile cutoffs with `digitize`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained `trial_frames` indices used for the neural data.

ii. 
```python
neural = spk[:, trial_frames].astype(np.float32)
speed = ft_RunSpeed[trial_frames]
```

iii. The AI's general alignment rule is that all time-varying outputs share the neural frame grid after corridor/running filtering.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates framewise behavioral arrays to `n_use = min(n_frames, len(beh['ft']))`, skips sessions whose behavior cannot be found, allows missing retinotopy by labeling all neurons `"unassigned"`, and drops trials with fewer than 2 retained frames.

ii. 
```python
beh = load_beh_for_recording(key, exp_type)
if beh is None:
    return None
...
n_use = min(n_frames, len(beh['ft']))
```
```python
brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
```
```python
if len(trial_frames) < 2:
    continue
```

iii. The notes mention handling off-by-one behavior/spike mismatches with `min(n_frames, n_beh)` and describe the dataset as otherwise clean.

## 12-a. What are the most time-consuming steps of the code?

i. The code suggests two main expensive steps: loading and concatenating the large spike files for every session, and the full-dataset pre-pass that computes global running-speed quartiles before conversion.

ii. 
```python
spk = load_spk(mname, datexp, blk)
...
t_load = time.time() - t0
```
```python
speed_bins = compute_running_speed_quartiles(all_recordings)
```

iii. The code prints separate load and processing times per session and separately logs the quartile computation pass, indicating those were treated as the main bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain non-vectorized: iterating through every recording to collect speed data, every trial in every session to build outputs, and every lick within a trial to find the nearest retained frame.

ii. 
```python
for key, db_info, exp_type in recordings:
    ...
```
```python
for trial in range(ntrials):
    ...
```
```python
for lf in lick_by_trial[trial]:
    diffs = np.abs(trial_frames - lf)
    closest = np.argmin(diffs)
```

iii. The AI does not explicitly justify these loops beyond general practicality; its notes emphasize matching the reference processing more than optimizing vectorization.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats several dataset-wide or session-wide passes: it reads behavior repeatedly through per-recording lookup instead of grouping all sessions by behavior file for final conversion, clears and refills the behavior cache, and scans the full dataset once just to compute speed quartiles before scanning it again to build the converted output.

ii. 
```python
speed_bins = compute_running_speed_quartiles(all_recordings)
_beh_cache.clear()
...
for i, (key, db_info, exp_type) in enumerate(recordings):
    result = process_session(key, db_info, exp_type, speed_bins)
```
```python
if (i + 1) % 10 == 0:
    _beh_cache.clear()
```

iii. The notes justify the global speed pre-pass as necessary to get balanced quartile bins across the retained running frames.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI keeps all neurons including `"unassigned"` ones, stores neural data in `float32`, computes and stores a 15-class visual-stimulus coding instead of the coarser 4-class grouping used in the reference, and optionally supports plotting helper outputs that are not needed by downstream decoder training.

ii. 
```python
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
...
neural = spk[:, trial_frames].astype(np.float32)
```
```python
'output_values': [
    ALL_STIMULI,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
],
```
```python
if show_processing:
    create_processing_plots(data)
```

iii. The notes justify the extra stimulus granularity and inclusion of all neurons as preserving information for the decoder, but those choices increased output size and diverged from the leaner reference representation.
